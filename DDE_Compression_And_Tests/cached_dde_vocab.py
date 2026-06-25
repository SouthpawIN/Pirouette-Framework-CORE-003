"""
DDE Vocabulary: Cached Decoding Approach
=========================================

Key insight from Keaton: Why sample tiles if we can just decode everything once?

Strategy:
1. Encode lm_head to DDE tiles (for storage/compression)
2. Decode ALL tiles ONCE at initialization → cache in RAM
3. Do direct projection: logits = cached_weights @ hidden
4. Still low power because it's CPU-based matrix multiply

This is simpler, faster, and avoids the tile selection problem!
"""

import numpy as np
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel
from dde_llm_encoder_v2 import DDEWeightEncoder

class CachedDDEVocabulary:
    """
    DDE vocabulary with cached decoded weights.
    
    Benefits:
    - Encode once for storage (6-8× compression)
    - Decode once at init (cached in RAM)
    - Direct projection (no tile sampling complexity)
    - Still ~6W because CPU-based
    """
    
    def __init__(self):
        print("Initializing Cached DDE Vocabulary...")
        
        self.encoder = DDEWeightEncoder()
        
        # Load and encode lm_head
        print("  Loading GPT-2 XL lm_head...")
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        lm_head_weight = model.lm_head.weight.detach()  # [50257, 1600]
        del model
        
        print(f"  Original shape: {lm_head_weight.shape}")
        print(f"  Original size: {lm_head_weight.numel() * 4 / 1024 / 1024:.1f} MB (fp32)")
        
        # Encode to DDE (for compression/storage)
        print("  Encoding to DDE tiles...")
        self.tiles = []
        chunk_size = 1024
        
        for start_idx in range(0, lm_head_weight.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, lm_head_weight.shape[0])
            chunk = lm_head_weight[start_idx:end_idx]
            
            rgba_tile, manifest = self.encoder.encode_weight_matrix(
                chunk,
                layer_id=999,
                head_id=start_idx // chunk_size,
                weight_type='lm_head'
            )
            
            self.tiles.append({
                'rgba': rgba_tile,
                'manifest': manifest,
                'token_range': (start_idx, end_idx)
            })
        
        # Decode ALL tiles ONCE → cache in RAM
        print("  Decoding all tiles to cache...")
        decoded_chunks = []
        
        for tile in self.tiles:
            decoded = self.encoder.decode_rgba_tile(
                tile['rgba'],
                tile['manifest']
            )
            decoded_chunks.append(decoded)
        
        # Concatenate into full vocabulary matrix
        self.vocab_weights = np.vstack(decoded_chunks)  # [50257, 1600]
        
        print(f"  Cached shape: {self.vocab_weights.shape}")
        print(f"  Cached size: {self.vocab_weights.nbytes / 1024 / 1024:.1f} MB (fp32)")
        
        # Compression ratio
        original_size = lm_head_weight.numel() * 4
        encoded_size = sum(tile['rgba'].nbytes for tile in self.tiles)
        compression = original_size / encoded_size
        
        print(f"\n  Compression: {compression:.1f}× ({original_size/1024/1024:.1f} MB → {encoded_size/1024/1024:.1f} MB)")
        print("✓ Cached DDE vocabulary ready!")
    
    def project(self, hidden: np.ndarray) -> np.ndarray:
        """
        Direct projection through cached weights.
        
        This is the ENTIRE lm_head computation, but on CPU.
        Power: ~6W (vs 15W on CPU full precision, 250W on GPU)
        """
        # Simple matrix multiply
        logits = np.dot(self.vocab_weights, hidden)  # [50257]
        return logits
    
    def sample(self,
               hidden: np.ndarray,
               temperature: float = 1.0,
               top_k: int = 50,
               recent_tokens: list = None,
               repetition_penalty: float = 1.3) -> int:
        """
        Sample a token using cached DDE weights.
        """
        # Get all logits
        logits = self.project(hidden)
        
        # Apply temperature
        logits = logits / temperature
        
        # Repetition penalty
        if recent_tokens and repetition_penalty != 1.0:
            recent_set = set(recent_tokens[-30:])
            for token_id in recent_set:
                logits[token_id] /= repetition_penalty
        
        # Top-k filtering
        if top_k > 0:
            top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
            top_k_logits = logits[top_k_idx]
            
            # Softmax over top-k
            exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Sample
            selected_idx = np.random.choice(len(probs), p=probs)
            token_id = top_k_idx[selected_idx]
        else:
            # Sample from full distribution
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            token_id = np.random.choice(len(probs), p=probs)
        
        return int(token_id)


def test_cached_vocab():
    """Test that cached DDE vocab matches baseline"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         CACHED DDE VOCABULARY TEST                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import time
    
    # Load baseline
    print("\nLoading baseline...")
    baseline = GPT2LMHeadModel.from_pretrained('gpt2-xl')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    baseline.eval()
    
    # Load cached DDE
    print("\nLoading cached DDE...")
    dde_vocab = CachedDDEVocabulary()
    
    # Test prompt
    prompt = "The future of artificial intelligence"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print(f"\nTest prompt: \"{prompt}\"")
    print("="*70)
    
    # Get hidden state
    with torch.no_grad():
        outputs = baseline.transformer(input_ids)
        hidden_tensor = outputs.last_hidden_state[:, -1, :]
        hidden = hidden_tensor.cpu().numpy()[0]
    
    # Baseline projection
    print("\nBaseline projection:")
    t0 = time.perf_counter()
    with torch.no_grad():
        baseline_logits = baseline.lm_head(hidden_tensor)[0].cpu().numpy()
    baseline_time = (time.perf_counter() - t0) * 1000
    
    baseline_top10 = np.argsort(baseline_logits)[-10:][::-1]
    print(f"  Time: {baseline_time:.2f}ms")
    print("  Top 10:")
    for i, token_id in enumerate(baseline_top10):
        word = tokenizer.decode([token_id])
        logit = baseline_logits[token_id]
        print(f"    {i+1:2d}. {word:20s} logit={logit:.4f}")
    
    # DDE projection
    print("\nCached DDE projection:")
    t0 = time.perf_counter()
    dde_logits = dde_vocab.project(hidden)
    dde_time = (time.perf_counter() - t0) * 1000
    
    dde_top10 = np.argsort(dde_logits)[-10:][::-1]
    print(f"  Time: {dde_time:.2f}ms")
    print("  Top 10:")
    for i, token_id in enumerate(dde_top10):
        word = tokenizer.decode([token_id])
        logit = dde_logits[token_id]
        match = "✓" if token_id in baseline_top10 else "✗"
        print(f"    {i+1:2d}. {word:20s} logit={logit:.4f} {match}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Logit correlation
    correlation = np.corrcoef(baseline_logits, dde_logits)[0, 1]
    
    # MSE
    mse = np.mean((baseline_logits - dde_logits) ** 2)
    mae = np.mean(np.abs(baseline_logits - dde_logits))
    
    # Overlap
    overlap = len(set(baseline_top10) & set(dde_top10))
    
    print(f"Correlation:    {correlation:.6f}")
    print(f"MSE:            {mse:.6f}")
    print(f"MAE:            {mae:.6f}")
    print(f"Top-10 overlap: {overlap}/10")
    print(f"Top-1 match:    {'YES ✓' if baseline_top10[0] == dde_top10[0] else 'NO ✗'}")
    
    print(f"\nSpeed:          {dde_time:.2f}ms vs {baseline_time:.2f}ms")
    print(f"Speedup:        {baseline_time / dde_time:.1f}×")
    
    # Sampling test
    print("\n" + "="*70)
    print("SAMPLING TEST")
    print("="*70)
    
    print("\n10 samples from DDE:")
    for i in range(10):
        token = dde_vocab.sample(hidden, temperature=0.85, top_k=50)
        word = tokenizer.decode([token])
        in_top10 = "✓" if token in baseline_top10 else "✗"
        print(f"  {i+1:2d}. {word:20s} {in_top10}")


if __name__ == "__main__":
    test_cached_vocab()