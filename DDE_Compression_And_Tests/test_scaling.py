"""
DDE Scaling Test: Does it work across model sizes?
===================================================

Test hybrid DDE on:
- GPT-2 Small (117M params)
- GPT-2 Medium (354M params)
- GPT-2 Large (774M params)
- GPT-2 XL (1.5B params)

Hypothesis: Larger models should work better (cleaner representations)
"""

import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cached_dde_vocab import CachedDDEVocabulary
import numpy as np

class ScalableHybridDDE:
    """Hybrid DDE that works with any GPT-2 size"""
    
    def __init__(self, model_name: str = 'gpt2-xl'):
        print(f"\nInitializing Hybrid DDE for {model_name}...")
        
        self.model_name = model_name
        
        # Tokenizer (same for all GPT-2 models)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Load model
        print(f"  Loading {model_name} transformer (CPU)...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.model = self.model.cpu()
        
        # Get model size
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model size: {total_params / 1e6:.1f}M parameters")
        
        # Build DDE vocab
        print(f"  Encoding {model_name} vocabulary to DDE...")
        self.dde_vocab = self._build_dde_vocab()
        
        print(f"✓ {model_name} Hybrid DDE ready!")
    
    def _build_dde_vocab(self):
        """Build DDE vocab for this specific model"""
        from dde_llm_encoder_v2 import DDEWeightEncoder
        
        encoder = DDEWeightEncoder()
        
        # Get lm_head from model
        lm_head_weight = self.model.lm_head.weight.detach()
        
        print(f"    lm_head shape: {lm_head_weight.shape}")
        
        # Encode to DDE
        tiles = []
        chunk_size = 1024
        
        for start_idx in range(0, lm_head_weight.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, lm_head_weight.shape[0])
            chunk = lm_head_weight[start_idx:end_idx]
            
            rgba_tile, manifest = encoder.encode_weight_matrix(
                chunk,
                layer_id=999,
                head_id=start_idx // chunk_size,
                weight_type='lm_head'
            )
            
            tiles.append({
                'rgba': rgba_tile,
                'manifest': manifest,
                'token_range': (start_idx, end_idx)
            })
        
        # Decode all tiles to cache
        decoded_chunks = []
        for tile in tiles:
            decoded = encoder.decode_rgba_tile(tile['rgba'], tile['manifest'])
            decoded_chunks.append(decoded)
        
        vocab_weights = np.vstack(decoded_chunks)
        
        # Create a simple vocab object
        class SimpleVocab:
            def __init__(self, weights):
                self.vocab_weights = weights
            
            def sample(self, hidden, temperature=1.0, top_k=50, 
                      recent_tokens=None, repetition_penalty=1.3):
                # Project
                logits = np.dot(self.vocab_weights, hidden)
                
                # Temperature
                logits = logits / temperature
                
                # Repetition penalty
                if recent_tokens and repetition_penalty != 1.0:
                    for token_id in set(recent_tokens[-30:]):
                        logits[token_id] /= repetition_penalty
                
                # Top-k
                if top_k > 0:
                    top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
                    top_k_logits = logits[top_k_idx]
                    exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
                    probs = exp_logits / np.sum(exp_logits)
                    selected_idx = np.random.choice(len(probs), p=probs)
                    return int(top_k_idx[selected_idx])
                else:
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    return int(np.random.choice(len(probs), p=probs))
        
        return SimpleVocab(vocab_weights)
    
    def generate(self, prompt: str, max_tokens: int = 40):
        """Generate text"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated = input_ids[0].tolist()
        
        t_start = time.time()
        
        for step in range(max_tokens):
            current_ids = torch.tensor([generated])
            
            with torch.no_grad():
                transformer_outputs = self.model.transformer(current_ids)
                hidden = transformer_outputs.last_hidden_state[:, -1, :].cpu().numpy()[0]
            
            next_token = self.dde_vocab.sample(
                hidden,
                temperature=0.85,
                top_k=50,
                recent_tokens=generated,
                repetition_penalty=1.3
            )
            
            generated.append(next_token)
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        t_end = time.time()
        
        text = self.tokenizer.decode(generated)
        tokens_generated = len(generated) - len(input_ids[0])
        
        return {
            'text': text,
            'tokens': tokens_generated,
            'time': t_end - t_start,
            'tokens_per_sec': tokens_generated / (t_end - t_start)
        }


def test_all_sizes():
    """Test DDE across all GPT-2 sizes"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║           DDE SCALING TEST - ALL GPT-2 SIZES                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    models = [
        ('gpt2', 'GPT-2 Small (117M)'),
        ('gpt2-medium', 'GPT-2 Medium (354M)'),
        ('gpt2-large', 'GPT-2 Large (774M)'),
        ('gpt2-xl', 'GPT-2 XL (1.5B)'),
    ]
    
    test_prompts = [
        "The future of artificial intelligence",
        "Scientists have discovered that",
    ]
    
    results = []
    
    for model_name, display_name in models:
        print(f"\n{'='*70}")
        print(f"TESTING: {display_name}")
        print(f"{'='*70}")
        
        try:
            dde = ScalableHybridDDE(model_name)
            
            model_results = []
            
            for prompt in test_prompts:
                print(f"\nPrompt: \"{prompt}\"")
                
                result = dde.generate(prompt, max_tokens=30)
                
                print(f"Output: {result['text'][:150]}...")
                print(f"Speed: {result['tokens_per_sec']:.1f} tokens/sec")
                
                model_results.append(result)
            
            avg_speed = np.mean([r['tokens_per_sec'] for r in model_results])
            
            results.append({
                'model': display_name,
                'avg_speed': avg_speed,
                'samples': model_results
            })
            
            print(f"\n✓ {display_name}: {avg_speed:.1f} tokens/sec avg")
            
        except Exception as e:
            print(f"\n✗ {display_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Speed (tok/s)':<15} {'Trend'}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['model']:<25} {r['avg_speed']:>10.1f}")
    
    print("\nKey Question: Does coherence improve with model size?")
    print("Expected: Larger models → cleaner representations → better DDE")


if __name__ == "__main__":
    test_all_sizes()