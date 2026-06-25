"""
DDE-LLM: FIXED Encoder v2 - Ensures manifest saves/loads correctly

CRITICAL FIX: channel_offsets must always be a list [0,0,0,0], never None
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class EncodingManifest:
    """Provenance and reconstruction metadata per ENG-DDE-005"""
    layer_id: int
    head_id: int
    weight_type: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    checksum: str
    shape: Tuple[int, ...]
    encoding_method: str = "logarithmic"
    channel_offsets: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    
    def to_dict(self):
        return {
            'layer_id': self.layer_id,
            'head_id': self.head_id,
            'weight_type': self.weight_type,
            'min_val': float(self.min_val),
            'max_val': float(self.max_val),
            'mean_val': float(self.mean_val),
            'std_val': float(self.std_val),
            'checksum': self.checksum,
            'shape': list(self.shape),
            'encoding_method': self.encoding_method,
            'channel_offsets': self.channel_offsets  # Always a list
        }
    
    @classmethod
    def from_dict(cls, d):
        """Safely reconstruct from JSON dict"""
        return cls(
            layer_id=d['layer_id'],
            head_id=d['head_id'],
            weight_type=d['weight_type'],
            min_val=d['min_val'],
            max_val=d['max_val'],
            mean_val=d['mean_val'],
            std_val=d['std_val'],
            checksum=d['checksum'],
            shape=tuple(d['shape']),
            encoding_method=d.get('encoding_method', 'logarithmic'),
            channel_offsets=d.get('channel_offsets', [0.0, 0.0, 0.0, 0.0])
        )

class DDEWeightEncoder:
    """
    Encodes neural network weights as RGBA images using logarithmic scaling.
    v2: Fixed manifest serialization/deserialization
    """
    
    def __init__(self, precision_bits: int = 8):
        self.precision_bits = precision_bits
        self.max_value = 2**precision_bits - 1
        
    def logarithmic_normalize(self, weights: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Logarithmic scaling preserves relative magnitudes"""
        if abs(max_val - min_val) < 1e-10:
            return np.full_like(weights, self.max_value // 2, dtype=np.uint8)
        
        shifted = np.abs(weights - min_val)
        range_val = abs(max_val - min_val)
        
        normalized = self.max_value * (
            np.log1p(shifted) / np.log1p(range_val)
        )
        return np.clip(normalized, 0, self.max_value).astype(np.uint8)
    
    def encode_weight_matrix(self,
                           weights: torch.Tensor,
                           layer_id: int,
                           head_id: int,
                           weight_type: str) -> Tuple[np.ndarray, EncodingManifest]:
        """Encode weight matrix to RGBA tile"""
        
        # 1. Prepare Data
        w = weights.detach().cpu().numpy()
        original_shape = w.shape
        w_flat = w.flatten()
        
        # Stats for manifest
        min_val = float(w_flat.min())
        max_val = float(w_flat.max())
        mean_val = float(w_flat.mean())
        std_val = float(w_flat.std())
        
        # 2. Determine Even Tile Size
        n_weights = len(w_flat)
        tile_size = int(np.ceil(np.sqrt(n_weights)))
        
        # Force even dimensions
        if tile_size % 2 != 0:
            tile_size += 1
            
        # Pad to square
        padding_needed = tile_size**2 - n_weights
        if padding_needed > 0:
            w_flat = np.pad(w_flat, (0, padding_needed), mode='constant', constant_values=0)
        
        # Reshape to square
        w_square = w_flat.reshape(tile_size, tile_size)
        
        # 3. Split into 4 Quadrants
        h2, w2 = tile_size // 2, tile_size // 2
        
        quadrants = [
            w_square[:h2, :w2],   # R
            w_square[:h2, w2:],   # G  
            w_square[h2:, :w2],   # B
            w_square[h2:, w2:]    # A
        ]
        
        # 4. Encode Channels (no entropy balance)
        channels = []
        for quad in quadrants:
            normalized = self.logarithmic_normalize(quad, min_val, max_val)
            channels.append(normalized)
        
        # Stack into RGBA
        rgba_tile = np.stack(channels, axis=-1)
        
        # Create manifest with explicit channel_offsets
        manifest = EncodingManifest(
            layer_id=layer_id,
            head_id=head_id,
            weight_type=weight_type,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
            checksum=self._compute_checksum(w),
            shape=original_shape,
            channel_offsets=[0.0, 0.0, 0.0, 0.0]  # Explicit
        )
        
        return rgba_tile, manifest
    
    def decode_rgba_tile(self, rgba_tile: np.ndarray, manifest: EncodingManifest) -> np.ndarray:
        """Reverse the encoding process"""
        min_val = manifest.min_val
        max_val = manifest.max_val
        range_val = abs(max_val - min_val)
        
        # Decode each channel
        channels_data = rgba_tile.astype(np.float32)
        decoded_quadrants = []
        
        for i in range(4):
            channel = channels_data[:, :, i]
            
            # Reverse logarithmic normalization
            normalized = channel / self.max_value
            decoded = np.expm1(normalized * np.log1p(range_val)) + min_val
            decoded_quadrants.append(decoded)
        
        # Reconstruct from quadrants
        q_h, q_w = decoded_quadrants[0].shape
        full_size = q_h * 2
        reconstructed_square = np.zeros((full_size, full_size), dtype=np.float32)
        
        # Place quadrants
        reconstructed_square[:q_h, :q_w] = decoded_quadrants[0]  # R
        reconstructed_square[:q_h, q_w:] = decoded_quadrants[1]  # G
        reconstructed_square[q_h:, :q_w] = decoded_quadrants[2]  # B
        reconstructed_square[q_h:, q_w:] = decoded_quadrants[3]  # A
        
        # Flatten and crop
        decoded_flat = reconstructed_square.flatten()[:np.prod(manifest.shape)]
        decoded_weights = decoded_flat.reshape(manifest.shape)
        
        return decoded_weights
    
    def _compute_checksum(self, weights: np.ndarray) -> str:
        import hashlib
        return hashlib.sha256(weights.tobytes()).hexdigest()[:16]
    
    def validate_reconstruction(self, original, reconstructed, tolerance=0.10):
        """Measure reconstruction fidelity"""
        mse = np.mean((original - reconstructed)**2)
        mae = np.mean(np.abs(original - reconstructed))
        max_error = np.max(np.abs(original - reconstructed))
        rel_mse = mse / (np.var(original) + 1e-10)
        rel_mae = mae / (np.std(original) + 1e-10)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'max_error': float(max_error),
            'relative_mse': float(rel_mse),
            'relative_mae': float(rel_mae),
            'passed': rel_mae < tolerance
        }


class TransformerToDDE:
    """Convert entire transformer models to DDE photo album format"""
    
    def __init__(self, output_dir: str = "dde_weights"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.encoder = DDEWeightEncoder()
        self.manifests: List[EncodingManifest] = []
        
    def encode_transformer(self, 
                          model: torch.nn.Module,
                          model_name: str = "transformer") -> Dict[str, any]:
        """Encode all weights of a transformer as RGBA tiles"""
        import PIL.Image as Image
        
        tiles = []
        statistics = {
            'total_parameters': 0,
            'total_tiles': 0,
            'encoding_errors': []
        }
        
        # Iterate through model parameters
        for name, param in model.named_parameters():
            if len(param.shape) < 2:
                continue  # Skip biases
            
            # Parse layer info
            parts = name.split('.')
            try:
                layer_id = int([p for p in parts if p.isdigit()][0])
            except (IndexError, ValueError):
                layer_id = 0
            
            weight_type = 'attention' if 'attn' in name else 'mlp' if 'mlp' in name else 'other'
            
            # Encode
            rgba_tile, manifest = self.encoder.encode_weight_matrix(
                param, layer_id, 0, weight_type
            )
            
            # Save with explicit RGBA mode
            tile_filename = f"{model_name}_layer{layer_id}_{weight_type}_{len(tiles)}.png"
            tile_path = self.output_dir / tile_filename
            
            img = Image.fromarray(rgba_tile, mode='RGBA')
            img.save(tile_path, format='PNG')
            
            tiles.append({
                'path': str(tile_path),
                'name': name,
                'shape': list(param.shape),
                'manifest': manifest
            })
            
            self.manifests.append(manifest)
            statistics['total_parameters'] += param.numel()
            statistics['total_tiles'] += 1
            
            # Test reconstruction with 10% tolerance
            decoded = self.encoder.decode_rgba_tile(rgba_tile, manifest)
            original = param.detach().cpu().numpy()
            validation = self.encoder.validate_reconstruction(original, decoded, tolerance=0.10)
            statistics['encoding_errors'].append(validation)
        
        # Save manifest
        manifest_path = self.output_dir / f"{model_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'total_tiles': len(tiles),
                'tiles': [
                    {
                        'path': t['path'],
                        'name': t['name'],
                        'shape': t['shape'],
                        'manifest': t['manifest'].to_dict()
                    }
                    for t in tiles
                ]
            }, f, indent=2)
        
        print(f"✓ Encoded {statistics['total_tiles']} weight matrices")
        print(f"✓ Total parameters: {statistics['total_parameters']:,}")
        print(f"✓ Manifest saved: {manifest_path}")
        
        return {
            'tiles': tiles,
            'manifests': self.manifests,
            'statistics': statistics
        }


if __name__ == "__main__":
    print("DDE-LLM Encoder v2")
    print("Fixed: Manifest serialization and PNG save/load")