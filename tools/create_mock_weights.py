#!/usr/bin/env python3
"""
Create a mock TinyLlama safetensors file for testing
This creates a minimal weight file with correct structure
"""

import struct
import json
import numpy as np

def create_mock_tinyllama():
    # TinyLlama config (FULL VERSION)
    vocab_size = 32000
    hidden_dim = 2048
    num_layers = 22
    intermediate_dim = 5632
    
    # Create mock weights (small random values)
    np.random.seed(42)
    
    tensors = {}
    data_parts = []
    current_offset = 0
    
    def add_tensor(name, shape):
        nonlocal current_offset
        # Create random FP16 data
        data = np.random.randn(*shape).astype(np.float16)
        data_bytes = data.tobytes()
        
        tensors[name] = {
            "dtype": "F16",
            "shape": list(shape),
            "data_offsets": [current_offset, current_offset + len(data_bytes)]
        }
        
        data_parts.append(data_bytes)
        current_offset += len(data_bytes)
        return data
    
    print("Creating mock TinyLlama weights...")
    
    # Embedding
    print(f"  embed_tokens: [{vocab_size}, {hidden_dim}]")
    add_tensor("model.embed_tokens.weight", (vocab_size, hidden_dim))
    
    # Layer weights
    for i in range(num_layers):
        print(f"  Layer {i+1}/{num_layers}...", end="\r")
        
        # Attention
        add_tensor(f"model.layers.{i}.input_layernorm.weight", (hidden_dim,))
        add_tensor(f"model.layers.{i}.self_attn.q_proj.weight", (hidden_dim, hidden_dim))
        add_tensor(f"model.layers.{i}.self_attn.k_proj.weight", (hidden_dim // 8, hidden_dim))
        add_tensor(f"model.layers.{i}.self_attn.v_proj.weight", (hidden_dim // 8, hidden_dim))
        add_tensor(f"model.layers.{i}.self_attn.o_proj.weight", (hidden_dim, hidden_dim))
        
        # FFN
        add_tensor(f"model.layers.{i}.post_attention_layernorm.weight", (hidden_dim,))
        add_tensor(f"model.layers.{i}.mlp.gate_proj.weight", (intermediate_dim, hidden_dim))
        add_tensor(f"model.layers.{i}.mlp.up_proj.weight", (intermediate_dim, hidden_dim))
        add_tensor(f"model.layers.{i}.mlp.down_proj.weight", (hidden_dim, intermediate_dim))
    
    print()
    
    # Final norm and LM head
    print(f"  final norm and lm_head...")
    add_tensor("model.norm.weight", (hidden_dim,))
    add_tensor("lm_head.weight", (vocab_size, hidden_dim))
    
    # Create header JSON
    header = json.dumps(tensors, separators=(',', ':'))
    header_bytes = header.encode('utf-8')
    header_size = len(header_bytes)
    
    # Write safetensors file
    output_path = "mock_tinyllama.safetensors"
    print(f"\nWriting to {output_path}...")
    
    with open(output_path, 'wb') as f:
        # Header size (8 bytes, little-endian)
        f.write(struct.pack('<Q', header_size))
        
        # Header JSON
        f.write(header_bytes)
        
        # All tensor data
        for data in data_parts:
            f.write(data)
    
    file_size = header_size + 8 + sum(len(d) for d in data_parts)
    print(f"✅ Created {output_path}")
    print(f"   Size: {file_size / 1024 / 1024:.1f} MB")
    print(f"   Tensors: {len(tensors)}")
    print(f"\nTo test:")
    print(f"  adb push {output_path} /data/local/tmp/tinyllama.safetensors")
    print(f"  adb shell 'cd /data/local/tmp && ./llama_inference_real'")

if __name__ == "__main__":
    create_mock_tinyllama()
