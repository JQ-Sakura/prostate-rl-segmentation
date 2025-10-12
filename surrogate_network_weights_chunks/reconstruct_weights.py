#!/usr/bin/env python3
"""
Script to reconstruct UNet weights from split chunks
"""

import torch
import os
from pathlib import Path

def reconstruct_weights(chunks_dir: str = "surrogate_network_weights_chunks", 
                       output_path: str = "surrogate_network_weights/unet_backbone_weights.pth"):
    """Reconstruct the original weights file from chunks"""
    
    print("[RECONSTRUCT] Reconstructing UNet weights from chunks...")
    
    # Find all chunk files
    chunk_files = []
    for file in Path(chunks_dir).glob("unet_weights_part_*.pth"):
        chunk_files.append(file)
    
    # Sort by part number
    chunk_files.sort()
    
    if not chunk_files:
        print("[ERROR] No chunk files found!")
        return False
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Load and merge chunks
    merged_weights = {}
    
    for chunk_file in chunk_files:
        print(f"Loading {chunk_file.name}...")
        chunk_weights = torch.load(chunk_file, map_location='cpu')
        merged_weights.update(chunk_weights)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged weights
    torch.save(merged_weights, output_path)
    
    print(f"[SUCCESS] Weights reconstructed successfully!")
    print(f"   Output: {output_path}")
    print(f"   Total tensors: {len(merged_weights)}")
    
    # Verify
    file_size = os.path.getsize(output_path)
    print(f"   File size: {file_size / (1024*1024):.2f} MB")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        chunks_dir = sys.argv[1]
    else:
        chunks_dir = "surrogate_network_weights_chunks"
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = "surrogate_network_weights/unet_backbone_weights.pth"
    
    success = reconstruct_weights(chunks_dir, output_path)
    
    if success:
        print("\n[SUCCESS] Reconstruction complete!")
        print("You can now use the weights with your UNet model.")
    else:
        print("\n[ERROR] Reconstruction failed!")
        sys.exit(1)
