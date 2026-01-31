#!/usr/bin/env python3
"""
Merge 3D Parallel Checkpoints

This utility merges checkpoint shards saved during 3D parallel training
into a single full model checkpoint that can be used for inference.

With DP=2, TP=2, PP=2, training saves 4 unique shards:
    - final_model_pp0_tp0.pt  (Stage 0, TP rank 0)
    - final_model_pp0_tp1.pt  (Stage 0, TP rank 1)
    - final_model_pp1_tp0.pt  (Stage 1, TP rank 0)
    - final_model_pp1_tp1.pt  (Stage 1, TP rank 1)
    (DP replicas save identical weights, so we only need one)

This utility:
    1. Loads all shards
    2. Merges TP shards (concatenate column-parallel, stack row-parallel)
    3. Combines PP stages (different layers)
    4. Saves as a HuggingFace-compatible checkpoint

Usage:
    python merge_checkpoints.py --input_dir /path/to/shards --output /path/to/merged.pt
"""

import argparse
import torch
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_shards(input_dir: str, prefix: str = "final_model") -> Dict:
    """Load all checkpoint shards from directory."""
    shards = defaultdict(dict)
    
    for filename in os.listdir(input_dir):
        if filename.startswith(prefix) and filename.endswith('.pt'):
            # Parse rank info from filename: {prefix}_pp{pp}_tp{tp}.pt
            parts = filename.replace('.pt', '').split('_')
            pp_rank = None
            tp_rank = None
            
            for part in parts:
                if part.startswith('pp'):
                    pp_rank = int(part[2:])
                elif part.startswith('tp'):
                    tp_rank = int(part[2:])
            
            if pp_rank is not None and tp_rank is not None:
                path = os.path.join(input_dir, filename)
                checkpoint = torch.load(path, map_location='cpu')
                shards[pp_rank][tp_rank] = checkpoint
                print(f"  Loaded: {filename} (PP={pp_rank}, TP={tp_rank})")
    
    return dict(shards)


def merge_tp_shards(tp_shards: Dict[int, Dict], pp_rank: int) -> Dict[str, torch.Tensor]:
    """
    Merge tensor parallel shards for a single pipeline stage.
    
    Column-parallel layers (c_attn, c_fc): concatenate along output dimension
    Row-parallel layers (c_proj): concatenate along input dimension
    Other layers: just use rank 0's copy
    """
    if len(tp_shards) == 1:
        # No TP, just return the single shard
        return tp_shards[0]['model_state_dict']
    
    tp_size = len(tp_shards)
    merged = {}
    
    # Get all keys from rank 0
    rank0_state = tp_shards[0]['model_state_dict']
    
    for key in rank0_state.keys():
        tensors = [tp_shards[r]['model_state_dict'][key] for r in range(tp_size)]
        
        # Determine merge strategy based on layer type
        if 'c_attn.weight' in key or 'c_fc.weight' in key:
            # Column parallel: out_features is split, so concat along dim 0
            merged[key] = torch.cat(tensors, dim=0)
        elif 'c_attn.bias' in key or 'c_fc.bias' in key:
            # Column parallel bias: concat along dim 0
            merged[key] = torch.cat(tensors, dim=0)
        elif 'c_proj.weight' in key:
            # Row parallel: in_features is split, so concat along dim 1
            merged[key] = torch.cat(tensors, dim=1)
        elif 'c_proj.bias' in key:
            # Row parallel bias: should be identical, use rank 0
            merged[key] = tensors[0]
        else:
            # Non-sharded layers (LayerNorm, embeddings): use rank 0
            merged[key] = tensors[0]
    
    return merged


def merge_pp_stages(pp_stages: Dict[int, Dict[str, torch.Tensor]], config: Dict) -> Dict[str, torch.Tensor]:
    """
    Merge pipeline parallel stages into a single model state dict.
    
    Each stage has different layers:
        - Stage 0: wte, wpe, h.0-h.5, ln_f (if first_and_last)
        - Stage 1: h.6-h.11, ln_f, lm_head
    
    The key mapping depends on how the model was split.
    """
    pp_size = len(pp_stages)
    n_layers = config.get('n_layer', 12)
    layers_per_stage = n_layers // pp_size
    
    merged = {}
    
    for pp_rank, stage_state in sorted(pp_stages.items()):
        layer_offset = pp_rank * layers_per_stage
        
        for key, value in stage_state.items():
            # Map layer indices based on stage
            new_key = key
            
            # Handle transformer blocks: blocks.X.* -> h.{X + offset}.*
            if 'blocks.' in key:
                # Parse block index
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'blocks':
                        block_idx = int(parts[i + 1])
                        global_idx = block_idx + layer_offset
                        parts[i + 1] = str(global_idx)
                        new_key = '.'.join(parts)
                        # Convert to HF naming: blocks.X -> h.X
                        new_key = new_key.replace('blocks.', 'h.')
                        break
            
            # Handle embeddings (only in first stage)
            if pp_rank == 0:
                if 'token_embedding' in key:
                    new_key = 'wte.' + key.split('.')[-1]
                elif 'position_embedding' in key:
                    new_key = 'wpe.' + key.split('.')[-1]
            
            # Handle final layer norm and lm_head (only in last stage)
            if pp_rank == pp_size - 1:
                if 'final_ln' in key or 'ln_f' in key:
                    new_key = 'ln_f.' + key.split('.')[-1]
                elif 'lm_head' in key:
                    new_key = key
            
            merged[new_key] = value
    
    return merged


def convert_to_hf_format(merged_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert merged state dict to HuggingFace GPT2LMHeadModel format.
    
    Our format -> HF format mapping:
        blocks.X.ln1.* -> transformer.h.X.ln_1.*
        blocks.X.ln2.* -> transformer.h.X.ln_2.*
        blocks.X.attention.* -> transformer.h.X.attn.*
        blocks.X.mlp.* -> transformer.h.X.mlp.*
        etc.
    """
    hf_state = {}
    
    for key, value in merged_state.items():
        new_key = key
        
        # Add transformer. prefix for non-lm_head keys
        if not key.startswith('lm_head'):
            new_key = 'transformer.' + key
        
        # Convert layer norm naming
        new_key = new_key.replace('.ln1.', '.ln_1.')
        new_key = new_key.replace('.ln2.', '.ln_2.')
        
        # Handle Conv1D transpose (our Linear -> HF Conv1D)
        if 'c_attn.weight' in key or 'c_proj.weight' in key:
            value = value.t()
        elif 'c_fc.weight' in key or 'mlp.c_proj.weight' in key:
            value = value.t()
        
        hf_state[new_key] = value
    
    return hf_state


def merge_checkpoints(input_dir: str, output_path: str, prefix: str = "final_model"):
    """Main merge function."""
    print("=" * 70)
    print("🔄 MERGING 3D PARALLEL CHECKPOINTS")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output path: {output_path}")
    print()
    
    # Step 1: Load all shards
    print("📂 Loading shards...")
    shards = load_shards(input_dir, prefix)
    
    if not shards:
        print("❌ No checkpoint shards found!")
        return
    
    pp_size = len(shards)
    tp_size = len(shards[0]) if shards else 1
    print(f"\n  Found: PP={pp_size}, TP={tp_size}")
    
    # Get config from first shard
    config = shards[0][0].get('parallelism_info', {})
    model_config = shards[0][0].get('config', {}).get('model_config', {})
    
    # Step 2: Merge TP shards for each PP stage
    print("\n🔗 Merging TP shards...")
    pp_stages = {}
    for pp_rank in sorted(shards.keys()):
        tp_shards = shards[pp_rank]
        merged_stage = merge_tp_shards(tp_shards, pp_rank)
        pp_stages[pp_rank] = merged_stage
        print(f"  ✓ Merged PP stage {pp_rank}")
    
    # Step 3: Merge PP stages
    print("\n🔗 Merging PP stages...")
    full_state = merge_pp_stages(pp_stages, model_config)
    print(f"  ✓ Combined {pp_size} pipeline stages")
    
    # Step 4: Convert to HF format
    print("\n🔄 Converting to HuggingFace format...")
    hf_state = convert_to_hf_format(full_state)
    
    # Step 5: Save merged checkpoint
    print(f"\n💾 Saving merged checkpoint: {output_path}")
    torch.save({
        'model_state_dict': hf_state,
        'config': model_config,
    }, output_path)
    
    print("\n" + "=" * 70)
    print("✅ MERGE COMPLETE")
    print("=" * 70)
    print(f"Total parameters: {sum(p.numel() for p in hf_state.values()):,}")


def main():
    parser = argparse.ArgumentParser(description="Merge 3D Parallel Checkpoints")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing checkpoint shards')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for merged checkpoint')
    parser.add_argument('--prefix', type=str, default='final_model',
                        help='Checkpoint prefix (default: final_model)')
    
    args = parser.parse_args()
    
    merge_checkpoints(args.input_dir, args.output, args.prefix)


if __name__ == '__main__':
    main()
