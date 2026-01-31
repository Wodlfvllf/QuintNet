"""
Test script to verify weight loading shapes for GPT-2 with Tensor Parallelism.

This tests the slicing and transpose logic in distributed_loading.py
to ensure weights match what ColumnParallelLinear and RowParallelLinear expect.

Usage: python QuintNet/test_weight_shapes.py
"""

import torch
from safetensors import safe_open

# Configuration
CHECKPOINT_PATH = "/Users/shashank/Deep_Learning/codebase/Pretrained_models/model.safetensors"
TP_SIZE = 2
EMBED_DIM = 768
HIDDEN_DIM = 3072

def test_weight_shapes():
    print("=" * 70)
    print("TESTING WEIGHT LOADING SHAPES FOR GPT-2 + TENSOR PARALLELISM")
    print("=" * 70)
    print(f"TP_SIZE: {TP_SIZE}")
    print(f"EMBED_DIM: {EMBED_DIM}")
    print(f"HIDDEN_DIM: {HIDDEN_DIM}")
    print()

    with safe_open(CHECKPOINT_PATH, framework='pt') as f:
        # Test layer 0
        prefix = "h.0"
        
        # ═══════════════════════════════════════════════════════════════════
        # TEST 1: c_attn (Column Parallel)
        # ═══════════════════════════════════════════════════════════════════
        print("─" * 70)
        print("TEST 1: c_attn (Column Parallel)")
        print("─" * 70)
        
        c_attn_w_full = f.get_tensor(f"{prefix}.attn.c_attn.weight")
        print(f"Original Conv1D shape: {c_attn_w_full.shape}")
        # Expected: [768, 2304]
        
        # What ColumnParallelLinear expects:
        # nn.Linear(in_features=768, out_features_per_rank=1152)
        # creates weight shape: [1152, 768]
        expected_shape = (3 * EMBED_DIM // TP_SIZE, EMBED_DIM)
        print(f"ColumnParallelLinear expects: {expected_shape}")
        
        for tp_rank in range(TP_SIZE):
            c_attn_out_dim = 3 * EMBED_DIM  # 2304
            cols_per_rank = c_attn_out_dim // TP_SIZE  # 1152
            col_start = tp_rank * cols_per_rank
            col_end = col_start + cols_per_rank
            
            # Slice columns in Conv1D format
            c_attn_weight_slice = c_attn_w_full[:, col_start:col_end]
            print(f"  TP_RANK {tp_rank}: After slice [:, {col_start}:{col_end}] = {c_attn_weight_slice.shape}")
            
            # Transpose to Linear format
            c_attn_weight_final = c_attn_weight_slice.t().contiguous()
            print(f"  TP_RANK {tp_rank}: After transpose = {c_attn_weight_final.shape}")
            
            if c_attn_weight_final.shape == expected_shape:
                print(f"  ✅ PASS: Shape matches expected {expected_shape}")
            else:
                print(f"  ❌ FAIL: Expected {expected_shape}, got {c_attn_weight_final.shape}")
        
        # ═══════════════════════════════════════════════════════════════════
        # TEST 2: attn.c_proj (Row Parallel)
        # ═══════════════════════════════════════════════════════════════════
        print()
        print("─" * 70)
        print("TEST 2: attn.c_proj (Row Parallel)")
        print("─" * 70)
        
        c_proj_w_full = f.get_tensor(f"{prefix}.attn.c_proj.weight")
        print(f"Original Conv1D shape: {c_proj_w_full.shape}")
        # Expected: [768, 768]
        
        # What RowParallelLinear expects:
        # nn.Linear(in_features_per_rank=384, out_features=768)
        # creates weight shape: [768, 384]
        expected_shape = (EMBED_DIM, EMBED_DIM // TP_SIZE)
        print(f"RowParallelLinear expects: {expected_shape}")
        
        for tp_rank in range(TP_SIZE):
            rows_per_rank = EMBED_DIM // TP_SIZE  # 384
            row_start = tp_rank * rows_per_rank
            row_end = row_start + rows_per_rank
            
            # Slice ROWS in Conv1D format (input dimension)
            c_proj_weight_slice = c_proj_w_full[row_start:row_end, :]
            print(f"  TP_RANK {tp_rank}: After slice [{row_start}:{row_end}, :] = {c_proj_weight_slice.shape}")
            
            # Transpose to Linear format
            c_proj_weight_final = c_proj_weight_slice.t().contiguous()
            print(f"  TP_RANK {tp_rank}: After transpose = {c_proj_weight_final.shape}")
            
            if c_proj_weight_final.shape == expected_shape:
                print(f"  ✅ PASS: Shape matches expected {expected_shape}")
            else:
                print(f"  ❌ FAIL: Expected {expected_shape}, got {c_proj_weight_final.shape}")
        
        # ═══════════════════════════════════════════════════════════════════
        # TEST 3: mlp.c_fc (Column Parallel)
        # ═══════════════════════════════════════════════════════════════════
        print()
        print("─" * 70)
        print("TEST 3: mlp.c_fc (Column Parallel)")
        print("─" * 70)
        
        c_fc_w_full = f.get_tensor(f"{prefix}.mlp.c_fc.weight")
        print(f"Original Conv1D shape: {c_fc_w_full.shape}")
        # Expected: [768, 3072]
        
        # What ColumnParallelLinear expects:
        # nn.Linear(in_features=768, out_features_per_rank=1536)
        # creates weight shape: [1536, 768]
        expected_shape = (HIDDEN_DIM // TP_SIZE, EMBED_DIM)
        print(f"ColumnParallelLinear expects: {expected_shape}")
        
        for tp_rank in range(TP_SIZE):
            cols_per_rank = HIDDEN_DIM // TP_SIZE  # 1536
            col_start = tp_rank * cols_per_rank
            col_end = col_start + cols_per_rank
            
            c_fc_weight_slice = c_fc_w_full[:, col_start:col_end]
            print(f"  TP_RANK {tp_rank}: After slice [:, {col_start}:{col_end}] = {c_fc_weight_slice.shape}")
            
            c_fc_weight_final = c_fc_weight_slice.t().contiguous()
            print(f"  TP_RANK {tp_rank}: After transpose = {c_fc_weight_final.shape}")
            
            if c_fc_weight_final.shape == expected_shape:
                print(f"  ✅ PASS: Shape matches expected {expected_shape}")
            else:
                print(f"  ❌ FAIL: Expected {expected_shape}, got {c_fc_weight_final.shape}")
        
        # ═══════════════════════════════════════════════════════════════════
        # TEST 4: mlp.c_proj (Row Parallel)
        # ═══════════════════════════════════════════════════════════════════
        print()
        print("─" * 70)
        print("TEST 4: mlp.c_proj (Row Parallel)")
        print("─" * 70)
        
        mlp_c_proj_w_full = f.get_tensor(f"{prefix}.mlp.c_proj.weight")
        print(f"Original Conv1D shape: {mlp_c_proj_w_full.shape}")
        # Expected: [3072, 768]
        
        # What RowParallelLinear expects:
        # nn.Linear(in_features_per_rank=1536, out_features=768)
        # creates weight shape: [768, 1536]
        expected_shape = (EMBED_DIM, HIDDEN_DIM // TP_SIZE)
        print(f"RowParallelLinear expects: {expected_shape}")
        
        for tp_rank in range(TP_SIZE):
            rows_per_rank = HIDDEN_DIM // TP_SIZE  # 1536
            row_start = tp_rank * rows_per_rank
            row_end = row_start + rows_per_rank
            
            # Slice ROWS in Conv1D format (input dimension)
            mlp_c_proj_weight_slice = mlp_c_proj_w_full[row_start:row_end, :]
            print(f"  TP_RANK {tp_rank}: After slice [{row_start}:{row_end}, :] = {mlp_c_proj_weight_slice.shape}")
            
            mlp_c_proj_weight_final = mlp_c_proj_weight_slice.t().contiguous()
            print(f"  TP_RANK {tp_rank}: After transpose = {mlp_c_proj_weight_final.shape}")
            
            if mlp_c_proj_weight_final.shape == expected_shape:
                print(f"  ✅ PASS: Shape matches expected {expected_shape}")
            else:
                print(f"  ❌ FAIL: Expected {expected_shape}, got {mlp_c_proj_weight_final.shape}")
    
    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_weight_shapes()
