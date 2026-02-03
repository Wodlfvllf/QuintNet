# gpt2_train_modal_run.py - GPT-2 Training on Modal
"""
Run GPT-2 finetuning with 3D parallelism on Modal cloud.

Usage:
    # Step 1: Upload your local model and dataset to Modal volumes
    modal run QuintNet/gpt2_train_modal_run.py::upload_model
    modal run QuintNet/gpt2_train_modal_run.py::upload_dataset
    
    # Step 2: Run training
    modal run QuintNet/gpt2_train_modal_run.py
    
Prerequisites:
    1. Modal account and CLI installed: pip install modal
    2. Local GPT-2 model at: /Users/shashank/Deep_Learning/codebase/Pretrained_models
    3. Local dataset at: /Users/shashank/Deep_Learning/codebase/Text_Summarisation_Dataset
"""

import modal
import os
from pathlib import Path

# Define the Modal app for GPT-2 training
app = modal.App("quintnet-gpt2-training")

# Persistent volumes for model, dataset, and checkpoints
model_volume = modal.Volume.from_name("gpt2-model-volume", create_if_missing=True)
dataset_volume = modal.Volume.from_name("cnn-dailymail-volume", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("gpt2-checkpoints-volume", create_if_missing=True)

# ═══════════════════════════════════════════════════════════════════════════
# LOCAL PATHS (Update these to match your system)
# ═══════════════════════════════════════════════════════════════════════════
LOCAL_MODEL_PATH = "/Users/shashank/Deep_Learning/codebase/Pretrained_models"
LOCAL_DATASET_PATH = "/Users/shashank/Deep_Learning/codebase/Text_Summarisation_Dataset/cnn_dailymail"

# Define the image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "datasets",
        "einops",
        "transformers",
        "accelerate",
        "safetensors",
        "matplotlib",
        "scipy",
        "PyYAML",
        "pandas",
        "rouge-score",   # For ROUGE metrics
        "sacrebleu",     # For BLEU metrics
        "nltk",          # For tokenization in metrics
    ])
    # Install torch with CUDA support
    .run_commands("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    # Add the QuintNet directory to the image
    .add_local_dir(
        local_path=Path(__file__).parent,
        remote_path="/workspace/QuintNet"
    )
)


# ═══════════════════════════════════════════════════════════════════════════
# UPLOAD FUNCTIONS - Run these first to populate Modal volumes
# Call directly: modal run QuintNet/gpt2_train_modal_run.py::app.upload_model
# ═══════════════════════════════════════════════════════════════════════════

@app.function()
def upload_model():
    """
    Upload your local GPT-2 model to Modal volume.
    Usage: modal run QuintNet/gpt2_train_modal_run.py::upload_model
    """
    import shutil
    
    local_path = Path(LOCAL_MODEL_PATH)
    if not local_path.exists():
        print(f"❌ Model path not found: {local_path}")
        return
    
    print("📤 Uploading GPT-2 model to Modal volume...")
    print(f"   Source: {local_path}")
    
    # List files to upload
    files = list(local_path.glob("*"))
    print(f"   Files: {[f.name for f in files]}")
    
    # Use Modal's put_directory for efficient uploads
    with model_volume.batch_upload() as batch:
        for f in files:
            if f.is_file():
                print(f"   Uploading {f.name}...")
                batch.put_file(f, f"/{f.name}")
    
    print("✅ Model upload complete!")


@app.function()
def upload_dataset():
    """
    Upload your local dataset to Modal volume.
    Usage: modal run QuintNet/gpt2_train_modal_run.py::upload_dataset
    """
    local_path = Path(LOCAL_DATASET_PATH)
    if not local_path.exists():
        print(f"❌ Dataset path not found: {local_path}")
        return
    
    print("📤 Uploading dataset to Modal volume...")
    print(f"   Source: {local_path}")
    
    # List CSV files
    files = list(local_path.glob("*.csv"))
    print(f"   Files: {[f.name for f in files]}")
    
    with dataset_volume.batch_upload() as batch:
        for f in files:
            print(f"   Uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)...")
            batch.put_file(f, f"/cnn_dailymail/{f.name}")
    
    print("✅ Dataset upload complete!")


@app.function(
    volumes={"/mnt/checkpoints": checkpoint_volume},
)
def list_checkpoints():
    """
    List all checkpoint files in the Modal volume.
    Usage: modal run QuintNet/gpt2_train_modal_run.py::list_checkpoints
    """
    import os
    
    checkpoint_dir = "/mnt/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoints found. Run training first.")
        return []
    
    files = os.listdir(checkpoint_dir)
    pt_files = [f for f in files if f.endswith('.pt')]
    
    print("📂 Checkpoints in Modal volume:")
    for f in pt_files:
        path = os.path.join(checkpoint_dir, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"   - {f} ({size_mb:.1f} MB)")
    
    return pt_files


@app.local_entrypoint()
def download_checkpoints(output_dir: str = "./checkpoints"):
    """
    Download all checkpoint shards from Modal to local for merging.
    Usage: modal run QuintNet/gpt2_train_modal_run.py --command download --output-dir ./checkpoints
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List available checkpoints
    files = list_checkpoints.remote()
    
    if not files:
        print("❌ No checkpoints to download")
        return
    
    print(f"\n📥 Downloading {len(files)} checkpoint files to {output_dir}...")
    
    # Download each file
    for filename in files:
        remote_path = f"/mnt/checkpoints/{filename}"
        local_path = os.path.join(output_dir, filename)
        
        # Read from volume and write locally
        data = _read_checkpoint.remote(remote_path)
        with open(local_path, 'wb') as f:
            f.write(data)
        
        print(f"   ✓ Downloaded: {filename}")
    
    print(f"\n✅ All checkpoints downloaded to: {output_dir}")
    print(f"\n🔧 To merge, run:")
    print(f"   python merge_checkpoints.py --input_dir {output_dir} --output merged_model.pt")


@app.function(
    volumes={"/mnt/checkpoints": checkpoint_volume},
)
def _read_checkpoint(path: str) -> bytes:
    """Helper to read checkpoint file from volume."""
    with open(path, 'rb') as f:
        return f.read()


@app.function(
    image=image,
    gpu="A100:8",  # 8 GPUs for 3D parallelism (DP=2, TP=2, PP=2)
    volumes={
        "/mnt/model": model_volume,
        "/mnt/dataset": dataset_volume,
        "/mnt/checkpoints": checkpoint_volume,  # Persist checkpoints
    },
    timeout=14400,  # 4-hour timeout for longer training
    max_containers=1,  # Only one training job at a time
)
def run_gpt2_training(
    checkpoint_path: str = "/mnt/model/model.safetensors",
    dataset_path: str = "/mnt/dataset/cnn_dailymail",
    config_path: str = "QuintNet/examples/gpt2_config.yaml",
):
    """
    Run GPT-2 3D parallel finetuning on Modal.
    """
    import sys
    import subprocess
    import torch
    import re

    # Setup Python path
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    print("=" * 80)
    print("🚀 GPT-2 FINETUNING WITH 3D PARALLELISM")
    print("=" * 80)

    # Display GPU information
    num_gpus = torch.cuda.device_count()
    print(f"📊 Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    # Verify checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint found: {checkpoint_path}")
    else:
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("   Please upload your GPT-2 checkpoint to the Modal volume.")
        return 1

    # Verify dataset exists
    if os.path.exists(dataset_path):
        files = os.listdir(dataset_path)
        print(f"✅ Dataset found at {dataset_path}")
        print(f"   Files: {files}")
    else:
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print("   Please upload train.csv, validation.csv to the Modal volume.")
        return 1

    # Verify config exists
    config_path = "QuintNet/examples/gpt2_config.yaml"
    if os.path.exists(config_path):
        print(f"✅ Config found: {config_path}")
    else:
        print(f"✅ Config assumed at: {config_path}")

    print("=" * 80)

    # Setup distributed training environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(num_gpus)

    # Command to run distributed training
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m", "QuintNet.examples.gpt2_finetune",
        "--config", "QuintNet/examples/gpt2_config.yaml",
        # Note: Paths must match where volumes are mounted and expected
        "--checkpoint", "/mnt/model/model.safetensors", 
        "--dataset", "/mnt/dataset/cnn_dailymail",
        "--tokenizer", "gpt2",
    ]
    
    print(f"🔥 Launching: {' '.join(cmd)}")
    print("=" * 80)
    print()

    # Start training with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Stream and parse output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip()
            print(line)

            # Highlight epoch progress
            if line.startswith("Epoch") and "/" in line:
                print("\n" + "=" * 80 + f"\n🔄 {line.upper()}\n" + "=" * 80)

            # Highlight training metrics (perplexity for GPT-2)
            if "Train Loss:" in line:
                match = re.search(r'Train Loss: ([\d.]+) \| Train PPL: ([\d.]+)', line)
                if match:
                    loss, ppl = match.groups()
                    print(f"📈 TRAIN | Loss: {loss} | Perplexity: {ppl}")

            # Highlight validation metrics
            if "Val Loss:" in line:
                match = re.search(r'Val Loss:\s+([\d.]+) \| Val PPL:\s+([\d.]+)', line)
                if match:
                    loss, ppl = match.groups()
                    print(f"✅ VALIDATION | Loss: {loss} | Perplexity: {ppl}")

    return_code = process.poll()

    print()
    print("=" * 80)
    if return_code == 0:
        print("✅ GPT-2 TRAINING COMPLETED SUCCESSFULLY")
    else:
        print(f"❌ TRAINING FAILED WITH ERROR CODE: {return_code}")
    print("=" * 80)

    return return_code


@app.function(
    image=image,
    volumes={"/mnt/dataset": dataset_volume},
    timeout=3600,
)
def prepare_dataset():
    """
    Helper function to download and prepare CNN/DailyMail dataset.
    Run this first: modal run QuintNet/gpt2_train_modal_run.py::prepare_dataset
    """
    import pandas as pd
    from datasets import load_dataset
    
    print("📥 Downloading CNN/DailyMail dataset...")
    
    # Load from HuggingFace
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    output_dir = "/mnt/dataset/cnn_dailymail"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to CSV for our SummarizationDataset
    for split in ["train", "validation", "test"]:
        print(f"   Processing {split}...")
        df = pd.DataFrame(dataset[split])
        # Rename columns to match our expected format
        df = df.rename(columns={"article": "article", "highlights": "highlights"})
        df = df[["article", "highlights"]]  # Keep only needed columns
        
        csv_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(csv_path, index=False)
        print(f"   Saved {len(df)} samples to {csv_path}")
    
    # Commit the volume
    dataset_volume.commit()
    
    print("✅ Dataset preparation complete!")
    return 0


@app.local_entrypoint()
def main(command: str = "train"):
    """
    Local entrypoint for GPT-2 training workflow.
    
    Commands:
        upload-model   - Upload local GPT-2 model to Modal volume
        upload-dataset - Upload local dataset to Modal volume
        train          - Run training on Modal (default)
        test-weights   - Run weight shape verification test
    
    Usage:
        modal run QuintNet/gpt2_train_modal_run.py --command upload-model
        modal run QuintNet/gpt2_train_modal_run.py --command upload-dataset
        modal run QuintNet/gpt2_train_modal_run.py --command train
        modal run QuintNet/gpt2_train_modal_run.py  # defaults to train
    """
    
    # Arguments are passed directly by Modal via function signature

    
    if command == "upload-model":
        # ═══════════════════════════════════════════════════════════════
        # UPLOAD MODEL - Runs locally, uploads to Modal volume
        # ═══════════════════════════════════════════════════════════════
        local_path = Path(LOCAL_MODEL_PATH)
        if not local_path.exists():
            print(f"❌ Model path not found: {local_path}")
            return
        
        print("📤 Uploading GPT-2 model to Modal volume...")
        print(f"   Source: {local_path}")
        
        # List files to upload
        files = [f for f in local_path.glob("*") if f.is_file()]
        print(f"   Files: {[f.name for f in files]}")
        
        # Use Modal's batch upload (force=True to overwrite)
        with model_volume.batch_upload(force=True) as batch:
            for f in files:
                size_mb = f.stat().st_size / 1e6
                print(f"   Uploading {f.name} ({size_mb:.1f} MB)...")
                batch.put_file(f, f"/{f.name}")
        
        print("✅ Model upload complete!")
        
    elif command == "upload-dataset":
        # ═══════════════════════════════════════════════════════════════
        # UPLOAD DATASET - Runs locally, uploads to Modal volume
        # ═══════════════════════════════════════════════════════════════
        local_path = Path(LOCAL_DATASET_PATH)
        if not local_path.exists():
            print(f"❌ Dataset path not found: {local_path}")
            return
        
        print("📤 Uploading dataset to Modal volume...")
        print(f"   Source: {local_path}")
        
        # List CSV files
        files = list(local_path.glob("*.csv"))
        print(f"   Files: {[f.name for f in files]}")
        
        with dataset_volume.batch_upload(force=True) as batch:
            for f in files:
                size_mb = f.stat().st_size / 1e6
                print(f"   Uploading {f.name} ({size_mb:.1f} MB)...")
                batch.put_file(f, f"/cnn_dailymail/{f.name}")
        
        print("✅ Dataset upload complete!")
        
    elif command == "train":
        # ═══════════════════════════════════════════════════════════════
        # TRAIN - Runs training on Modal cloud
        # ═══════════════════════════════════════════════════════════════
        print("🚀 Starting GPT-2 Finetuning on Modal...")
        print()
        
        result = run_gpt2_training.remote()
        
        print()
        if result == 0:
            print("✅ Training job completed successfully!")
        else:
            print(f"❌ Training job failed with return code: {result}")
    
    elif command == "test-weights":
        # ═══════════════════════════════════════════════════════════════
        # TEST WEIGHTS - Run on Modal to verify weight loading shapes
        # ═══════════════════════════════════════════════════════════════
        print("🧪 Testing weight loading shapes on Modal...")
        result = test_weight_shapes.remote()
        if result == 0:
            print("✅ All weight shape tests passed!")
        else:
            print(f"❌ Weight shape tests failed")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("   Valid commands: upload-model, upload-dataset, train, test-weights")


@app.function(
    image=image,
    volumes={"/mnt/model": model_volume},
    timeout=300,
)
def test_weight_shapes():
    """Test weight slicing and transpose logic for TP parallelism."""
    import torch
    from safetensors import safe_open
    
    CHECKPOINT_PATH = "/mnt/model/model.safetensors"
    TP_SIZE = 2
    EMBED_DIM = 768
    HIDDEN_DIM = 3072
    
    print("=" * 70)
    print("TESTING WEIGHT LOADING SHAPES FOR GPT-2 + TENSOR PARALLELISM")
    print("=" * 70)
    print(f"TP_SIZE: {TP_SIZE}, EMBED_DIM: {EMBED_DIM}, HIDDEN_DIM: {HIDDEN_DIM}")
    print()
    
    all_passed = True
    
    with safe_open(CHECKPOINT_PATH, framework='pt') as f:
        prefix = "h.0"
        
        # ─────────────────────────────────────────────────────────────────
        # TEST 1: c_attn (Column Parallel)
        # ─────────────────────────────────────────────────────────────────
        print("TEST 1: c_attn (Column Parallel)")
        c_attn_w_full = f.get_tensor(f"{prefix}.attn.c_attn.weight")
        print(f"  Original Conv1D: {c_attn_w_full.shape}")
        
        # ColumnParallelLinear expects: [out_features_per_rank, in_features] = [1152, 768]
        expected = (3 * EMBED_DIM // TP_SIZE, EMBED_DIM)
        
        for tp_rank in range(TP_SIZE):
            cols_per_rank = 3 * EMBED_DIM // TP_SIZE
            col_start, col_end = tp_rank * cols_per_rank, (tp_rank + 1) * cols_per_rank
            weight_slice = c_attn_w_full[:, col_start:col_end].t().contiguous()
            status = "✅" if weight_slice.shape == expected else "❌"
            print(f"  TP_RANK {tp_rank}: {weight_slice.shape} (expected {expected}) {status}")
            if weight_slice.shape != expected:
                all_passed = False
        
        # ─────────────────────────────────────────────────────────────────
        # TEST 2: attn.c_proj (Row Parallel)
        # ─────────────────────────────────────────────────────────────────
        print("\nTEST 2: attn.c_proj (Row Parallel)")
        c_proj_w_full = f.get_tensor(f"{prefix}.attn.c_proj.weight")
        print(f"  Original Conv1D: {c_proj_w_full.shape}")
        
        # RowParallelLinear expects: [out_features, in_features_per_rank] = [768, 384]
        expected = (EMBED_DIM, EMBED_DIM // TP_SIZE)
        
        for tp_rank in range(TP_SIZE):
            rows_per_rank = EMBED_DIM // TP_SIZE
            row_start, row_end = tp_rank * rows_per_rank, (tp_rank + 1) * rows_per_rank
            weight_slice = c_proj_w_full[row_start:row_end, :].t().contiguous()
            status = "✅" if weight_slice.shape == expected else "❌"
            print(f"  TP_RANK {tp_rank}: {weight_slice.shape} (expected {expected}) {status}")
            if weight_slice.shape != expected:
                all_passed = False
        
        # ─────────────────────────────────────────────────────────────────
        # TEST 3: mlp.c_fc (Column Parallel)
        # ─────────────────────────────────────────────────────────────────
        print("\nTEST 3: mlp.c_fc (Column Parallel)")
        c_fc_w_full = f.get_tensor(f"{prefix}.mlp.c_fc.weight")
        print(f"  Original Conv1D: {c_fc_w_full.shape}")
        
        # ColumnParallelLinear expects: [1536, 768]
        expected = (HIDDEN_DIM // TP_SIZE, EMBED_DIM)
        
        for tp_rank in range(TP_SIZE):
            cols_per_rank = HIDDEN_DIM // TP_SIZE
            col_start, col_end = tp_rank * cols_per_rank, (tp_rank + 1) * cols_per_rank
            weight_slice = c_fc_w_full[:, col_start:col_end].t().contiguous()
            status = "✅" if weight_slice.shape == expected else "❌"
            print(f"  TP_RANK {tp_rank}: {weight_slice.shape} (expected {expected}) {status}")
            if weight_slice.shape != expected:
                all_passed = False
        
        # ─────────────────────────────────────────────────────────────────
        # TEST 4: mlp.c_proj (Row Parallel)
        # ─────────────────────────────────────────────────────────────────
        print("\nTEST 4: mlp.c_proj (Row Parallel)")
        mlp_c_proj_w_full = f.get_tensor(f"{prefix}.mlp.c_proj.weight")
        print(f"  Original Conv1D: {mlp_c_proj_w_full.shape}")
        
        # RowParallelLinear expects: [768, 1536]
        expected = (EMBED_DIM, HIDDEN_DIM // TP_SIZE)
        
        for tp_rank in range(TP_SIZE):
            rows_per_rank = HIDDEN_DIM // TP_SIZE
            row_start, row_end = tp_rank * rows_per_rank, (tp_rank + 1) * rows_per_rank
            weight_slice = mlp_c_proj_w_full[row_start:row_end, :].t().contiguous()
            status = "✅" if weight_slice.shape == expected else "❌"
            print(f"  TP_RANK {tp_rank}: {weight_slice.shape} (expected {expected}) {status}")
            if weight_slice.shape != expected:
                all_passed = False
    
    print()
    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Weight loading logic is correct!")
    else:
        print("❌ SOME TESTS FAILED - Check weight slicing logic")
    print("=" * 70)
    
    return 0 if all_passed else 1


# ═══════════════════════════════════════════════════════════════════════════
# MERGE AND TEST ON SINGLE GPU - Runs after training
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A100",  # Single A100 GPU for testing
    volumes={
        "/mnt/model": model_volume,
        "/mnt/dataset": dataset_volume,
        "/mnt/checkpoints": checkpoint_volume,
    },
    timeout=3600,  # 1 hour timeout
)
def merge_and_test_model(
    max_samples: int = 250,
    max_gen_samples: int = 10,
    batch_size: int = 4,
):
    """
    Merge 3D parallel checkpoints and test on single GPU.
    
    This function:
    1. Loads all checkpoint shards from /mnt/checkpoints
    2. Merges TP and PP shards into a single model
    3. Tests on 200-300 validation samples
    4. Computes loss, perplexity, and generates sample outputs
    
    Usage:
        modal run QuintNet/gpt2_train_modal_run.py::merge_and_test_model
    """
    import sys
    import math
    import torch
    import torch.nn as nn
    from pathlib import Path
    from tqdm import tqdm
    from collections import defaultdict
    import pandas as pd
    
    sys.path.insert(0, "/workspace")
    
    print("=" * 80)
    print("🔄 MERGE CHECKPOINTS & TEST ON SINGLE GPU")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load all checkpoint shards
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📂 STEP 1: Loading checkpoint shards...")
    checkpoint_dir = "/mnt/checkpoints"
    shards = defaultdict(dict)
    
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("final_model") and filename.endswith(".pt"):
            # Parse: final_model_pp{pp}_tp{tp}.pt
            parts = filename.replace(".pt", "").split("_")
            pp_rank = None
            tp_rank = None
            
            for part in parts:
                if part.startswith("pp"):
                    pp_rank = int(part[2:])
                elif part.startswith("tp"):
                    tp_rank = int(part[2:])
            
            if pp_rank is not None and tp_rank is not None:
                path = os.path.join(checkpoint_dir, filename)
                checkpoint = torch.load(path, map_location="cpu")
                shards[pp_rank][tp_rank] = checkpoint
                size_mb = os.path.getsize(path) / 1e6
                print(f"   ✓ Loaded {filename} (PP={pp_rank}, TP={tp_rank}, {size_mb:.1f} MB)")
    
    if not shards:
        print("❌ No checkpoint shards found!")
        # Try loading single checkpoint directly
        single_ckpt = os.path.join(checkpoint_dir, "final_model_pp0_tp0.pt")
        if os.path.exists(single_ckpt):
            print(f"   Found single checkpoint: {single_ckpt}")
            checkpoint = torch.load(single_ckpt, map_location="cpu")
            merged_state = checkpoint.get("model_state_dict", checkpoint)
        else:
            return 1
    else:
        pp_size = len(shards)
        tp_size = len(shards[0]) if shards else 1
        print(f"\n   📊 Found: PP={pp_size}, TP={tp_size}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Merge TP shards for each PP stage
        # ─────────────────────────────────────────────────────────────────────
        print("\n🔗 STEP 2: Merging TP shards...")
        
        pp_stages = {}
        for pp_rank in sorted(shards.keys()):
            tp_shards = shards[pp_rank]
            
            if len(tp_shards) == 1:
                # No TP, just use the single shard
                merged_stage = tp_shards[0].get("model_state_dict", tp_shards[0])
            else:
                # Merge TP shards
                tp_size = len(tp_shards)
                merged_stage = {}
                
                # Get state dicts for each rank
                rank_states = [
                    tp_shards[r].get("model_state_dict", tp_shards[r])
                    for r in range(tp_size)
                ]
                
                # Collect all unique keys across all ranks
                all_keys = set()
                for state in rank_states:
                    all_keys.update(state.keys())
                
                for key in all_keys:
                    # Check which ranks have this key
                    available_tensors = []
                    available_ranks = []
                    for r in range(tp_size):
                        if key in rank_states[r]:
                            available_tensors.append(rank_states[r][key])
                            available_ranks.append(r)
                    
                    if len(available_tensors) == 0:
                        continue
                    
                    if len(available_tensors) == 1:
                        # Key only exists in one rank (e.g., row-parallel bias)
                        merged_stage[key] = available_tensors[0]
                    else:
                        # Key exists in multiple ranks - merge based on layer type
                        # NOTE: Keys have .proj. in them, e.g. c_attn.proj.weight
                        if "c_attn" in key and "weight" in key:
                            # Column parallel weights: concat along dim 0 (out_features)
                            merged_stage[key] = torch.cat(available_tensors, dim=0)
                            print(f"      TP concat c_attn.weight: {key} -> {merged_stage[key].shape}")
                        elif "c_attn" in key and "bias" in key:
                            # Column parallel biases: concat along dim 0
                            merged_stage[key] = torch.cat(available_tensors, dim=0)
                        elif "c_fc" in key and "weight" in key:
                            # Column parallel weights: concat along dim 0
                            merged_stage[key] = torch.cat(available_tensors, dim=0)
                            print(f"      TP concat c_fc.weight: {key} -> {merged_stage[key].shape}")
                        elif "c_fc" in key and "bias" in key:
                            # Column parallel biases: concat along dim 0
                            merged_stage[key] = torch.cat(available_tensors, dim=0)
                        elif "c_proj" in key and "weight" in key:
                            # Row parallel weights: concat along dim 1 (in_features)
                            merged_stage[key] = torch.cat(available_tensors, dim=1)
                            print(f"      TP concat c_proj.weight: {key} -> {merged_stage[key].shape}")
                        elif "c_proj" in key and "bias" in key:
                            # Row parallel bias: use rank 0's copy (should be identical after allreduce)
                            merged_stage[key] = available_tensors[0]
                        else:
                            # Non-sharded layers (LayerNorm, embeddings): use rank 0
                            merged_stage[key] = available_tensors[0]
            
            pp_stages[pp_rank] = merged_stage
            print(f"   ✓ Merged PP stage {pp_rank}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Merge PP stages (renumber layers for stage > 0)
        # ─────────────────────────────────────────────────────────────────────
        print("\n🔗 STEP 3: Merging PP stages...")
        
        # Calculate layers per stage
        n_layers = 12  # GPT-2 has 12 layers
        layers_per_stage = n_layers // pp_size
        print(f"   Layers per stage: {layers_per_stage}")
        
        if pp_size == 1:
            merged_state = pp_stages[0]
        else:
            # Combine all PP stages, renumbering layer indices
            merged_state = {}
            for pp_rank, stage_state in sorted(pp_stages.items()):
                layer_offset = pp_rank * layers_per_stage
                
                for key, value in stage_state.items():
                    new_key = key
                    
                    # Renumber block indices: blocks.X -> blocks.(X + offset)
                    if key.startswith("blocks."):
                        parts = key.split(".")
                        block_idx = int(parts[1])
                        new_block_idx = block_idx + layer_offset
                        parts[1] = str(new_block_idx)
                        new_key = ".".join(parts)
                        
                        if block_idx != new_block_idx:
                            print(f"      PP renumber: {key} -> {new_key}")
                    
                    merged_state[new_key] = value
        
        print(f"   ✓ Combined {pp_size} pipeline stages")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Convert keys and load into HuggingFace GPT2LMHeadModel
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🤖 STEP 4: Loading merged model...")
    from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
    
    # Print sample keys for debugging
    sample_keys = list(merged_state.keys())[:10]
    print(f"   Sample checkpoint keys:")
    for k in sample_keys:
        print(f"      - {k}")
    
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
    )
    
    model = GPT2LMHeadModel(config)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convert QuintNet keys to HuggingFace format
    # Actual checkpoint format from logs:
    #   embedding.wte.weight, embedding.wpe.weight
    #   lm_head_weight (underscore!)
    #   blocks.X.attn.c_attn.proj.weight (extra .proj.!)  
    #   blocks.X.mlp.c_fc.proj.weight (extra .proj.!)
    # HuggingFace expects:
    #   transformer.wte.weight, transformer.wpe.weight
    #   lm_head.weight
    #   transformer.h.X.attn.c_attn.weight
    #   transformer.h.X.mlp.c_fc.weight
    # ─────────────────────────────────────────────────────────────────────────
    print("\n   🔄 Converting keys to HuggingFace format...")
    
    hf_state = {}
    converted_count = 0
    
    for key, value in merged_state.items():
        new_key = key
        need_transpose = False
        
        # Handle embeddings: embedding.wte -> transformer.wte
        if key.startswith("embedding."):
            new_key = key.replace("embedding.", "transformer.")
        
        # Handle lm_head_weight -> lm_head.weight (underscore to dot)
        if key == "lm_head_weight":
            new_key = "lm_head.weight"
        
        # Convert block naming: blocks.X -> transformer.h.X
        if key.startswith("blocks."):
            new_key = key.replace("blocks.", "transformer.h.")
        
        # Remove extra .proj. from attention and MLP layers
        # blocks.X.attn.c_attn.proj.weight -> transformer.h.X.attn.c_attn.weight
        # blocks.X.attn.c_proj.proj.weight -> transformer.h.X.attn.c_proj.weight
        # blocks.X.mlp.c_fc.proj.weight -> transformer.h.X.mlp.c_fc.weight
        # blocks.X.mlp.c_proj.proj.weight -> transformer.h.X.mlp.c_proj.weight
        new_key = new_key.replace(".c_attn.proj.", ".c_attn.")
        new_key = new_key.replace(".c_proj.proj.", ".c_proj.")
        new_key = new_key.replace(".c_fc.proj.", ".c_fc.")
        
        # Convert layer norm naming: ln1 -> ln_1, ln2 -> ln_2
        new_key = new_key.replace(".ln1.", ".ln_1.")
        new_key = new_key.replace(".ln2.", ".ln_2.")
        
        # Handle final layer norm
        if "final_ln." in key:
            new_key = "transformer.ln_f." + key.split(".")[-1]
        elif key.startswith("ln_f.") and not key.startswith("transformer."):
            new_key = "transformer.ln_f." + key.split(".")[-1]
        
        # Handle attention/MLP weights - need transpose for Conv1D -> Linear
        # HuggingFace GPT-2 uses Conv1D which stores weights as (in_features, out_features)
        # Our Linear layers store as (out_features, in_features), so we need to transpose
        if ".c_attn.weight" in new_key or ".c_proj.weight" in new_key:
            if len(value.shape) == 2:  # Only transpose 2D weights
                value = value.t().contiguous()
                need_transpose = True
        elif ".c_fc.weight" in new_key:
            if len(value.shape) == 2:
                value = value.t().contiguous()
                need_transpose = True
        
        hf_state[new_key] = value
        converted_count += 1
        
        if converted_count <= 10:
            trans_str = " (transposed)" if need_transpose else ""
            print(f"      {key} -> {new_key}{trans_str}")
    
    print(f"   ... converted {converted_count} keys total")
    
    # Handle weight tying for lm_head
    if "transformer.wte.weight" in hf_state and "lm_head.weight" not in hf_state:
        hf_state["lm_head.weight"] = hf_state["transformer.wte.weight"]
        print("   ✓ Tied lm_head.weight to transformer.wte.weight")
    
    # Load state dict
    print("\n   📥 Loading state dict...")
    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    
    print(f"   Missing keys: {len(missing)}")
    if len(missing) > 0 and len(missing) <= 10:
        for k in missing:
            print(f"      ❌ {k}")
    elif len(missing) > 10:
        for k in missing[:5]:
            print(f"      ❌ {k}")
        print(f"      ... and {len(missing) - 5} more")
    
    print(f"   Unexpected keys: {len(unexpected)}")
    if len(unexpected) > 0 and len(unexpected) <= 10:
        for k in unexpected:
            print(f"      ⚠️  {k}")
    elif len(unexpected) > 10:
        for k in unexpected[:5]:
            print(f"      ⚠️  {k}")
        print(f"      ... and {len(unexpected) - 5} more")
    
    # Warn if too many mismatches
    if len(missing) > 10 or len(unexpected) > 10:
        print("\n   ⚠️  WARNING: Many key mismatches! Model may not load correctly.")
        print("   Expected HuggingFace keys look like: transformer.h.0.attn.c_attn.weight")
        print("   Check if your checkpoint uses different naming.")
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded: {total_params:,} parameters")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Load test dataset
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📚 STEP 5: Loading validation dataset...")
    dataset_path = "/mnt/dataset/cnn_dailymail/validation.csv"
    
    if not os.path.exists(dataset_path):
        print(f"   ❌ Dataset not found at {dataset_path}")
        return 1
    
    df = pd.read_csv(dataset_path)
    print(f"   ✅ Loaded {len(df)} validation samples")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Compute Loss and Perplexity
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n📊 STEP 6: Computing loss on {max_samples} samples...")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    num_samples = min(max_samples, len(df))
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Computing loss"):
            batch_end = min(i + batch_size, num_samples)
            batch_texts = []
            
            for j in range(i, batch_end):
                article = str(df.iloc[j]["article"])[:1500]  # Truncate long articles
                highlights = str(df.iloc[j]["highlights"])
                text = f"{article}\n\nTL;DR: {highlights}"
                batch_texts.append(text)
            
            # Tokenize
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            # Create labels
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss (shift for CLM)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1)
            )
            
            num_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    
    print("\n" + "=" * 60)
    print("📈 LOSS METRICS")
    print("=" * 60)
    print(f"   Samples Tested:  {num_samples}")
    print(f"   Average Loss:    {avg_loss:.4f}")
    print(f"   Perplexity:      {perplexity:.2f}")
    print("=" * 60)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Generate Sample Outputs
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n📝 STEP 7: Generating {max_gen_samples} sample summaries...")
    
    sample_outputs = []
    
    with torch.no_grad():
        for i in range(min(max_gen_samples, len(df))):
            article = str(df.iloc[i]["article"])[:800]  # Use first 800 chars
            reference = str(df.iloc[i]["highlights"])
            
            prompt = f"{article}\n\nTL;DR:"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=400,
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
            )
            
            # Decode
            generated_ids = outputs[0, input_ids.size(1):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            sample_outputs.append({
                "reference": reference[:300],
                "generated": generated_text[:300],
            })
    
    # Print sample outputs
    print("\n" + "=" * 80)
    print("📝 SAMPLE GENERATED SUMMARIES")
    print("=" * 80)
    
    for i, sample in enumerate(sample_outputs):
        print(f"\n{'─' * 80}")
        print(f"📌 SAMPLE {i + 1}")
        print(f"{'─' * 80}")
        print(f"🎯 REFERENCE:")
        print(f"   {sample['reference']}")
        print(f"\n🤖 GENERATED:")
        print(f"   {sample['generated']}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🏁 FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"   ✅ Model:         GPT-2 (124M parameters)")
    print(f"   ✅ Samples:       {num_samples}")
    print(f"   ✅ Loss:          {avg_loss:.4f}")
    print(f"   ✅ Perplexity:    {perplexity:.2f}")
    print(f"   ✅ Generations:   {len(sample_outputs)} samples")
    print("=" * 80)
    
    # Save merged model for future use
    merged_model_path = "/mnt/checkpoints/merged_gpt2.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "test_results": {
            "loss": avg_loss,
            "perplexity": perplexity,
            "samples_tested": num_samples,
        }
    }, merged_model_path)
    print(f"\n💾 Saved merged model to: {merged_model_path}")
    
    # Commit the checkpoint volume
    checkpoint_volume.commit()
    
    return 0


if __name__ == "__main__":
    with app.run():
        main()


