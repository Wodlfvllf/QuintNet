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

# Persistent volumes for model and dataset
model_volume = modal.Volume.from_name("gpt2-model-volume", create_if_missing=True)
dataset_volume = modal.Volume.from_name("cnn-dailymail-volume", create_if_missing=True)

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
    image=image,
    gpu="A100:8",  # 8 GPUs for 3D parallelism (DP=2, TP=2, PP=2)
    volumes={
        "/mnt/model": model_volume,
        "/mnt/dataset": dataset_volume,
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


if __name__ == "__main__":
    with app.run():
        main()


