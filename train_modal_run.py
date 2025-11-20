# run_training.py (at root level)
import modal
import os
from pathlib import Path

# Define the Modal app for QuintNet 3D training
app = modal.App("quintnet-3d-training")

# Use a persistent volume for the dataset to avoid re-downloading
volume = modal.Volume.from_name("mnist-volume", create_if_missing=True)

# Define the image with dependencies for distributed training
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
    ])
    # Install torch with the correct CUDA version
    .run_commands("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    # Add the QuintNet directory to the image
    .add_local_dir(
        str(Path(__file__).parent.parent), # Get the root QuintNet dir
        remote_path="/workspace/QuintNet"
    )
)

@app.function(
    image=image,
    gpu="T4:8",  # 8 GPUs for 3D parallelism (DP=2, TP=2, PP=2)
    volumes={"/mnt/dataset": volume},  # Mount the persistent dataset volume
    timeout=7200,  # 2-hour timeout
    concurrency_limit=1,  # Only one training job at a time
)
def run_3d_parallel_training():
    """
    Run QuintNet 3D parallel training on Modal with real-time monitoring.
    """
    import sys
    import subprocess
    import torch
    import re

    # Setup Python path and working directory
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace/QuintNet")

    # Print startup info
    print("="*80)
    print("🚀 QUINTNET 3D PARALLEL TRAINING")
    print("="*80)

    # Display GPU information
    num_gpus = torch.cuda.device_count()
    print(f"📊 Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    # Verify dataset is mounted
    if os.path.exists("/mnt/dataset/mnist"):
        print(f"✅ Dataset found at /mnt/dataset/mnist")
    else:
        print("❌ ERROR: Dataset not found at /mnt/dataset/mnist")
        # As a fallback, create the directory for the script to download into
        os.makedirs("/mnt/dataset/mnist", exist_ok=True)
        print("   Created directory /mnt/dataset/mnist. Dataset will be downloaded.")

    # Verify code is mounted
    print(f"✅ Code directory mounted: /workspace/QuintNet")

    print("="*80)

    # Setup distributed training environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "8"

    # Command to run distributed training for QuintNet
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "examples/full_3d.py"
    ]

    print(f"🔥 Launching: {' '.join(cmd)}")
    print("="*80)
    print()

    # Start training process with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # Stream and parse output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip()
            print(line) # Print every line for full visibility

            # Highlight epoch progress
            if line.startswith("Epoch") and "/" in line:
                 print("\n" + "="*80 + f"\n🔄 {line.upper()}\n" + "="*80)

            # Highlight training metrics
            if "Train Loss:" in line:
                train_match = re.search(r'Train Loss: ([\d.]+) \| Train Acc: ([\d.]+)%', line)
                if train_match:
                    loss, acc = train_match.groups()
                    print(f"📈 TRAIN | Loss: {loss} | Accuracy: {acc}%")

            # Highlight validation metrics
            if "Val Loss:" in line:
                val_match = re.search(r'Val Loss:   ([\d.]+) \| Val Acc:   ([\d.]+)%', line)
                if val_match:
                    loss, acc = val_match.groups()
                    print(f"✅ VALIDATION | Loss: {loss} | Accuracy: {acc}%")

    # Get final return code
    return_code = process.poll()

    print()
    print("="*80)
    if return_code == 0:
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
    else:
        print(f"❌ TRAINING FAILED WITH ERROR CODE: {return_code}")
    print("="*80)

    return return_code

@app.local_entrypoint()
def main():
    """
    Local entrypoint to run the training from your machine.
    Usage: modal run QuintNet/train_modal_run.py
    """
    print("🚀 Starting QuintNet 3D Parallel Training on Modal...")
    print()
    result = run_3d_parallel_training.remote()
    print()
    if result == 0:
        print("✅ Training job completed successfully!")
    else:
        print(f"❌ Training job failed with return code: {result}")

if __name__ == "__main__":
    # This allows you to also run: python QuintNet/train_modal_run.py
    with app.run():
        main()
