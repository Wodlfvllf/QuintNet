"""
Modal wrapper for running test.py on cloud GPU.

Usage:
    modal run QuintNet/test_modal_run.py
"""

import modal
from pathlib import Path

app = modal.App("quintnet-gpt2-test")

# Reuse existing volumes
model_volume = modal.Volume.from_name("gpt2-model-volume", create_if_missing=True)
dataset_volume = modal.Volume.from_name("cnn-dailymail-volume", create_if_missing=True)

# Image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "transformers",
        "safetensors",
        "pandas",
        "tqdm",
        "rouge-score",
        "sacrebleu",
        "nltk",
    ])
    .run_commands("pip install torch --index-url https://download.pytorch.org/whl/cu121")
    .add_local_dir(
        local_path=Path(__file__).parent,
        remote_path="/workspace/QuintNet"
    )
)


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/mnt/model": model_volume,
        "/mnt/dataset": dataset_volume,
    },
    timeout=1800,  # 30 minutes
)
def run_test(
    max_loss_samples: int = 200,
    max_gen_samples: int = 50,
):
    """Run single-GPU test on Modal."""
    import subprocess
    import sys
    import os
    
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    
    cmd = [
        "python", "-m", "QuintNet.test",
        "--checkpoint", "/mnt/model/model.safetensors",
        "--dataset", "/mnt/dataset/cnn_dailymail",
        "--split", "validation",
        "--max_loss_samples", str(max_loss_samples),
        "--max_gen_samples", str(max_gen_samples),
        "--batch_size", "4",
    ]
    
    print(f"🔥 Running: {' '.join(cmd)}")
    print("=" * 70)
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


@app.local_entrypoint()
def main(
    max_loss_samples: int = 200,
    max_gen_samples: int = 50,
):
    """Run GPT-2 test on Modal."""
    print("🧪 Starting GPT-2 Single-GPU Test on Modal...")
    print(f"   Loss samples: {max_loss_samples}")
    print(f"   Gen samples:  {max_gen_samples}")
    print()
    
    result = run_test.remote(
        max_loss_samples=max_loss_samples,
        max_gen_samples=max_gen_samples,
    )
    
    if result == 0:
        print("\n✅ Test completed successfully!")
    else:
        print(f"\n❌ Test failed with code: {result}")
