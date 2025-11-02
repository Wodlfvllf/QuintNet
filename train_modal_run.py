# run_training.py (at root level)
import modal
import os
from pathlib import Path

app = modal.App("mnist-pipeline-training")

# Use the existing persistent volume with dataset
volume = modal.Volume.from_name("mnist-volume", create_if_missing=False)

# Image with distributed training dependencies AND local code
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch", 
        "torchvision", 
        "tqdm", 
        "numpy",
        "datasets",  # for your arrow files
    ])
    .run_commands("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    # Add local directories to the image (NOT using deprecated Mount)
    .add_local_dir(
        str(Path(__file__).parent / "Mnist-Digit-Classification"), 
        remote_path="/workspace/Mnist-Digit-Classification"
    )
    .add_local_dir(
        str(Path(__file__).parent / "QuintNet"), 
        remote_path="/workspace/QuintNet"
    )
)

@app.function(
    image=image,
    gpu="A10:2",  # 2 GPUs for pipeline parallelism
    volumes={"/mnt/dataset": volume},  # Mount persistent dataset volume
    timeout=7200,  # 2 hour timeout
    concurrency_limit=1,  # Only one training job at a time
)
def run_pipeline_parallel_training():
    """
    Run distributed pipeline parallel training on Modal with real-time monitoring.
    """
    import sys
    import subprocess
    import torch
    import re
    
    # Setup Python path for imports
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    
    # Print startup info
    print("="*80)
    print("🚀 MNIST PIPELINE PARALLEL TRAINING")
    print("="*80)
    
    # Display GPU information
    num_gpus = torch.cuda.device_count()
    print(f"📊 Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Verify dataset is mounted
    if os.path.exists("/mnt/dataset/mnist"):
        print(f"✅ Dataset found at /mnt/dataset/mnist")
        dataset_files = os.listdir("/mnt/dataset/mnist")
        print(f"   Dataset structure: {dataset_files}")
    else:
        print("❌ ERROR: Dataset not found at /mnt/dataset/mnist")
        return 1
    
    # Verify code is mounted
    print(f"✅ Code directories mounted:")
    print(f"   - /workspace/Mnist-Digit-Classification: {os.path.exists('/workspace/Mnist-Digit-Classification')}")
    print(f"   - /workspace/QuintNet: {os.path.exists('/workspace/QuintNet')}")
    
    print("="*80)
    
    # Setup distributed training environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "2"
    
    # Command to run distributed training
    cmd = [
        "torchrun", 
        "--nproc_per_node=2",  # 2 processes for 2 GPUs
        "-m", "Mnist-Digit-Classification.Training_Pipeline_Parallelism.train_pp"
    ]
    
    print(f"🔥 Launching: {' '.join(cmd)}")
    print("="*80)
    print()
    
    # Start training process with real-time output streaming
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        universal_newlines=True,  # Text mode
        bufsize=1  # Line buffered for real-time output
    )
    
    # Track metrics
    current_epoch = 0
    best_acc = 0.0
    training_started = False
    
    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        
        # Check if process finished
        if output == '' and process.poll() is not None:
            break
            
        if output:
            line = output.strip()
            
            # Print every line for full visibility
            print(line)
            
            # Mark when training actually starts
            if "INITIALIZING PIPELINE PARALLEL TRAINING" in line:
                training_started = True
                print("="*80)
                print("🎯 TRAINING INITIALIZATION COMPLETE")
                print("="*80)
            
            # Parse and highlight epoch progress
            if "Epoch" in line and "/" in line and training_started:
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))
                    print()
                    print("="*80)
                    print(f"🔄 EPOCH {current_epoch}/{total_epochs}")
                    print("="*80)
            
            # Highlight training metrics
            if "Train Loss:" in line:
                train_match = re.search(r'Train Loss: ([\d.]+), Train Acc: ([\d.]+)%', line)
                if train_match:
                    loss, acc = train_match.groups()
                    print(f"📈 TRAIN | Loss: {loss} | Accuracy: {acc}%")
            
            # Highlight validation metrics
            if "Val Loss:" in line:
                val_match = re.search(r'Val Loss: ([\d.]+), Val Acc: ([\d.]+)%', line)
                if val_match:
                    loss, acc = val_match.groups()
                    acc_float = float(acc)
                    improvement = "🎉 NEW BEST!" if acc_float > best_acc else ""
                    best_acc = max(best_acc, acc_float)
                    print(f"✅ VALIDATION | Loss: {loss} | Accuracy: {acc}% {improvement}")
                    print()
            
            # Highlight patience/early stopping info
            if "Patience:" in line:
                patience_match = re.search(r'Patience: (\d+)', line)
                if patience_match:
                    patience_left = patience_match.group(1)
                    print(f"⏱️  Patience remaining: {patience_left}")
            
            # Highlight early stopping
            if "Early stopping triggered" in line:
                print("="*80)
                print("⚠️  EARLY STOPPING ACTIVATED")
                print("="*80)
            
            # Highlight final results
            if "Best validation accuracy:" in line:
                final_match = re.search(r'Best validation accuracy: ([\d.]+)%', line)
                if final_match:
                    final_acc = final_match.group(1)
                    print()
                    print("="*80)
                    print(f"🏆 FINAL RESULT: {final_acc}% Best Validation Accuracy")
                    print("="*80)
            
            # Highlight training time
            if "Training completed in" in line:
                time_match = re.search(r'Training completed in (.+)', line)
                if time_match:
                    time_str = time_match.group(1)
                    print(f"⏱️  Total Training Time: {time_str}")
    
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
    Usage: modal run run_training.py
    """
    print("🚀 Starting MNIST Pipeline Parallel Training on Modal...")
    print()
    result = run_pipeline_parallel_training.remote()
    print()
    if result == 0:
        print("✅ Training job completed successfully!")
    else:
        print(f"❌ Training job failed with return code: {result}")


if __name__ == "__main__":
    # This allows you to also run: python run_training.py
    import modal.runner
    with app.run():
        main()
