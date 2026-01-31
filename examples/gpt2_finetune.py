"""
GPT-2 Finetuning with 3D Parallelism (DP + PP + TP).

This script finetunes GPT-2 on text summarization using CNN/DailyMail dataset.
It uses the full 3D parallelism stack with distributed weight loading.

Usage:
    torchrun --nproc_per_node=8 -m QuintNet.examples.gpt2_finetune \
        --config QuintNet/examples/gpt2_config.yaml \
        --checkpoint /path/to/gpt2.safetensors \
        --dataset /path/to/cnn_dailymail

Requirements:
    - GPT-2 checkpoint in safetensors format
    - CNN/DailyMail dataset with train.csv, validation.csv, test.csv
    - Each CSV should have 'article' and 'highlights' columns
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
from pathlib import Path

# HuggingFace tokenizer
from transformers import GPT2Tokenizer

# QuintNet imports
from ..utils.Dataloader import SummarizationDataset, SummarizationCollator
from ..core import load_config, init_process_groups
from ..strategy import get_strategy
from ..GPT2_Trainer import GPT2Trainer


def main():
    """
    Main function to set up and run GPT-2 finetuning.
    """
    parser = argparse.ArgumentParser(description="GPT-2 Finetuning with 3D Parallelism")
    parser.add_argument('--config', type=str, default='QuintNet/examples/gpt2_config.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to GPT-2 safetensors checkpoint.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory with train.csv, validation.csv.')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='HuggingFace tokenizer name/path (default: gpt2)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit dataset size for testing (e.g. 200).')
    
    args = parser.parse_args()
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. LOAD CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    config = load_config(args.config)
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. INITIALIZE DISTRIBUTED ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    global_rank = dist.get_rank()
    
    if global_rank == 0:
        print("=" * 80)
        print("🚀 GPT-2 FINETUNING WITH 3D PARALLELISM")
        print("=" * 80)
        print(f"   Checkpoint: {args.checkpoint}")
        print(f"   Dataset: {args.dataset}")
        print(f"   Tokenizer: {args.tokenizer}")
        print(f"   GPUs: {dist.get_world_size()}")
        print("=" * 80)
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. INITIALIZE PROCESS GROUPS
    # ═══════════════════════════════════════════════════════════════════
    pg_manager = init_process_groups(
        device_type=config['device_type'],
        mesh_dim=config['mesh_dim'],
        mesh_name=config['mesh_name']
    )
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Get coordinates for data parallel sharding
    coords = pg_manager.get_coordinates_tensor_search(global_rank)
    dp_rank = coords[config['mesh_name'].index('dp')]
    dp_size = config['mesh_dim'][config['mesh_name'].index('dp')]
    
    if global_rank == 0:
        print(f"[Init] DP={dp_size}, TP={config['mesh_dim'][config['mesh_name'].index('tp')]}, "
              f"PP={config['mesh_dim'][config['mesh_name'].index('pp')]}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. LOAD TOKENIZER
    # ═══════════════════════════════════════════════════════════════════
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if global_rank == 0:
        print(f"[Tokenizer] Loaded with vocab_size={tokenizer.vocab_size}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. CREATE DATASETS
    # ═══════════════════════════════════════════════════════════════════
    dataset_path = Path(args.dataset)
    
    train_dataset = SummarizationDataset(dataset_path, split='train')
    
    # Allow controlling dataset size via config or CLI (config takes precedence if both exist, or CLI overrides? Usually CLI overrides config)
    # But user asked to embed in config. Let's support both: CLI > Config > None.
    max_samples = args.max_samples if args.max_samples is not None else config.get('max_samples', None)
    
    if max_samples is not None:
        if global_rank == 0:
            print(f"[Dataset] Limiting train dataset to {max_samples} samples.")
        train_dataset.data = train_dataset.data.iloc[:max_samples]
        
    val_dataset = SummarizationDataset(dataset_path, split='validation')
    
    # Also limit validation dataset if configured
    max_val_samples = config.get('max_val_samples', None)
    if max_val_samples is not None:
        if global_rank == 0:
            print(f"[Dataset] Limiting val dataset to {max_val_samples} samples.")
        val_dataset.data = val_dataset.data.iloc[:max_val_samples]
    
    if global_rank == 0:
        print(f"[Dataset] Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # ═══════════════════════════════════════════════════════════════════
    # 6. CREATE COLLATOR AND DATALOADERS
    # ═══════════════════════════════════════════════════════════════════
    collator = SummarizationCollator(
        tokenizer=tokenizer,
        max_length=config.get('max_seq_length', 512),
        max_target_length=config.get('max_target_length', 128),
    )
    
    # Distributed samplers for DP
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=dp_size, 
        rank=dp_rank, 
        shuffle=True, 
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=dp_size, 
        rank=dp_rank, 
        shuffle=False, 
        seed=42
    )
    
    # Calculate micro-batch size for pipeline parallelism
    micro_batch_size = config['batch_size'] // (config['grad_acc_steps'] * dp_size)
    if micro_batch_size < 1:
        micro_batch_size = 1
    
    if global_rank == 0:
        print(f"[DataLoader] micro_batch_size={micro_batch_size}, grad_acc_steps={config['grad_acc_steps']}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=config.get('num_workers', 0),
        drop_last=True,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,  # Large batch for fast validation (no gradients)
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=config.get('num_workers', 0),
        drop_last=True,
        pin_memory=True,
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # 7. LOAD MODEL WITH DISTRIBUTED STRATEGY
    # ═══════════════════════════════════════════════════════════════════
    if global_rank == 0:
        print(f"[Model] Loading from checkpoint with 3D parallelism...")
    
    # In staged mode, model is built from checkpoint
    strategy = get_strategy(
        config['strategy_name'],
        pg_manager,
        config,
        checkpoint_path=args.checkpoint,
        is_staged=True,  # Enable distributed loading
    )
    
    # Apply parallelization (model=None means build from checkpoint)
    parallel_model = strategy.apply(None)
    
    if global_rank == 0:
        print(f"[Model] Parallelized successfully!")
    
    # ═══════════════════════════════════════════════════════════════════
    # 8. CREATE TRAINER AND START TRAINING
    # ═══════════════════════════════════════════════════════════════════
    
    # Get the model config for vocab size etc.
    from ..utils.GPT2 import GPT2Config
    if 'model_config' in config:
        model_config = GPT2Config.from_dict(config['model_config'])
    else:
        model_config = GPT2Config()  # Use defaults
    
    trainer = GPT2Trainer(
        model=parallel_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        pg_manager=pg_manager,
        model_config=model_config,
    )
    
    if global_rank == 0:
        print("\n" + "=" * 80)
        print("🔥 STARTING TRAINING")
        print("=" * 80 + "\n")
    
    # Start training
    trainer.fit()
    
    # ═══════════════════════════════════════════════════════════════════
    # 9. CLEANUP
    # ═══════════════════════════════════════════════════════════════════
    dist.barrier()
    dist.destroy_process_group()
    
    if global_rank == 0:
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()
