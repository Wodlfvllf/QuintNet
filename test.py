#!/usr/bin/env python3
"""
Single-GPU GPT-2 Test Script

This script loads a GPT-2 model on a single GPU and evaluates it on a test dataset.
It computes:
  - Loss and Perplexity (token-level language modeling metrics)
  - ROUGE and BLEU scores (generation-based summarization metrics)

This serves as a baseline to verify that distributed training metrics are correct.

Usage:
    python test.py --checkpoint /path/to/model.safetensors --dataset /path/to/dataset
    
    # On Modal:
    modal run QuintNet/test.py --checkpoint /mnt/model/model.safetensors --dataset /mnt/dataset/cnn_dailymail
"""

import argparse
import math
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer


def load_gpt2_model(checkpoint_path: str, device: torch.device):
    """
    Load GPT-2 model from checkpoint.
    
    Supports:
        - .safetensors: Original pretrained checkpoint
        - .pt: Merged finetuned checkpoint from 3D parallel training
    """
    from transformers import GPT2LMHeadModel, GPT2Config as HFConfig
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load HuggingFace GPT-2 config
    hf_config = HFConfig(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        attn_pdrop=0.1,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
    )
    
    # Create model
    model = GPT2LMHeadModel(hf_config)
    
    # Determine checkpoint format
    if checkpoint_path.endswith('.safetensors'):
        # Original pretrained checkpoint
        from safetensors import safe_open
        
        state_dict = {}
        with safe_open(checkpoint_path, framework='pt') as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        # GPT-2 safetensors uses keys without 'transformer.' prefix
        # HuggingFace GPT2LMHeadModel expects 'transformer.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            # Handle Conv1D to Linear weight transpose
            if 'attn.c_attn.weight' in key or 'attn.c_proj.weight' in key:
                value = value.t()
            elif 'mlp.c_fc.weight' in key or 'mlp.c_proj.weight' in key:
                value = value.t()
            
            new_state_dict[f'transformer.{key}'] = value
        
        # Handle lm_head weight tying
        if 'transformer.wte.weight' in new_state_dict:
            new_state_dict['lm_head.weight'] = new_state_dict['transformer.wte.weight']
        
        state_dict = new_state_dict
        print("  Format: safetensors (original pretrained)")
        
    elif checkpoint_path.endswith('.pt'):
        # Merged finetuned checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        print("  Format: .pt (merged finetuned)")
        
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")
    
    # Load with strict=False to handle any mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        if len(missing) < 10:
            for k in missing:
                print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        if len(unexpected) < 10:
            for k in unexpected:
                print(f"    - {k}")
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def load_test_dataset(dataset_path: str, split: str = 'validation'):
    """Load test dataset."""
    import pandas as pd
    
    csv_path = Path(dataset_path) / f'{split}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} samples from {split} split")
    
    return df


def compute_loss_metrics(
    model,
    tokenizer,
    dataset,
    device,
    max_samples: int = 200,
    max_length: int = 512,
    batch_size: int = 4,
):
    """Compute loss and perplexity on test dataset."""
    print(f"\n{'='*60}")
    print("📊 COMPUTING LOSS METRICS")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    total_loss = 0.0
    total_tokens = 0
    num_samples = min(max_samples, len(dataset))
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Computing loss"):
            batch_end = min(i + batch_size, num_samples)
            batch_texts = []
            
            for j in range(i, batch_end):
                article = dataset.iloc[j]['article']
                highlights = dataset.iloc[j]['highlights']
                text = f"{article}\n\nTL;DR: {highlights}"
                batch_texts.append(text)
            
            # Tokenize
            encodings = tokenizer(
                batch_texts,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length',
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Create labels (same as input_ids, with padding masked)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss
            # Shift for CLM: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Accumulate
            num_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    print(f"\n📈 Loss Metrics ({num_samples} samples):")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Perplexity:   {perplexity:.2f}")
    
    return avg_loss, perplexity


def compute_generation_metrics(
    model,
    tokenizer,
    dataset,
    device,
    max_samples: int = 50,
    max_new_tokens: int = 128,
):
    """Compute ROUGE and BLEU scores using generation."""
    print(f"\n{'='*60}")
    print("📊 COMPUTING GENERATION METRICS (ROUGE/BLEU)")
    print(f"{'='*60}")
    
    try:
        from rouge_score import rouge_scorer
        import sacrebleu
    except ImportError as e:
        print(f"⚠️  Could not import metrics libraries: {e}")
        return None
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    predictions = []
    references = []
    num_samples = min(max_samples, len(dataset))
    
    model.eval()
    print(f"Generating {num_samples} summaries...")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Generating"):
            article = dataset.iloc[i]['article']
            reference = dataset.iloc[i]['highlights']
            
            # Create prompt
            prompt = f"{article}\n\nTL;DR:"
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors='pt',
                max_length=384,  # Leave room for generation
                truncation=True,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Decode only the generated part
            generated_ids = outputs[0, input_ids.size(1):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            predictions.append(generated_text.strip())
            references.append(reference)
    
    # Compute ROUGE scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    
    # Compute BLEU
    try:
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        bleu_score = bleu.score
    except Exception:
        bleu_score = 0.0
    
    print(f"\n📈 Generation Metrics ({num_samples} samples):")
    print(f"   ROUGE-1: {avg_rouge1 * 100:.2f}")
    print(f"   ROUGE-2: {avg_rouge2 * 100:.2f}")
    print(f"   ROUGE-L: {avg_rougeL * 100:.2f}")
    print(f"   BLEU:    {bleu_score:.2f}")
    
    # Show a few examples
    print(f"\n{'='*60}")
    print("📝 SAMPLE GENERATIONS")
    print(f"{'='*60}")
    for i in range(min(3, len(predictions))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Reference: {references[i][:200]}...")
        print(f"Generated: {predictions[i][:200]}...")
    
    return {
        'rouge1': avg_rouge1 * 100,
        'rouge2': avg_rouge2 * 100,
        'rougeL': avg_rougeL * 100,
        'bleu': bleu_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Single-GPU GPT-2 Test Script")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model.safetensors checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory containing CSV files')
    parser.add_argument('--split', type=str, default='validation',
                        help='Dataset split to use (default: validation)')
    parser.add_argument('--max_loss_samples', type=int, default=200,
                        help='Max samples for loss computation (default: 200)')
    parser.add_argument('--max_gen_samples', type=int, default=50,
                        help='Max samples for generation metrics (default: 50)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for loss computation (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧪 GPT-2 SINGLE-GPU TEST SCRIPT")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print("=" * 70)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded (vocab_size={tokenizer.vocab_size})")
    
    # Load model
    model = load_gpt2_model(args.checkpoint, device)
    
    # Load dataset
    dataset = load_test_dataset(args.dataset, args.split)
    
    # Compute loss metrics
    loss, ppl = compute_loss_metrics(
        model, tokenizer, dataset, device,
        max_samples=args.max_loss_samples,
        batch_size=args.batch_size,
    )
    
    # Compute generation metrics
    gen_metrics = compute_generation_metrics(
        model, tokenizer, dataset, device,
        max_samples=args.max_gen_samples,
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Loss:       {loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    if gen_metrics:
        print(f"ROUGE-1:    {gen_metrics['rouge1']:.2f}")
        print(f"ROUGE-2:    {gen_metrics['rouge2']:.2f}")
        print(f"ROUGE-L:    {gen_metrics['rougeL']:.2f}")
        print(f"BLEU:       {gen_metrics['bleu']:.2f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
