"""
Summarization Metrics for GPT-2 Finetuning

This module provides ROUGE and BLEU metric computation for evaluating
text summarization quality.
"""

from typing import List, Dict, Tuple, Optional
import torch


def compute_rouge_bleu(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE and BLEU scores for summarization evaluation.
    
    Args:
        predictions: List of generated summaries
        references: List of reference summaries
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores
    """
    try:
        from rouge_score import rouge_scorer
        import sacrebleu
    except ImportError as e:
        print(f"Warning: Could not import metrics libraries: {e}")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bleu': 0.0,
        }
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
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
    
    # Average ROUGE scores
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    
    # Compute BLEU score
    try:
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        bleu_score = bleu.score
    except Exception:
        bleu_score = 0.0
    
    return {
        'rouge1': avg_rouge1 * 100,  # Convert to percentage
        'rouge2': avg_rouge2 * 100,
        'rougeL': avg_rougeL * 100,
        'bleu': bleu_score,
    }


def generate_summary(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 128,
    device: torch.device = None,
) -> str:
    """
    Generate a summary for the given input text using greedy decoding.
    
    Args:
        model: The GPT-2 model (can be pipeline-wrapped)
        tokenizer: The tokenizer
        input_text: The article text to summarize
        max_new_tokens: Maximum tokens to generate
        device: The device to use
        
    Returns:
        Generated summary text
    """
    # Format the prompt (same as training)
    prompt = f"{input_text}\n\nTL;DR:"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        max_length=384,  # Leave room for generation
        truncation=True,
        padding='max_length',
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Find length of actual input (not padding)
    input_len = attention_mask.sum().item()
    
    # Simple greedy generation
    generated_ids = input_ids.clone()
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get position ids
            seq_len = generated_ids.size(1)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            
            # Forward pass - handle different model wrappers
            if hasattr(model, 'model') and hasattr(model.model, 'local_module'):
                # Pipeline wrapped model - this won't work for generation
                # Need to use full forward pass
                logits = model.model.local_module(generated_ids, position_ids=position_ids)
            elif hasattr(model, 'local_module'):
                logits = model.local_module(generated_ids, position_ids=position_ids)
            else:
                logits = model(generated_ids, position_ids=position_ids)
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode only the generated part
    generated_text = tokenizer.decode(
        generated_ids[0, input_len:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()


def evaluate_generation(
    model,
    tokenizer,
    val_dataset,
    device: torch.device,
    num_samples: int = 50,
    max_new_tokens: int = 128,
) -> Dict[str, float]:
    """
    Evaluate model generation quality using ROUGE and BLEU metrics.
    
    Args:
        model: The GPT-2 model
        tokenizer: The tokenizer
        val_dataset: Validation dataset with 'article' and 'highlights'
        device: The device to use
        num_samples: Number of samples to evaluate (for speed)
        max_new_tokens: Maximum tokens to generate per sample
        
    Returns:
        Dictionary with ROUGE and BLEU scores
    """
    predictions = []
    references = []
    
    # Sample from validation set
    num_samples = min(num_samples, len(val_dataset))
    
    print(f"Generating {num_samples} summaries for evaluation...")
    
    for i in range(num_samples):
        sample = val_dataset[i]
        article = sample['article']
        reference = sample['highlights']
        
        # Generate summary
        try:
            generated = generate_summary(
                model, tokenizer, article,
                max_new_tokens=max_new_tokens,
                device=device
            )
            predictions.append(generated)
            references.append(reference)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_samples} summaries")
        except Exception as e:
            print(f"  Error generating sample {i}: {e}")
            continue
    
    # Compute metrics
    metrics = compute_rouge_bleu(predictions, references)
    
    return metrics
