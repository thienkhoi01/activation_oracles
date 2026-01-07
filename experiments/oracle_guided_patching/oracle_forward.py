"""
Differentiable forward pass through the activation oracle.

This module provides functions to run the oracle in a way that preserves
gradients, enabling optimization of activation perturbations.

Key difference from the standard oracle evaluation:
- Uses model.forward() instead of model.generate()
- Returns logits instead of generated text
- Does NOT wrap in torch.no_grad()
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from .gradient_steering import get_gradient_steering_hook, get_gradient_steering_hook_frozen_norm, add_hook


def oracle_forward_with_gradients(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steering_vectors: list[torch.Tensor],
    positions: list[list[int]],
    injection_submodule: torch.nn.Module,
    steering_coefficient: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Forward pass through oracle with gradient-enabled steering.
    
    This runs the oracle model with activation steering and returns logits
    (not generated text), allowing gradients to flow back through the
    steering vectors for optimization.
    
    Args:
        model: The oracle model (with LoRA adapter already set).
        input_ids: Token IDs for the oracle prompt. Shape: [batch, seq_len]
        attention_mask: Attention mask. Shape: [batch, seq_len]
        steering_vectors: List of steering vectors, one per batch element.
                          Each has shape [num_positions, d_model].
                          These can have requires_grad=True for optimization.
        positions: List of position lists. positions[b] contains the token
                   indices in the oracle prompt where steering should be applied.
        injection_submodule: The model layer where steering is injected (typically layer 1).
        steering_coefficient: Multiplier for steering strength. Default 1.0.
        device: Device for computations. Defaults to input_ids.device.
        dtype: Dtype for steering. Defaults to model.dtype.
        
    Returns:
        logits: Shape [batch, seq_len, vocab_size]. The model's output logits
                with gradients flowing through steering_vectors.
    """
    if device is None:
        device = input_ids.device
    if dtype is None:
        dtype = next(model.parameters()).dtype
        
    # Create gradient-enabled steering hook
    hook_fn = get_gradient_steering_hook(
        vectors=steering_vectors,
        positions=positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    
    # Forward pass with steering hook applied
    with add_hook(injection_submodule, hook_fn):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    
    return outputs.logits


def oracle_forward_frozen_norm(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    steering_vectors: list[torch.Tensor],
    base_norms: list[torch.Tensor],
    positions: list[list[int]],
    injection_submodule: torch.nn.Module,
    steering_coefficient: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Forward pass through oracle with frozen-norm steering for clean gradients.
    
    Uses the formula: h' = h + X * ||h|| / ||base|| * coefficient
    
    This gives clean gradient flow where dh'/d(delta) = ||h|| / ||base||,
    a simple constant scaling factor.
    
    Args:
        model: The oracle model (with LoRA adapter already set).
        input_ids: Token IDs for the oracle prompt. Shape: [batch, seq_len]
        attention_mask: Attention mask. Shape: [batch, seq_len]
        steering_vectors: List of steering vectors (base + delta), one per batch element.
                          Each has shape [num_positions, d_model].
        base_norms: List of frozen base activation norms (||base||), one per batch element.
                    Each has shape [num_positions] or [num_positions, 1].
        positions: List of position lists for steering injection.
        injection_submodule: The model layer where steering is injected.
        steering_coefficient: Multiplier for steering strength. Default 1.0.
        device: Device for computations.
        dtype: Dtype for steering.
        
    Returns:
        logits: Shape [batch, seq_len, vocab_size].
    """
    if device is None:
        device = input_ids.device
    if dtype is None:
        dtype = next(model.parameters()).dtype
    
    hook_fn = get_gradient_steering_hook_frozen_norm(
        vectors=steering_vectors,
        base_norms=base_norms,
        positions=positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    
    with add_hook(injection_submodule, hook_fn):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
    
    return outputs.logits


def compute_target_token_loss(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute cross-entropy loss for specific target tokens at specific positions.
    
    This is useful for optimizing toward a specific oracle response.
    For example, if we want the oracle to say "frown" instead of "smile",
    we compute the loss for the "frown" tokens.
    
    Args:
        logits: Model output logits. Shape: [batch, seq_len, vocab_size]
        target_token_ids: Target token IDs. Shape: [batch, num_targets] or [num_targets]
        target_positions: Positions where targets should appear. Shape: [batch, num_targets] or [num_targets]
        reduction: How to reduce the loss. "mean", "sum", or "none".
        
    Returns:
        Loss value (scalar if reduction != "none", else per-position losses).
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Handle unbatched inputs
    if target_token_ids.dim() == 1:
        target_token_ids = target_token_ids.unsqueeze(0).expand(batch_size, -1)
    if target_positions.dim() == 1:
        target_positions = target_positions.unsqueeze(0).expand(batch_size, -1)
        
    # Gather logits at target positions
    # target_positions: [batch, num_targets]
    # We need logits at position p-1 to predict token at position p
    # (standard autoregressive setup)
    pred_positions = target_positions - 1
    
    losses = []
    for b in range(batch_size):
        for i in range(target_token_ids.shape[1]):
            pos = pred_positions[b, i]
            target = target_token_ids[b, i]
            
            if pos < 0 or pos >= seq_len:
                continue
                
            # Cross-entropy for this position
            logit = logits[b, pos, :]  # [vocab_size]
            loss = torch.nn.functional.cross_entropy(
                logit.unsqueeze(0),
                target.unsqueeze(0),
                reduction='none'
            )
            losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    losses = torch.stack(losses)
    
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        return losses


def compute_next_token_distribution(
    logits: torch.Tensor,
    generation_position: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Get the probability distribution for the next token at a specific position.
    
    Args:
        logits: Model output logits. Shape: [batch, seq_len, vocab_size]
        generation_position: The position whose next-token distribution to return.
                             Typically the last position of the prompt.
        temperature: Temperature for softmax. Default 1.0.
        
    Returns:
        probabilities: Shape [batch, vocab_size]
    """
    # Logits at generation_position predict token at generation_position+1
    next_token_logits = logits[:, generation_position, :]  # [batch, vocab_size]
    
    if temperature != 1.0:
        next_token_logits = next_token_logits / temperature
        
    return torch.nn.functional.softmax(next_token_logits, dim=-1)


def compute_token_probability(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    position: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute the probability of specific tokens at a position.
    
    Useful for computing how likely the oracle is to output a specific word.
    
    Args:
        logits: Model output logits. Shape: [batch, seq_len, vocab_size]
        token_ids: Token IDs to get probabilities for. Shape: [batch] or [batch, num_tokens]
        position: Position in sequence (logits at position predict token at position+1).
        temperature: Temperature for softmax.
        
    Returns:
        probabilities: Shape [batch] or [batch, num_tokens]
    """
    probs = compute_next_token_distribution(logits, position, temperature)
    
    if token_ids.dim() == 1:
        # Single token per batch element
        return probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    else:
        # Multiple tokens per batch element
        return probs.gather(1, token_ids)


def maximize_entropy_loss(
    logits: torch.Tensor,
    position: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Loss that maximizes entropy (uncertainty) of the next token distribution.
    
    Useful for finding perturbations that make the oracle uncertain.
    
    Args:
        logits: Model output logits. Shape: [batch, seq_len, vocab_size]
        position: Position whose distribution to maximize entropy for.
        temperature: Temperature for softmax.
        
    Returns:
        Negative entropy (to minimize for maximum uncertainty).
    """
    probs = compute_next_token_distribution(logits, position, temperature)
    
    # Entropy = -sum(p * log(p))
    # We return negative entropy so minimizing the loss maximizes entropy
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return -entropy.mean()


def push_away_from_tokens_loss(
    logits: torch.Tensor,
    avoid_token_ids: torch.Tensor,
    position: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Loss that pushes away from specific tokens (increases their negative log-prob).
    
    Useful for making the oracle NOT say something specific.
    
    Args:
        logits: Model output logits. Shape: [batch, seq_len, vocab_size]
        avoid_token_ids: Token IDs to avoid. Shape: [num_tokens]
        position: Position where we want to avoid these tokens.
        temperature: Temperature for softmax.
        
    Returns:
        Negative of the probability of avoid tokens (minimize to reduce their probability).
    """
    probs = compute_next_token_distribution(logits, position, temperature)
    
    # Sum probability mass on tokens to avoid
    avoid_probs = probs[:, avoid_token_ids].sum(dim=-1)
    
    # Minimize this to push probability away from these tokens
    return avoid_probs.mean()

