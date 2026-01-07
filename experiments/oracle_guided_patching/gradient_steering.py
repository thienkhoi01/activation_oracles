"""
Gradient-enabled steering hooks for activation perturbation optimization.

This module provides steering hooks that preserve gradients, allowing backpropagation
through the oracle to optimize activation perturbations.

Key difference from nl_probes/utils/steering_hooks.py:
- No .detach() calls on steering vectors
- Designed for optimization, not just inference
"""

import contextlib
from typing import Callable

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module.
    
    Identical to the version in steering_hooks.py, included here for self-containment.
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_gradient_steering_hook(
    vectors: list[torch.Tensor],  # len B, each tensor is (K_b, d_model) - MAY have requires_grad=True
    positions: list[list[int]],  # len B, each list has length K_b
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Steering hook that preserves gradients for optimization.
    
    This is a gradient-enabled version of get_hf_activation_steering_hook from
    nl_probes/utils/steering_hooks.py. The key differences are:
    
    1. No .detach() on normalized vectors - allows gradient flow
    2. No .detach() on steered output - allows gradient flow back to input vectors
    
    Semantics:
        For each batch item b and slot k, the residual at token index positions[b][k]
        is modified to: original + normalize(vectors[b][k]) * ||original|| * coefficient
        
    This additive formulation (rather than replacement) helps with gradient stability.
    
    Args:
        vectors: List of steering vectors, one per batch element. Each has shape (K_b, d_model).
                 These can have requires_grad=True for optimization.
        positions: List of position lists, one per batch element. positions[b][k] is the
                   token index where vectors[b][k] should be injected.
        steering_coefficient: Multiplier for the steering effect.
        device: Device for tensor operations.
        dtype: Dtype for steering computations.
        
    Returns:
        A hook function to be registered on a transformer layer.
    """
    assert len(vectors) == len(positions), "vectors and positions must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    # Pre-normalize vectors - NO .detach() to allow gradient flow
    normed_list = [F.normalize(v_b, dim=-1) for v_b in vectors]

    def hook_fn(module, _input, output):
        # Handle different output formats across model families
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")

        # Skip single-token forward passes (during generation's decoding phase)
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        # Apply steering to each batch element
        for b in range(B):
            pos_b = positions[b]
            pos_tensor = torch.tensor(pos_b, dtype=torch.long, device=device)
            
            assert pos_tensor.min() >= 0, f"Negative position: {pos_tensor.min()}"
            assert pos_tensor.max() < L, f"Position {pos_tensor.max()} >= sequence length {L}"
            
            # Get original activations at steering positions
            orig_KD = resid_BLD[b, pos_tensor, :]  # (K_b, d_model)
            
            # Compute norms for scaling (detach to avoid norm being part of gradient path)
            # We want gradients w.r.t. direction, not magnitude
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True).detach()  # (K_b, 1)
            
            # Build steered vectors - NO .detach() on normed_list[b]
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            
            # ADDITIVE steering - key for gradient flow
            # NO .detach() here - this is where gradients flow back
            resid_BLD[b, pos_tensor, :] = orig_KD + steered_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


def get_gradient_steering_hook_replacement(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Alternative steering hook that REPLACES activations instead of adding.
    
    This matches the original behavior from steering_hooks.py more closely:
    resid[pos] = normalize(vector) * ||original|| * coefficient
    
    May be useful for comparison experiments.
    """
    assert len(vectors) == len(positions)
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")

    normed_list = [F.normalize(v_b, dim=-1) for v_b in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch")

        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        for b in range(B):
            pos_tensor = torch.tensor(positions[b], dtype=torch.long, device=device)
            orig_KD = resid_BLD[b, pos_tensor, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True).detach()
            
            # REPLACEMENT instead of addition
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_tensor, :] = steered_KD  # No + orig_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


def get_gradient_steering_hook_frozen_norm(
    vectors: list[torch.Tensor],  # X = base + delta, where delta is learnable
    base_norms: list[torch.Tensor],  # ||base|| for each batch element (frozen)
    positions: list[list[int]],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Steering hook with frozen base normalization for clean gradient flow.
    
    Formula: h' = h + X * ||h|| / ||base|| * coefficient
    
    Where:
    - X = base + delta (the perturbed activations)
    - ||base|| is the frozen norm of the original base activations
    - ||h|| is the norm of the hidden state at the injection point (detached)
    
    Benefits:
    - At delta=0: exactly matches original steering (h' = h + base_normalized * ||h||)
    - Clean gradients: dh'/d(delta) = ||h|| / ||base|| (constant scaling)
    - Perturbation magnitude is meaningful and directly affects output
    - No vanishing gradients as ||delta|| grows
    
    Args:
        vectors: List of steering vectors (base + delta), one per batch element.
                 Each has shape [num_positions, d_model].
        base_norms: List of frozen base activation norms, one per batch element.
                    Each has shape [num_positions, 1] or [num_positions].
        positions: List of position lists for steering injection.
        steering_coefficient: Multiplier for steering strength.
        device: Device for tensor operations.
        dtype: Dtype for computations.
        
    Returns:
        Hook function for transformer layer.
    """
    assert len(vectors) == len(positions) == len(base_norms), \
        "vectors, positions, and base_norms must have same batch length"
    B = len(vectors)
    if B == 0:
        raise ValueError("Empty batch")
    
    # Ensure base_norms are detached and properly shaped
    frozen_base_norms = []
    for bn in base_norms:
        bn_detached = bn.detach().to(device, dtype)
        if bn_detached.dim() == 1:
            bn_detached = bn_detached.unsqueeze(-1)  # [K] -> [K, 1]
        frozen_base_norms.append(bn_detached)

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model = resid_BLD.shape
        if B_actual != B:
            raise ValueError(f"Batch mismatch: module B={B_actual}, provided vectors B={B}")

        # Skip single-token forward passes (decoding phase)
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        for b in range(B):
            pos_b = positions[b]
            pos_tensor = torch.tensor(pos_b, dtype=torch.long, device=device)
            
            assert pos_tensor.min() >= 0, f"Negative position: {pos_tensor.min()}"
            assert pos_tensor.max() < L, f"Position {pos_tensor.max()} >= sequence length {L}"
            
            # Get original hidden state and its norm (detached)
            orig_KD = resid_BLD[b, pos_tensor, :]  # [K, d_model]
            h_norms_K1 = orig_KD.norm(dim=-1, keepdim=True).detach()  # [K, 1]
            
            # Scaling factor: ||h|| / ||base||
            # This is constant w.r.t. delta, so gradients flow cleanly
            scale_K1 = (h_norms_K1 / (frozen_base_norms[b] + 1e-8)) * steering_coefficient
            
            # h' = h + X * ||h|| / ||base|| * coeff
            # where X = base + delta (vectors[b])
            steered_KD = (vectors[b] * scale_K1).to(dtype)
            
            # Additive steering
            resid_BLD[b, pos_tensor, :] = orig_KD + steered_KD

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn

