"""
Oracle-Guided Activation Patching

This module implements gradient-based optimization to find activation perturbations
that change what the oracle says, then tests whether these perturbations causally
affect the original model's behavior.

Main components:
- gradient_steering: Gradient-enabled steering hooks
- perturbation: Learnable perturbation classes
- oracle_forward: Differentiable oracle forward pass
"""

from .gradient_steering import (
    get_gradient_steering_hook,
    get_gradient_steering_hook_replacement,
    get_gradient_steering_hook_frozen_norm,
    add_hook,
)
from .perturbation import (
    ActivationPerturbation,
    BatchedActivationPerturbation,
)
from .oracle_forward import (
    oracle_forward_with_gradients,
    oracle_forward_frozen_norm,
    compute_target_token_loss,
    compute_next_token_distribution,
    compute_token_probability,
    maximize_entropy_loss,
    push_away_from_tokens_loss,
)

__all__ = [
    # Steering hooks
    "get_gradient_steering_hook",
    "get_gradient_steering_hook_replacement",
    "get_gradient_steering_hook_frozen_norm",
    "add_hook",
    # Perturbation classes
    "ActivationPerturbation",
    "BatchedActivationPerturbation",
    # Oracle forward
    "oracle_forward_with_gradients",
    "oracle_forward_frozen_norm",
    "compute_target_token_loss",
    "compute_next_token_distribution",
    "compute_token_probability",
    "maximize_entropy_loss",
    "push_away_from_tokens_loss",
]

