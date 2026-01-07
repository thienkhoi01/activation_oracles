"""
Learnable activation perturbations for oracle-guided patching.

This module defines the perturbation class that wraps base activations
and adds a learnable delta that can be optimized via gradient descent.
"""

import torch
import torch.nn as nn
from typing import Optional


class ActivationPerturbation(nn.Module):
    """
    Learnable perturbation added to base activations.
    
    Given base activations A from a target model, this class represents
    perturbed activations A' = A + delta, where delta is a learnable parameter.
    
    The delta can be optimized to change what the oracle says about the activations,
    then tested to see if it causally affects the original model's behavior.
    
    Attributes:
        base: The frozen base activations from the target model. Shape: [num_positions, d_model]
        delta: The learnable perturbation. Shape: [num_positions, d_model]
    """
    
    def __init__(
        self,
        base_activations: torch.Tensor,
        init_scale: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the perturbation.
        
        Args:
            base_activations: The base activations to perturb. Shape: [num_positions, d_model]
                              Will be detached and stored as a buffer (not optimized).
            init_scale: Initial scale for random perturbation. Default 0.0 means start at zero.
            device: Device for the perturbation. Defaults to base_activations.device.
            dtype: Dtype for the perturbation. Defaults to base_activations.dtype.
        """
        super().__init__()
        
        if device is None:
            device = base_activations.device
        if dtype is None:
            dtype = base_activations.dtype
            
        # Store base as buffer (not a parameter, won't be optimized)
        self.register_buffer('base', base_activations.detach().clone().to(device, dtype))
        
        # Initialize delta as learnable parameter
        if init_scale == 0.0:
            delta_init = torch.zeros_like(self.base)
        else:
            delta_init = torch.randn_like(self.base) * init_scale
            
        self.delta = nn.Parameter(delta_init)
        
    def forward(self) -> torch.Tensor:
        """
        Return perturbed activations: base + delta.
        
        Returns:
            Perturbed activations with gradients flowing through delta.
            Shape: [num_positions, d_model]
        """
        return self.base + self.delta
    
    @property
    def perturbation_norm(self) -> torch.Tensor:
        """L2 norm of the perturbation delta."""
        return self.delta.norm()
    
    @property  
    def perturbation_relative_norm(self) -> torch.Tensor:
        """Relative norm: ||delta|| / ||base||."""
        base_norm = self.base.norm()
        if base_norm < 1e-8:
            return self.delta.norm()
        return self.delta.norm() / base_norm
    
    @property
    def num_positions(self) -> int:
        """Number of token positions being perturbed."""
        return self.base.shape[0]
    
    @property
    def d_model(self) -> int:
        """Hidden dimension of the model."""
        return self.base.shape[1]
    
    def clamp_norm(self, max_norm: float) -> None:
        """
        Clamp the perturbation to have at most max_norm L2 norm.
        
        This can be called after optimizer steps to constrain perturbation magnitude.
        
        Args:
            max_norm: Maximum allowed L2 norm for the perturbation.
        """
        with torch.no_grad():
            current_norm = self.delta.norm()
            if current_norm > max_norm:
                self.delta.data = self.delta.data * (max_norm / current_norm)
                
    def clamp_elementwise(self, max_val: float) -> None:
        """
        Clamp each element of delta to [-max_val, max_val].
        
        Args:
            max_val: Maximum absolute value for each element.
        """
        with torch.no_grad():
            self.delta.data = self.delta.data.clamp(-max_val, max_val)
            
    def reset(self) -> None:
        """Reset delta to zeros."""
        with torch.no_grad():
            self.delta.data.zero_()
            
    def get_steering_vector(self) -> torch.Tensor:
        """
        Get just the perturbation delta for use as a steering vector.
        
        This returns a detached copy that can be applied to the original model
        without affecting optimization.
        
        Returns:
            Detached copy of delta. Shape: [num_positions, d_model]
        """
        return self.delta.detach().clone()


class BatchedActivationPerturbation(nn.Module):
    """
    Batched version of ActivationPerturbation for handling multiple samples.
    
    Each sample in the batch can have different base activations but shares
    the same perturbation delta (useful for testing the same perturbation
    across multiple prompts).
    
    Alternatively, can have separate deltas per batch element.
    """
    
    def __init__(
        self,
        base_activations_list: list[torch.Tensor],
        shared_delta: bool = False,
        init_scale: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize batched perturbations.
        
        Args:
            base_activations_list: List of base activations, one per batch element.
                                   Each has shape [num_positions_b, d_model].
            shared_delta: If True, use a single shared delta for all batch elements.
                          If False, each batch element has its own delta.
            init_scale: Initial scale for random perturbation.
            device: Device for perturbations.
            dtype: Dtype for perturbations.
        """
        super().__init__()
        
        self.shared_delta = shared_delta
        self.batch_size = len(base_activations_list)
        
        if device is None:
            device = base_activations_list[0].device
        if dtype is None:
            dtype = base_activations_list[0].dtype
            
        # Store base activations as buffers
        for i, base in enumerate(base_activations_list):
            self.register_buffer(f'base_{i}', base.detach().clone().to(device, dtype))
            
        if shared_delta:
            # Single shared delta - all bases must have same shape
            shapes = [b.shape for b in base_activations_list]
            assert all(s == shapes[0] for s in shapes), \
                f"Shared delta requires all bases to have same shape, got {shapes}"
            
            if init_scale == 0.0:
                delta_init = torch.zeros_like(base_activations_list[0])
            else:
                delta_init = torch.randn_like(base_activations_list[0]) * init_scale
            self.delta = nn.Parameter(delta_init.to(device, dtype))
        else:
            # Separate delta per batch element
            for i, base in enumerate(base_activations_list):
                if init_scale == 0.0:
                    delta_init = torch.zeros_like(base)
                else:
                    delta_init = torch.randn_like(base) * init_scale
                setattr(self, f'delta_{i}', nn.Parameter(delta_init.to(device, dtype)))
                
    def forward(self) -> list[torch.Tensor]:
        """
        Return list of perturbed activations.
        
        Returns:
            List of perturbed activations, one per batch element.
        """
        results = []
        for i in range(self.batch_size):
            base = getattr(self, f'base_{i}')
            if self.shared_delta:
                delta = self.delta
            else:
                delta = getattr(self, f'delta_{i}')
            results.append(base + delta)
        return results
    
    def get_steering_vectors(self) -> list[torch.Tensor]:
        """Get detached copies of all deltas."""
        if self.shared_delta:
            return [self.delta.detach().clone() for _ in range(self.batch_size)]
        else:
            return [getattr(self, f'delta_{i}').detach().clone() 
                    for i in range(self.batch_size)]

