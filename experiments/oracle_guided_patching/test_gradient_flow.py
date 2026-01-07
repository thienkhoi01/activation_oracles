"""
Test script to verify gradient flow through the oracle.

This script:
1. Loads the model and oracle adapter
2. Extracts activations from a target prompt
3. Creates a learnable perturbation
4. Runs a forward pass through the oracle
5. Computes a loss and backprops
6. Verifies that gradients flow to the perturbation

Run with:
    cd /root/activation_oracles
    source .venv/bin/activate
    python experiments/oracle_guided_patching/test_gradient_flow.py
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
sys.path.insert(0, "/root/activation_oracles")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.dataset_utils import SPECIAL_TOKEN, get_introspection_prefix

from experiments.oracle_guided_patching import (
    ActivationPerturbation,
    oracle_forward_with_gradients,
    get_gradient_steering_hook,
    add_hook,
)


def test_gradient_flow():
    """Main test function for gradient flow verification."""
    
    print("=" * 70)
    print("GRADIENT FLOW TEST")
    print("=" * 70)
    
    # Configuration
    model_name = "Qwen/Qwen3-8B"
    oracle_lora_path = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # --- Step 1: Load model and tokenizer ---
    print("\n[1/6] Loading model and tokenizer...")
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    
    # Add dummy adapter then load oracle
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")
    model.load_adapter(oracle_lora_path, adapter_name=oracle_lora_path, 
                       is_trainable=False, low_cpu_mem_usage=True)
    model.set_adapter(oracle_lora_path)
    
    print("   ✓ Model loaded")
    
    # --- Step 2: Create target prompt and extract activations ---
    print("\n[2/6] Extracting activations from target prompt...")
    
    # Simple target prompt
    target_prompt = "The secret word is hidden in my mind."
    target_messages = [{"role": "user", "content": target_prompt}]
    formatted_target = tokenizer.apply_chat_template(
        target_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    
    target_inputs = tokenizer(formatted_target, return_tensors="pt").to(device)
    
    # Extract activations at layer 50% (layer 18 for Qwen3-8B)
    act_layer = 18
    submodule = get_hf_submodule(model, act_layer)
    
    model.disable_adapters()  # Get base model activations
    with torch.no_grad():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules={act_layer: submodule},
            inputs_BL=target_inputs,
            min_offset=None,
            max_offset=None,
        )
    model.enable_adapters()
    model.set_adapter(oracle_lora_path)
    
    # Get activations for last few tokens (where interesting info is)
    base_activations = acts_by_layer[act_layer][0, -5:, :]  # [5, d_model]
    print(f"   ✓ Extracted activations: shape {base_activations.shape}")
    
    # --- Step 3: Create learnable perturbation ---
    print("\n[3/6] Creating learnable perturbation...")
    
    perturbation = ActivationPerturbation(base_activations, device=device, dtype=dtype)
    print(f"   ✓ Perturbation created: {perturbation.num_positions} positions, {perturbation.d_model} dims")
    print(f"   Initial perturbation norm: {perturbation.perturbation_norm.item():.6f}")
    
    # --- Step 4: Build oracle input ---
    print("\n[4/6] Building oracle input prompt...")
    
    # Oracle prompt format: "Layer: {layer}\n" + " ?" * num_positions + " \n{question}"
    num_positions = perturbation.num_positions
    oracle_question = "What is hidden in the model's mind?"
    prefix = get_introspection_prefix(act_layer, num_positions)
    oracle_prompt = prefix + oracle_question
    
    oracle_messages = [{"role": "user", "content": oracle_prompt}]
    formatted_oracle = tokenizer.apply_chat_template(
        oracle_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    
    oracle_inputs = tokenizer(formatted_oracle, return_tensors="pt").to(device)
    
    # Find positions of special tokens " ?" in the oracle prompt
    special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
    oracle_token_ids = oracle_inputs["input_ids"][0].tolist()
    steering_positions = [i for i, tid in enumerate(oracle_token_ids) if tid == special_token_id]
    
    print(f"   ✓ Oracle prompt built: {len(oracle_token_ids)} tokens")
    print(f"   Steering positions: {steering_positions}")
    
    # --- Step 5: Forward pass with gradients ---
    print("\n[5/6] Running forward pass with gradient-enabled steering...")
    
    # Get the perturbed activations
    perturbed_acts = perturbation()  # [num_positions, d_model]
    
    # Get injection layer (layer 1)
    injection_layer = 1
    injection_submodule = get_hf_submodule(model, injection_layer)
    
    # Forward pass
    logits = oracle_forward_with_gradients(
        model=model,
        input_ids=oracle_inputs["input_ids"],
        attention_mask=oracle_inputs["attention_mask"],
        steering_vectors=[perturbed_acts],
        positions=[steering_positions],
        injection_submodule=injection_submodule,
        steering_coefficient=1.0,
        device=device,
        dtype=dtype,
    )
    
    print(f"   ✓ Forward pass complete: logits shape {logits.shape}")
    
    # --- Step 6: Compute loss and backprop ---
    print("\n[6/6] Computing loss and backpropagating...")
    
    # Simple loss: just sum of logits (to test gradient flow)
    # In practice, you'd use a proper loss like cross-entropy
    loss = logits[:, -1, :].sum()  # Sum over last position's vocab logits
    
    print(f"   Loss value: {loss.item():.4f}")
    
    # Backprop
    loss.backward()
    
    # Check gradients
    if perturbation.delta.grad is not None:
        grad_norm = perturbation.delta.grad.norm().item()
        grad_nonzero = (perturbation.delta.grad.abs() > 1e-10).sum().item()
        total_params = perturbation.delta.numel()
        
        print(f"\n   ✓ GRADIENT FLOW VERIFIED!")
        print(f"   Gradient norm: {grad_norm:.6f}")
        print(f"   Non-zero gradients: {grad_nonzero}/{total_params} ({100*grad_nonzero/total_params:.1f}%)")
        
        # Quick sanity check: do one optimization step
        print("\n   Testing optimization step...")
        optimizer = torch.optim.Adam([perturbation.delta], lr=0.01)
        optimizer.step()
        
        new_norm = perturbation.perturbation_norm.item()
        print(f"   Perturbation norm after step: {new_norm:.6f}")
        
        success = True
    else:
        print("\n   ✗ GRADIENT FLOW FAILED - no gradients on perturbation!")
        success = False
    
    # --- Summary ---
    print("\n" + "=" * 70)
    if success:
        print("TEST PASSED - Gradients flow through the oracle!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Define a proper loss function (e.g., push away from 'smile')")
        print("  2. Run optimization loop to find effective perturbation")
        print("  3. Apply perturbation to original model and test causal effect")
    else:
        print("TEST FAILED - Check gradient blocking points")
        print("=" * 70)
    
    return success


if __name__ == "__main__":
    test_gradient_flow()

