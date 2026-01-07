"""
Quick verification script to test the Activation Oracles setup.
This loads the model, oracle, and runs a simple test.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

# Model configuration
model_name = "Qwen/Qwen3-8B"
oracle_lora_path = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"

device = torch.device("cuda")
dtype = torch.bfloat16
torch.set_grad_enabled(False)

print("=" * 60)
print("ACTIVATION ORACLES SETUP VERIFICATION")
print("=" * 60)

# Configure 8-bit quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

print(f"\n1. Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("   ✓ Tokenizer loaded")

print(f"\n2. Loading model: {model_name} with 8-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=dtype,
)
model.eval()
print("   ✓ Model loaded")

# Add dummy adapter for consistent PeftModel API
print("\n3. Adding dummy LoRA adapter...")
dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")
print("   ✓ Dummy adapter added")

# Load oracle adapter
print(f"\n4. Loading oracle adapter: {oracle_lora_path}")
model.load_adapter(oracle_lora_path, adapter_name=oracle_lora_path, is_trainable=False, low_cpu_mem_usage=True)
print("   ✓ Oracle adapter loaded")

# Verify model structure
print("\n5. Verifying model structure...")
print(f"   - Model type: {type(model)}")
print(f"   - Config: {model.config._name_or_path}")
print(f"   - Device: {next(model.parameters()).device}")
print(f"   - Number of layers: {model.config.num_hidden_layers}")

# Check layer access (this is how activations are extracted)
print("\n6. Testing layer access...")
layer_idx = 18  # Middle layer (50% of 36 layers)
layer = model.model.layers[layer_idx]
print(f"   - Layer {layer_idx} type: {type(layer).__name__}")

print("\n" + "=" * 60)
print("✓ ALL CHECKS PASSED - Setup is complete!")
print("=" * 60)

print("\n\nKEY CODEBASE COMPONENTS IDENTIFIED:")
print("-" * 60)
print("""
1. ACTIVATION EXTRACTION:
   - File: nl_probes/utils/activation_utils.py
   - Function: collect_activations_multiple_layers()
   - Hooks into: model.model.layers[layer] (for Qwen)
   - Output: dict[layer_idx -> tensor(B, L, D)]

2. ACTIVATION INJECTION (Steering):
   - File: nl_probes/utils/steering_hooks.py  
   - Function: get_hf_activation_steering_hook()
   - Injection layer: layer 1 (early in model)
   - Method: Add normalized vector * original_norm * coeff to residual

3. ORACLE RESPONSE GENERATION:
   - File: nl_probes/utils/eval.py
   - Function: run_evaluation()
   - Special tokens: " ?" used as placeholders for activation injection
   - Prompt prefix: "Layer: {layer}\\n" + " ?" * num_positions + " \\n"

4. LORA LOADING:
   - File: nl_probes/base_experiment.py
   - Uses PEFT library: model.load_adapter(), model.set_adapter()
   - Supports multiple adapters (oracle, target model)
""")

