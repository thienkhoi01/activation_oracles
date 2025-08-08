# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import contextlib
from typing import Callable, List, Dict, Tuple, Optional, Any
from jaxtyping import Float
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
import pickle
from dataclasses import dataclass, field, asdict
import einops
from rapidfuzz.distance import Levenshtein as lev
from tqdm import tqdm
import wandb
from torch.nn.utils import clip_grad_norm_
from peft import LoraConfig, get_peft_model
import json
import gc

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils
import interp_tools.introspect_utils as introspect_utils
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info
import self_training


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def collect_few_shot_examples(results: dict, max_examples: int = 5) -> list[dict]:
    examples = []

    for i, feature_result in enumerate(results["results"]):
        all_sentence_data = feature_result["sentence_data"]
        all_sentence_metrics = feature_result["sentence_metrics"]
        for i in range(len(all_sentence_data)):
            sentence_data = all_sentence_data[i]
            sentence_metrics = all_sentence_metrics[i]

            if sentence_metrics["original_max_activation"] < 1.0:
                continue

            max_act_ratio = (
                sentence_metrics["rewritten_max_activation"]
                / (sentence_metrics["original_max_activation"])
            )

            if max_act_ratio < 0.1 and sentence_metrics["sentence_distance"] < 0.1:
                example = {
                    "original_sentence": sentence_data["original_sentence"],
                    "rewritten_sentence": sentence_data["rewritten_sentence"],
                    "explanation": feature_result["explanation"],
                    "feature_idx": feature_result["feature_idx"],
                }

                examples.append(example)

                break

        if len(examples) >= max_examples:
            break

    return examples

"""Main script logic."""
cfg = SelfInterpTrainingConfig()

cfg.eval_set_size = 64
cfg.steering_coefficient = 2.0
cfg.train_batch_size = 4
cfg.eval_batch_size = 32
cfg.training_data_filename = (
    "contrastive_rewriting_results_google_gemma-2-9b-it_num_features_200000.pkl"
)
verbose = True
cfg.use_decoder_vectors = False
cfg.prefill_original_sentences = True

use_few_shot = False

if use_few_shot:
    cfg.use_lora = False
else:
    cfg.use_lora = True


torch.set_grad_enabled(False)

# %%
print(asdict(cfg))
dtype = torch.bfloat16
device = torch.device("cuda")

# %%

api_data_filename = cfg.training_data_filename

with open(api_data_filename, "rb") as f:
    api_data = pickle.load(f)

# %%

cfg.sae_width = api_data["config"]["sae_width"]
cfg.sae_layer = api_data["config"]["sae_layer"]
cfg.sae_filename = api_data["config"]["sae_filename"]
cfg.sae_repo_id = api_data["config"]["sae_repo_id"]

results_filename = "lora_eval_results.pkl"

model = introspect_utils.load_model(cfg, device, dtype, use_lora=cfg.use_lora)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
sae = introspect_utils.load_sae(cfg, device, dtype)
submodule = model_utils.get_submodule(model, cfg.sae_layer, cfg.use_lora)

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# %%

print(api_data["results"][0].keys())
print(api_data["results"][0]["sentence_data"][0].keys())

# %%

if use_few_shot:
    few_shot_examples = collect_few_shot_examples(api_data, max_examples=5)
    for example in few_shot_examples:
        print(example["feature_idx"])
        print(f"\nOriginal sentence: {example['original_sentence']}")
        print(f"\nRewritten sentence: {example['rewritten_sentence']}")
        print(f"\nExplanation: {example['explanation']}")
        print("-" * 100)
else:
    few_shot_examples = []

# %%

if cfg.use_lora:
    lora_path = "checkpoints_encoder_gemma/step_94000"
    # lora_path = "checkpoints_decoder_gemma/step_84000"
    run_name = lora_path
    model.load_adapter(
        lora_path,
        adapter_name=run_name,
        is_trainable=False,
        low_cpu_mem_usage=True,  # 4x speedup from this
    )
    model.set_adapter(run_name)

# %%
max_activation = 0
for feature_result in api_data["results"]:
    for sentence_metrics in feature_result["sentence_metrics"]:
        max_activation = max(
            max_activation, sentence_metrics["original_max_activation"]
        )

print(f"Max activation: {max_activation}")

cfg.max_activation_required = cfg.max_activation_percentage_required * max_activation

training_examples = self_training.collect_training_examples(
    api_data,
    cfg.max_acts_ratio_threshold,
    cfg.max_distance_threshold,
    cfg.max_activation_required,
)

# %%

train_features = set()

for example in training_examples:
    train_features.add(example["feature_idx"])

test_features = set()

for example in api_data["results"]:
    if use_few_shot and example["feature_idx"] in few_shot_examples:
        continue

    if example["feature_idx"] not in train_features:
        test_features.add(example["feature_idx"])

print(f"train examples: {len(training_examples)}")
print(f"Train features: {len(train_features)}, Test features: {len(test_features)}")

cfg.eval_features = list(test_features)[: cfg.eval_set_size]

assert len(cfg.eval_features) == cfg.eval_set_size

train_eval_prompt = self_training.build_training_prompt()

print(train_eval_prompt)
# %%

eval_data = self_training.construct_eval_dataset(
    cfg,
    cfg.eval_set_size,
    train_eval_prompt,
    cfg.eval_features,
    api_data,
    sae,
    tokenizer,
    few_shot_examples,
    prefill_original_sentences=cfg.prefill_original_sentences,
)


# %%

temp_e_batch = eval_data[:10]

temp_e_batch = self_training.construct_batch(temp_e_batch, tokenizer, device)

print(tokenizer.decode(temp_e_batch["input_ids"][0], skip_special_tokens=False))

# %%

print(temp_e_batch.keys())
print(temp_e_batch["steering_vectors"][0][0].shape)
print(len(temp_e_batch["steering_vectors"]))
print(len(temp_e_batch["steering_vectors"][0]))
print(temp_e_batch["feature_indices"][0])
print(temp_e_batch["feature_indices"][1])

# %%

print((temp_e_batch["steering_vectors"][0][0] - temp_e_batch["steering_vectors"][1][0]).sum())
print((temp_e_batch["steering_vectors"][0][0] - temp_e_batch["steering_vectors"][0][1]).sum())

# %%

temp_eval_data = eval_data[:]

print(f"eval data length: {len(eval_data)}")

model.eval()

cfg.generation_kwargs["do_sample"] = True
cfg.generation_kwargs["temperature"] = 1.0

all_sentence_metrics = []
all_sentence_data = []
all_feature_results = []
for i in tqdm(
    range(0, len(temp_eval_data), cfg.eval_batch_size),
    desc="Evaluating model",
):
    e_batch = temp_eval_data[i : i + cfg.eval_batch_size]
    e_batch = self_training.construct_batch(e_batch, tokenizer, device)

    feature_results = self_training.eval_features_batch(
        cfg=cfg,
        eval_batch=e_batch,
        model=model,
        submodule=submodule,
        sae=sae,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    all_feature_results.extend(feature_results)
    for res in feature_results:
        all_sentence_metrics.append(res["sentence_metrics"])
        all_sentence_data.append(res["sentence_data"])

if all_sentence_metrics:
    aggregated_metrics = {}
    metric_keys = all_sentence_metrics[0].keys()
    for key in metric_keys:
        avg_val = sum(m[key] for m in all_sentence_metrics) / len(all_sentence_metrics)
        aggregated_metrics[key] = avg_val

all_results = {
    "all_sentence_metrics": all_sentence_metrics,
    "all_sentence_data": all_sentence_data,
    "all_feature_results": all_feature_results,
    "aggregated_metrics": aggregated_metrics,
    "config": asdict(cfg),
}

with open(results_filename, "wb") as f:
    pickle.dump(all_results, f)

print(f"Results saved to {results_filename}")
print(aggregated_metrics)

# %%
