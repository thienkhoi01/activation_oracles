# Activation Oracles

This repository contains the code for training and evaluating **Activation Oracles (AOs)** - models trained to answer natural language questions about LLM activations.

## Overview

Large language model (LLM) activations are notoriously difficult to interpret. Activation Oracles take a simpler approach: they are LLMs trained to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language.

Key findings from our research:
- AOs can recover information fine-tuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text
- AOs generalize to far out-of-distribution settings, despite never seeing activations from fine-tuned models during training
- Our best AOs match or exceed prior white-box baselines on all four downstream tasks we study

## Installation

```bash
uv sync
source .venv/bin/activate
huggingface-cli login --token <your_token>
```

## Quick Start: Demo

The easiest way to get started is with our demo notebook ([Colab](https://colab.research.google.com/drive/1wMiy9b1ZHXQCGuodEB8n1oW-DSTvCqWI) | [Local](experiments/activation_oracle_demo.ipynb)), which demonstrates:
- Extracting hidden information (secret words) from fine-tuned models
- Detecting model goals without observing responses
- Analyzing emotions and reasoning in model activations

The Colab version runs on a free T4 GPU and uses pre-trained oracle weights.

## Training

To train an Activation Oracle, use the training script with `torchrun`:

```bash
torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py
```

By default, this trains a full Activation Oracle on Qwen3-8B using a diverse mixture of training tasks:
- System prompt question-answering (LatentQA)
- Binary classification tasks
- Self-supervised context prediction

Training configuration can be modified in `nl_probes/configs/sft_config.py`.

## Reproducing Paper Experiments

To replicate the evaluation results from the paper, run:

```bash
bash experiments/paper_evals.sh
```

This runs evaluations on four downstream tasks:
- Classification evaluation
- Gender bias detection
- Taboo word extraction
- Sleeper agent detection (SSC)
- PersonaQA evaluation

## Pre-trained Models

Pre-trained oracle weights are available on Hugging Face: [Activation Oracles Collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{activation_oracles,
  title={Activation Oracles},
  author={...},
  year={2025}
}
```

## License

[Add license information]
