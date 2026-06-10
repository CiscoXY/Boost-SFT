# Boost-SFT

Code for the Boost-SFT paper.

## Pipeline

1. **SID construction** — Train RQ-VAE to generate semantic IDs for items (`sid/RQVAE/`)
2. **Data preparation** — Build SFT/DPO training data from user sequences and item metadata (`src/data/`)
3. **Training** — SFT with attenuation strategy, DPO, or standard SFT via LLaMA-Factory (`src/train/`)
4. **Inference** — Batch generation with vLLM (`src/inference/`)
5. **Evaluation** — Hit Rate / NDCG metrics on recommendation tasks (`src/eval/`)

## Directory Structure

```
Boost-SFT/
├── src/
│   ├── data/           # Data preprocessing: user sequences, SID alignment, JSONL merging
│   ├── train/          # Training scripts: SFT (attenuation/loss variants), DPO
│   ├── eval/           # Evaluation: Hit Rate, NDCG at item and SID levels
│   ├── inference/      # vLLM-based batch inference
│   ├── model_tools/    # Tokenizer utilities (add/test special tokens)
│   └── sample_data_format/  # Example data format for reference
├── sid/
│   ├── RQVAE/          # RQ-VAE model for semantic ID construction
│   └── sid_evaluation/ # SID-level case analysis
└── data/final_result/  # Experiment result CSVs
```

## Quick Start

See individual scripts for usage. Training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the backend.
