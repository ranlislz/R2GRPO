# Scientific Entity and Relation Extraction (SciER)

This repository contains code for training and evaluating models for scientific entity recognition and relation extraction tasks.

## Overview

The project implements a two-stage training approach:
1. mimic-Supervised Fine-tuning (SFT)
2. R2GRPO optimization

## Requirements

See `requirements.txt` for complete dependencies. Key requirements:
- Python 3.8+
- PyTorch 2.5.1
- Transformers 4.49.0
- Unsloth 2025.3.14
- VLLM 0.7.2

## Training Pipeline

### Stage 1: mimic-Supervised Fine-tuning

The SFT stage uses `newreasonallsft.py` to perform instruction tuning with multiple tasks:

- Named Entity Recognition (NER)
- Relation Extraction (RE) 
- End-to-end Entity and Relation Extraction

To run SFT training:
```bash
python mimicsft.py --model_name "Qwen/Qwen2.5-7B-Instruct" --tasks ner re re_plus --train_epochs 3 --cuda_device 4
python mimicsft.py --model_name "Qwen/Qwen2.5-7B-Instruct" --tasks end2end --train_epochs 3 --cuda_device 4
```

### Stage 2: R^2GRPO

To run R^2GRPO training:
```bash
python r2grpo.py --model_path [sft_model_path] --cuda_device 0
```

### Evaluation
```bash
python evaluate.py --model_path [model_path] --cuda_device 0
```

### Data format
Training data should follow this JSON format:
{
    "sentence": "text...",
    "ner": [["entity1", "type1"], ["entity2", "type2"]],
    "rel": [["subject", "relation", "object"]]
}


Our code is based on the [unsloth](https://github.com/unslothai/unsloth) package.

The original dataset can be found in [SciER](https://github.com/edzq/SciER)
