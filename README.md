# A Study in Silicon

Fine-tuning large language models to generate Victorian-era literary text using parameter-efficient techniques.

## Overview

This project explores domain adaptation of transformer-based language models through fine-tuning on classical literature. Using Google's Gemma architecture as a foundation, the model learns stylistic patterns from 12 complete Sherlock Holmes works (~140 chapters) to generate contextually appropriate Victorian prose.

**Key Technical Implementations:**
- Parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation)
- 8-bit quantization for memory-constrained training
- Custom preprocessing pipeline for handling historical text formatting
- Automated chapter segmentation with pattern matching fallbacks

## Dataset

**Source Material:** Complete Arthur Conan Doyle Sherlock Holmes corpus
- 12 full-length novels and story collections
- ~1,965 training examples after chunking
- 90/10 train-validation split

**Preprocessing Pipeline:**
- Automated removal of Project Gutenberg metadata
- Multi-pattern chapter detection (handles varying formats across publications)
- Sliding window tokenization (512 tokens, 64-token overlap)
- Maintains narrative context across chunks

## Architecture

**Base Model:** Gemma-3-27B-IT
- 27B parameters (trainable: ~2.8M via LoRA)
- Transformer architecture optimized for causal language modeling
- 8-bit quantization reduces memory footprint by ~75%

**Training Configuration:**
- LoRA rank: 16, alpha: 32
- Target modules: Q/K/V/O projection layers
- Batch size: 4 (effective: 16 with gradient accumulation)
- Learning rate: 2e-4 with warmup
- Mixed precision (FP16) training

## Results

The fine-tuned model demonstrates strong stylistic adaptation:
- Maintains Victorian vocabulary and sentence structure
- Generates contextually coherent mystery narratives
- Preserves character voice patterns (Holmes, Watson)
- Adapts to period-appropriate dialogue formatting

## Technical Stack

```
transformers>=4.36.0  # Model architecture & training
peft>=0.7.0           # LoRA implementation
bitsandbytes>=0.41.0  # 8-bit quantization
accelerate>=0.25.0    # Distributed training utilities
```

## Usage

**Data Preparation:**
```bash
python prepare_dataset.py
```
Processes all .txt files in `dataset/`, outputs to `processed_data/`

**Training:**
```bash
python finetune_gemma.py
```
Requires HuggingFace authentication and Gemma license acceptance

**Inference:**
```bash
python generate_sherlock.py
```

## Configuration

Training hyperparameters in `finetune_gemma.py`:
```python
MODEL_NAME = "google/gemma-3-27b-it"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
```

LoRA configuration:
```python
r=16, lora_alpha=32
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Performance Considerations

**Memory Requirements:**
- Gemma-3-27B (8-bit): ~40-50GB VRAM
- Recommended: A100 (40GB) or H100 (80GB)
- Optimization: Gradient checkpointing available for larger models

**Training Time:**
- ~8-12 hours on A100 (Gemma-3-27B, 3 epochs)
- Scales linearly with dataset size

## Project Structure

```
├── dataset/               # Source texts (12 books)
├── processed_data/        # Tokenized train/val splits
├── prepare_dataset.py     # Data preprocessing pipeline
├── finetune_gemma.py      # LoRA fine-tuning implementation
├── generate_sherlock.py   # Inference script
└── requirements.txt       # Python dependencies
```
