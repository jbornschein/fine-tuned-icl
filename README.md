# Qwen Finetuning with Hugging Face Transformers

A Python UV project for finetuning Qwen LLM models using Hugging Face Transformers with full fine-tuning (all parameters are trained).

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for Python package management. Make sure you have UV installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or install via pip:
```bash
pip install uv
```

Install dependencies:
```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

## Usage

### Basic Training

Run with default settings (full fine-tuning on Qwen2.5-0.5B-Instruct):
```bash
uv run python main.py
```

### Custom Model

Finetune a different Qwen model:
```bash
uv run python main.py --model-name Qwen/Qwen2.5-1.5B-Instruct
```

### Custom Training Parameters

Adjust training hyperparameters:
```bash
uv run python main.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --num-epochs 5 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --output-dir ./my-finetuned-model
```

## Command Line Arguments

- `--model-name`: Hugging Face model identifier (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `--output-dir`: Directory to save the finetuned model (default: `./qwen-finetuned`)
- `--num-epochs`: Number of training epochs (default: `3`)
- `--batch-size`: Training batch size per device (default: `4`)
- `--learning-rate`: Learning rate (default: `2e-4`)

## Features

- **Full Fine-Tuning**: All model parameters are trained (no parameter-efficient methods)
- **Flexible Configuration**: Command-line arguments for easy customization
- **Qwen Model Support**: Optimized for Qwen models from Alibaba

## Available Qwen Models

You can use any Qwen model from Hugging Face, such as:
- `Qwen/Qwen2.5-0.5B-Instruct` (smallest, good for testing)
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct` (requires more GPU memory)
- `Qwen/Qwen2.5-14B-Instruct` (requires significant GPU memory)
- `Qwen/Qwen2.5-32B-Instruct` (requires large GPU memory)

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended for training)
- UV package manager

## Notes

- The script uses a small subset of the dataset (1%) for demonstration. Modify the dataset loading in `main.py` to use more data.
- Full fine-tuning trains all model parameters, which requires more GPU memory and training time compared to parameter-efficient methods.
- Make sure you have sufficient GPU memory for the model size you choose.

