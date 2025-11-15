"""Finetune Qwen LLM with Hugging Face Transformers using full fine-tuning.
"""

from dataclasses import dataclass
from simple_parsing import ArgumentParser


import wandb
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)


@dataclass
class Config:
    """Configuration for the Qwen model and training."""
    model_name: str = "Qwen/Qwen3-1.7B"    
    # model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Start with small model for testing
    output_dir: str = "./qwen-finetuned"
    num_epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-5


def load_model_and_tokenizer(config: Config, device_map: str = "auto"):
    """Load Qwen model and tokenizer for full fine-tuning."""
    print(f"Loading model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    
    # Print trainable parameters for full fine-tuning
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


def prepare_dataset(
    tokenizer,
    dataset_path: str = "wikitext",
    dataset_name: str = "wikitext-2-raw-v1",
    split: str = "train[:1%]",
):
    """Prepare dataset for training."""
    print(f"Loading dataset: {dataset_path}/{dataset_name} split={split}")
    
    dataset = load_dataset(dataset_path, dataset_name, split=split)  # Use 1% for demo
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset


def main():
    parser = ArgumentParser(description="Finetune Qwen model")
    parser.add_arguments(Config, dest="config")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.config)
    
    # Prepare dataset
    train_dataset = prepare_dataset(tokenizer)
    eval_dataset = prepare_dataset(tokenizer, split="test[:1%]")

    
    # Setup training arguments
    training_args = TrainingArguments(
        do_eval=True,
        output_dir=args.config.output_dir,
        num_train_epochs=args.config.num_epochs,
        per_device_train_batch_size=args.config.batch_size,
        learning_rate=args.config.learning_rate,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_steps=500,
        save_total_limit=2,
        push_to_hub=False,
        # report_to="none",
        # fp16=True,
        # evaluation_strategy="no",
        # gradient_accumulation_steps=4,
        # warmup_steps=100,
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.config.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()