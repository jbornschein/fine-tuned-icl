from dataclasses import dataclass

import torch
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from data import load_bbh
from eval import eval_dataframe


@dataclass
class Config:
    """Configuration for the Qwen model and training."""

    model_name: str = "Qwen/Qwen3-1.7B"
    dataset: str = "geometric_shapes"
    num_test_examples: int = 100


if __name__ == "__main__":
    parser = ArgumentParser(description="Finetune Qwen model")
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()

    wandb.init(project="icl", config=args.config)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.config.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # Load data
    train_df, test_df = load_bbh(args.config.dataset, args.config.num_test_examples)

    for num_context in [3, 5, 10, 30]:
        for trial in range(3):
            context_examples = train_df.sample(n=num_context, random_state=trial)
            result = eval_dataframe(
                instruction="",
                context=context_examples,
                test_examples=test_df,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=8,
            )
            result.assign(num_context=num_context, trial=trial)

            accuracy = result.correct.mean()

            wandb.log(
                {
                    "accuracy": accuracy,
                    "num_context": num_context,
                    "trial": trial,
                    "table": wandb.Table(dataframe=result),
                }
            )
