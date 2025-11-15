"""Fine-Tuned In-Context Learning (FICL)."""

from dataclasses import dataclass

import numpy as np
import torch
from simple_parsing import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from data import format_prompt, load_bbh
from eval import eval_example


@dataclass
class Config:
    """Configuration for the Qwen model and training."""

    model_name: str = "Qwen/Qwen3-1.7B"
    dataset: str = "geometric_shapes"
    instruction: str = ""
    random_seed: int = 42
    num_train_examples: int | None = None
    num_test_examples: int = 100
    num_context: int = 3
    num_epochs: int = 3
    learning_rate: float = 3e-5
    max_sample_tokens: int = 8


if __name__ == "__main__":
    parser = ArgumentParser(description="Finetune Qwen model")
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()

    config = args.config

    wandb.init(project="ficl", config=config)

    rng = np.random.default_rng(config.random_seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Load data
    train_df, test_df = load_bbh(args.config.dataset, config.num_test_examples)

    if args.config.num_train_examples is not None:
        train_df = train_df[: config.num_train_examples]

    cum_correct = 0
    avg_accuracy = 0.0

    pbar = tqdm(train_df.itertuples(), total=len(train_df), desc="Training")
    for pos, example in enumerate(pbar):
        input = example.input
        target = example.target

        #
        num_context = min(pos, config.num_context)

        # Evaluate model against new datapoint
        context = train_df[:pos].sample(n=num_context, random_state=rng)

        result = eval_example(
            instruction=config.instruction,
            context=context,
            test_input=input,
            test_target=target,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_sample_tokens,
        )

        cum_correct += int(result.correct)
        avg_accuracy = cum_correct / (pos + 1)

        wandb.log(
            {
                "pos": pos,
                "correct": result.correct,
                "cum_correct": cum_correct,
                "avg_accuracy": avg_accuracy,
                "input": result.input,
                "target": result.target,
            }
        )

        model.train()
        for e in range(config.num_epochs):
            context = train_df[:pos].sample(n=num_context, random_state=rng)
            prompt = format_prompt(config.instruction, context, input, target)

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Zero gradients before forward pass
            optimizer.zero_grad()

            outputs = model.forward(
                input_ids=model_inputs.input_ids,
                labels=model_inputs.input_ids,  # When labels are provided, loss is computed
            )

            # Backward pass and gradient step
            outputs.loss.backward()
            optimizer.step()

        model.eval()

        # Update progress bar with statistics
        pbar.set_postfix({"acc": f"{avg_accuracy:.2%}"})
