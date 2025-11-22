"""Fine-Tuned In-Context Learning (FICL)."""

import dataclasses
import logging

import numpy as np
import structlog
import torch
from simple_parsing import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from data import format_prompt, load_bbh
from eval import LocalGenerator, eval_dataframe, eval_example

logger = structlog.get_logger()


@dataclasses.dataclass
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
    test_pos: str = "3,10,30,100,-1"
    optimizer: str = "adamw"


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tuned ICL")
    parser.add_arguments(Config, dest="config")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (default: INFO)",
    )
    args = parser.parse_args()

    # Configure structlog with stdlib logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))

    wandb.init(project="ficl", config=args.config, tags=["ficl-finetune"])

    # Use wandb.config (has sweep values if in sweep, CLI values otherwise)
    config_dict = {
        field.name: getattr(wandb.config, field.name, getattr(args.config, field.name))
        for field in dataclasses.fields(Config)
    }
    config = Config(**config_dict)

    logger.info("Starting FICL training", config=config)

    rng = np.random.default_rng(config.random_seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=torch.bfloat16, device_map="auto"
    )

    # Create generator for evaluation
    generator = LocalGenerator(model, tokenizer)

    # Create optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adafactor":
        optimizer = torch.optim.Adafactor(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Load data
    train_df, test_df = load_bbh(config.dataset, config.num_test_examples)

    if config.num_train_examples is not None:
        train_df = train_df[: config.num_train_examples]

    test_pos = [int(p) for p in config.test_pos.split(",") if p != ""]

    cum_correct = 0
    avg_accuracy = 0.0

    pbar = tqdm(train_df.itertuples(), total=len(train_df), desc="Training")
    for pos, example in enumerate(pbar):
        input = example.input
        target = example.target

        num_context = min(pos, config.num_context)

        # Evaluate model against new datapoint
        context = train_df[:pos].sample(n=num_context, random_state=rng)

        result = eval_example(
            instruction=config.instruction,
            context=context,
            test_input=input,
            test_target=target,
            generator=generator,
            max_new_tokens=config.max_sample_tokens,
        )

        cum_correct += int(result.correct)
        avg_accuracy = cum_correct / (pos + 1)

        diags = {
            "pos": pos,
            "correct": result.correct,
            "cum_correct": cum_correct,
            "avg_accuracy": avg_accuracy,
            "input": result.input,
            "target": result.target,
            "completion": result.completion,
        }

        # Do we perform a test-set evaluation?
        if pos in test_pos:
            test_result = eval_dataframe(
                instruction=config.instruction,
                context=train_df[:pos],
                num_context_examples=num_context,
                test_examples=test_df,
                generator=generator,
                max_new_tokens=config.max_sample_tokens,
                random_state=config.random_seed,
            )
            test_accuracy = float(test_result["correct"].mean())
            diags |= {
                "test_accuracy": test_accuracy,
            }

        # Gradient updates...

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

        # Log diagnostics and update progress bar
        wandb.log(diags)
        pbar.set_postfix({"# correct": cum_correct, "avg-acc": f"{avg_accuracy:.2}"})

    # Do we perform a final test-set evaluation?
    if -1 in test_pos:
        test_result = eval_dataframe(
            instruction=config.instruction,
            context=train_df,
            num_context_examples=num_context,
            test_examples=test_df,
            generator=generator,
            max_new_tokens=config.max_sample_tokens,
            random_state=config.random_seed,
        )
        test_accuracy = float(test_result["correct"].mean())
        diags = {
            "pos": len(train_df),
            "test_accuracy": test_accuracy,
        }
        wandb.log(diags)

    logger.info("Training complete", accuracy=avg_accuracy)
