import dataclasses
import logging

import structlog
from openai import OpenAI
from simple_parsing import ArgumentParser

import wandb
from data import load_bbh
from eval import APIGenerator, eval_dataframe


@dataclasses.dataclass
class Config:
    """Configuration for the Qwen model and training."""

    api_url: str = "http://strx:8080/v1"
    api_key: str | None = None
    model_name: str = "qwen3-1.7b"
    # For tokenizer only (not needed for API, but kept for prompt formatting compatibility)
    # tokenizer_name: str | None = None
    dataset: str = "geometric_shapes"
    instruction: str = ""
    random_seed: int = 42
    num_train_examples: int | None = None
    num_test_examples: int = 100
    test_pos: str = "3,10,30,100"
    max_sample_tokens: int = 8


if __name__ == "__main__":
    parser = ArgumentParser(description="ICL Evaluation")
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

    wandb.init(project="ficl", config=args.config, tags=["icl-only"])

    # Use wandb.config (has sweep values if in sweep, CLI values otherwise)
    config_dict = {
        field.name: getattr(wandb.config, field.name, getattr(args.config, field.name))
        for field in dataclasses.fields(Config)
    }
    config = Config(**config_dict)

    openai_client = OpenAI(
        base_url=config.api_url,
        api_key=config.api_key or "not-needed",
    )

    # Initialize API generator
    generator = APIGenerator(openai_client, model=config.model_name)

    # Load data
    train_df, test_df = load_bbh(config.dataset, config.num_test_examples)

    # Shuffle train_df based on random seed
    train_df = train_df.sample(frac=1, random_state=config.random_seed)

    if config.num_train_examples is not None:
        train_df = train_df[: config.num_train_examples]

    test_pos = [int(p) for p in config.test_pos.split(",") if p != ""]
    test_pos = [len(train_df) if p == -1 else p for p in test_pos]
    test_pos = [p for p in test_pos if p <= len(train_df)]

    for pos in test_pos:
        # For ICL-only, use ALL examples from train_df[:pos] (no subsampling)
        result = eval_dataframe(
            instruction=config.instruction,
            test_examples=test_df,
            context=train_df[:pos],
            num_context_examples=None,
            generator=generator,
            max_new_tokens=config.max_sample_tokens,
        )
        accuracy = result["correct"].mean()

        wandb.log(
            {
                "pos": pos,
                "num_context": pos,
                "test_accuracy": accuracy,
                "table": wandb.Table(dataframe=result),
            }
        )
