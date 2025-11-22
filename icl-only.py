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
    model: str = "qwen3-1.7b"
    # For tokenizer only (not needed for API, but kept for prompt formatting compatibility)
    # tokenizer_name: str | None = None
    dataset: str = "geometric_shapes"
    num_test_examples: int = 100


if __name__ == "__main__":
    parser = ArgumentParser(description="Finetune Qwen model")
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
    generator = APIGenerator(openai_client, model=config.model)

    # Load data
    train_df, test_df = load_bbh(args.config.dataset, args.config.num_test_examples)

    # for trial in range(3):
    for pos in [3, 5, 10, 30]:
        num_context_examples = pos
        result = eval_dataframe(
            instruction="",
            test_examples=test_df,
            context=train_df[:pos],
            num_context_examples=num_context_examples,
            generator=generator,
            max_new_tokens=8,
        )
        accuracy = result["correct"].mean()

        wandb.log(
            {
                "pos": pos,
                "accuracy": accuracy,
                "num_context": num_context_examples,
                "table": wandb.Table(dataframe=result),
            }
        )
