from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from data import format_prompt, load_bbh, match_completion


@dataclass
class Config:
    """Configuration for the Qwen model and training."""

    model_name: str = "Qwen/Qwen3-1.7B"
    dataset: str = "geometric_shapes"
    num_test_examples: int = 100


def eval_example(
    instruction: str,
    context: pd.DataFrame,
    test_input: str,
    test_target: str,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Assemble a k-shot prompt from the context and sample a response from the test example."""
    prompt = format_prompt(instruction, context, test_input)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )

    completion_ids = generated_ids[0][model_inputs.input_ids.shape[1] :]
    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()

    correct = match_completion(completion, test_target)
    if correct:
        print(f"CORRECT:   pred={completion}  vs.  true={test_target}")
    else:
        print(f"INCORRECT: pred={completion}  vs.  true={test_target}")

    return dict(
        input=test_input,
        target=test_target,
        correct=correct,
        completion=completion,
    )


def eval_examples(
    instruction: str,
    context: pd.DataFrame,
    test_examples: pd.DataFrame,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> pd.DataFrame:
    results = pd.DataFrame(
        [
            eval_example(
                instruction=instruction,
                context=context,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                test_input=test_example["input"],
                test_target=test_example["target"],
            )
            for _, test_example in test_examples.iterrows()
        ]
    )
    return results


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
            result = eval_examples(
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
