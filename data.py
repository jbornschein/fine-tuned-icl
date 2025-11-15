"""Data related utilities, including prompt formatting and completion matching."""

import re

import pandas as pd
from datasets import load_dataset


def load_bbh(name, num_test_examples: int = 100):
    """Load BigBenchHard dataset."""
    dataset = load_dataset("Fhrozen/big_bench_hard", name)
    df = dataset["train"].to_pandas()

    train_df = df.iloc[:-num_test_examples]
    test_df = df.iloc[-num_test_examples:]
    return train_df, test_df


def format_prompt(
    instruction: str,
    context: pd.DataFrame,
    input: str,
    target: str | None = None,
) -> str:
    prompt_segments = []
    if instruction:
        prompt_segments.append(instruction.strip())

    for _, example in context.iterrows():
        input_value = example["input"].strip()
        target_value = example["target"].strip()
        prompt_segments.append(
            f"== Next Example ==\nInput:\n{input_value}\nOutput:\n{target_value}"
        )

    prompt_segments.append(
        f"== Test Example ==\nInput:\n{input}\nOutput:{target if target is not None else ''}\n"
    )

    return "\n\n".join(segment for segment in prompt_segments if segment)


def match_completion(completion: str, desired: str) -> bool:
    """Match a completion against some desired output."""
    # Strip excess whitespace and newlines
    completion = completion.strip()
    desired = desired.strip()

    if completion.startswith(desired):
        return True

    patterns = [
        r"^(?P<answer>.+?)$",
        r"^Answer:\s*(?P<answer>.+?)$",
        r"^A:\s*(?P<answer>.+?)$",
        r"^\\box\{(?P<answer>.+?)\}$",
        r"^\\boxed\{(?P<answer>.+?)\}$",
    ]

    for pattern in patterns:
        match = re.search(pattern, completion, flags=re.DOTALL)
        if match and match.group("answer").strip() == desired:
            return True

    return False
