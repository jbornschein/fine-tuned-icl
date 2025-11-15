"""Evaluation for FICL."""

import dataclasses

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt, match_completion


@dataclasses.dataclass
class EvalResult:
    """Result of evaluating an example."""

    input: str
    target: str
    correct: bool
    completion: str


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
) -> EvalResult:
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

    return EvalResult(
        input=test_input,
        target=test_target,
        correct=correct,
        completion=completion,
    )


def eval_dataframe(
    instruction: str,
    context: pd.DataFrame,
    test_examples: pd.DataFrame,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> pd.DataFrame:
    results = [
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
    return pd.DataFrame([dataclasses.asdict(result) for result in results])
