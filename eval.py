"""Evaluation for FICL."""

import dataclasses

import pandas as pd
import structlog
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import format_prompt, match_completion

logger = structlog.get_logger()


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

    logger.debug(
        f"Example {'correct' if correct else 'incorrect'}",
        prediction=completion,
        target=test_target,
        correct=correct,
    )

    return EvalResult(
        input=test_input,
        target=test_target,
        correct=correct,
        completion=completion,
    )


def eval_dataframe(
    instruction: str,
    test_examples: pd.DataFrame,
    context: pd.DataFrame,
    num_context_examples: int | None,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> pd.DataFrame:
    logger.info(
        "Starting evaluation",
        num_examples=len(test_examples),
        num_context_examples=num_context_examples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    def sample_context():
        if num_context_examples is None:
            return context
        return context.sample(n=num_context_examples)

    results = [
        eval_example(
            instruction=instruction,
            context=sample_context(),
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            test_input=test_example["input"],
            test_target=test_example["target"],
        )
        for _, test_example in tqdm(
            test_examples.iterrows(), total=len(test_examples), desc="Evaluating"
        )
    ]

    df_results = pd.DataFrame([dataclasses.asdict(result) for result in results])
    accuracy = df_results["correct"].mean()

    logger.info(
        "Evaluation complete",
        accuracy=float(accuracy),
        num_correct=int(df_results["correct"].sum()),
        num_total=len(df_results),
    )

    return df_results
