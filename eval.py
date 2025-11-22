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


# Generator classes for text generation


class BaseGenerator:
    """Base class for text generation."""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """Generate text completion from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text completion
        """
        raise NotImplementedError


class LocalGenerator(BaseGenerator):
    """Generator for local HuggingFace models."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """Initialize the local generator.

        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """Generate text completion from a prompt using local model.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text completion
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                **kwargs,
            )

        completion_ids = generated_ids[0][model_inputs.input_ids.shape[1] :]
        completion = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()

        return completion

    @property
    def device(self):
        """Return the device the model is on."""
        return self.model.device


class APIGenerator(BaseGenerator):
    """Generator for API-based models using a direct OpenAI-compatible API client."""

    def __init__(self, openai_client, model: str):
        """
        Initialize the API generator.

        Args:
            openai_client: OpenAI client instance
            model: Name of the model to use for generation
        """
        self.openai_client = openai_client
        self.model = model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """
        Generate text completion from a prompt using an OpenAI-compatible API client.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters (passed to the API client)

        Returns:
            Generated text completion
        """

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"{prompt}\n</nothink>"}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        if hasattr(response, "choices") and response.choices:
            completion = response.choices[0].message.content
            completion = completion.strip() if completion else ""
            completion = completion.lstrip("</think>").strip()
            return completion
        return ""


# Evaluation functions


def eval_example(
    instruction: str,
    context: pd.DataFrame,
    test_input: str,
    test_target: str,
    *,
    generator: BaseGenerator,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> EvalResult:
    """Assemble a k-shot prompt from the context and sample a response from the test example.

    Args:
        instruction: Instruction text
        context: Context examples DataFrame
        test_input: Test input text
        test_target: Expected target text
        generator: Generator instance (LocalGenerator or APIGenerator)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature

    Returns:
        EvalResult with the evaluation outcome
    """
    prompt = format_prompt(instruction, context, test_input)
    completion = generator.generate(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )

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
    generator: BaseGenerator,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate a dataframe of test examples.

    Args:
        instruction: Instruction text
        test_examples: DataFrame of test examples with 'input' and 'target' columns
        context: DataFrame of context examples
        num_context_examples: Number of context examples to use (None = use all)
        generator: Generator instance (LocalGenerator or APIGenerator)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        random_state: Random state for deterministic sampling (None = non-deterministic)

    Returns:
        DataFrame with evaluation results
    """
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
        return context.sample(n=num_context_examples, random_state=random_state)

    results = [
        eval_example(
            instruction=instruction,
            context=sample_context(),
            generator=generator,
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
