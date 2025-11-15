"""Fine-Tuned In-Context Learning (FICL)."""

from dataclasses import dataclass

import torch
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from data import format_prompt, load_bbh


@dataclass
class Config:
    """Configuration for the Qwen model and training."""

    model_name: str = "Qwen/Qwen3-1.7B"
    dataset: str = "geometric_shapes"
    num_train_examples: int | None = None
    num_test_examples: int = 100
    num_context: int = 3
    num_epochs: int = 1
    learning_rate: float = 1e-5


if __name__ == "__main__":
    parser = ArgumentParser(description="Finetune Qwen model")
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()

    config = args.config

    wandb.init(project="ficl", config=config)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load data
    train_df, test_df = load_bbh(args.config.dataset, config.num_test_examples)

    if args.config.num_train_examples is not None:
        train_df = train_df[: config.num_train_examples]

    for pos, example in enumerate(train_df.itertuples()):
        input = example.input
        target = example.target

        m

        prompt = format_prompt(
            instruction="", context=train_df[:pos], input=input, target=target
        )

    # Actual training loop


# # Prepare input text
# text = "The capital of France is"
# inputs = tokenizer(text, return_tensors="pt")
# input_ids = inputs["input_ids"].to(model.device)
# attention_mask = inputs.get("attention_mask", None)
# if attention_mask is not None:
#     attention_mask = attention_mask.to(model.device)

# # OPTION 1: Explicit forward() call (old-school PyTorch style)
# print("=" * 60)
# print("OPTION 1: Explicit forward() call")
# print("=" * 60)

# model.eval()  # Set to eval mode
# with torch.no_grad():
#     # Direct forward pass - returns a ModelOutput object
#     outputs = model.forward(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#     )

# # Access the logits (shape: [batch_size, seq_len, vocab_size])
# logits = outputs.logits
# print(f"Logits shape: {logits.shape}")

# # Get predictions for the last token position
# last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
# predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
# predicted_token = tokenizer.decode([predicted_token_id])
# print(f"Predicted next token: '{predicted_token}' (ID: {predicted_token_id})")

# # Get top-5 predictions
# top_5_probs, top_5_indices = torch.topk(last_token_logits, k=5)
# print("\nTop 5 predicted tokens:")
# for prob, idx in zip(top_5_probs, top_5_indices):
#     token = tokenizer.decode([idx.item()])
#     print(f"  {token}: {prob.item():.4f}")

# # OPTION 2: Using model() directly (also PyTorch style, but more convenient)
# print("\n" + "=" * 60)
# print("OPTION 2: Using model() directly (calls forward internally)")
# print("=" * 60)

# with torch.no_grad():
#     # This also calls forward() under the hood
#     outputs = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#     )

# # Same as before
# logits = outputs.logits
# last_token_logits = logits[0, -1, :]
# predicted_token_id = torch.argmax(last_token_logits, dim=-1).item()
# predicted_token = tokenizer.decode([predicted_token_id])
# print(f"Predicted next token: '{predicted_token}' (ID: {predicted_token_id})")

# # OPTION 3: Accessing hidden states and other outputs
# print("\n" + "=" * 60)
# print("OPTION 3: Accessing hidden states")
# print("=" * 60)

# with torch.no_grad():
#     outputs = model.forward(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         output_hidden_states=True,  # Get all hidden states
#         output_attentions=True,  # Get attention weights
#     )

# # Access all layers' hidden states
# # hidden_states is a tuple, one per layer + embeddings
# if hasattr(outputs, "hidden_states") and outputs.hidden_states:
#     print(f"Number of hidden state layers: {len(outputs.hidden_states)}")
#     print(f"Hidden state shape (last layer): {outputs.hidden_states[-1].shape}")

# # Access attention weights
# if hasattr(outputs, "attentions") and outputs.attentions:
#     print(f"Number of attention layers: {len(outputs.attentions)}")
#     print(f"Attention shape (last layer): {outputs.attentions[-1].shape}")

# # OPTION 4: Training mode with gradients (for training/fine-tuning)
# print("\n" + "=" * 60)
# print("OPTION 4: Forward pass in training mode (with gradients)")
# print("=" * 60)

# model.train()  # Set to training mode

# # Example: compute loss for language modeling
# # Create labels (shifted input_ids)
# labels = input_ids.clone()

# # Forward pass with labels (computes loss internally)
# outputs = model.forward(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     labels=labels,  # When labels are provided, loss is computed
# )

# loss = outputs.loss
# print(f"Loss: {loss.item():.4f}")

# # Backward pass (if you want to do manual optimization)
# # loss.backward()
# # optimizer.step()
# # optimizer.zero_grad()

# print("\n" + "=" * 60)
# print("Summary:")
# print("=" * 60)
# print("✓ You can call model.forward() directly - it's just a PyTorch nn.Module")
# print("✓ Pass input_ids (required) and optionally attention_mask")
# print("✓ Returns ModelOutput with logits, hidden_states, attentions, etc.")
# print("✓ Use model.eval() and torch.no_grad() for inference")
# print("✓ Use model.train() for training (gradients enabled)")
# print("✓ Can pass labels to compute loss automatically")
