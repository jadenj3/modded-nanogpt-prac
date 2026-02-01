"""
Minimal wrapper for loading checkpoints and running inference.

Usage:
    from eval_wrapper import load_for_eval
    model, tokenizer = load_for_eval("logs/{run_id}/state_step001555.pt")

    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
    logits = model(input_ids)  # [B, T] -> [B, T, vocab_size]
"""
import torch
from dataclasses import dataclass

# Import from train_gpt (now safe since training code is in __main__ block)
from train_gpt import GPT, ForwardScheduleConfig, get_bigram_hash, args


@dataclass
class EvalConfig:
    ws_short: int = 6   # ws_final // 2
    ws_long: int = 13   # ws_final
    max_seq_len: int = 2048


class InferenceWrapper:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or EvalConfig()
        self.max_seq_len = self.config.max_seq_len

        # Fixed schedule config for inference (final training values)
        # mtp_weights is only used in training mode, so None is fine
        self.schedule_cfg = ForwardScheduleConfig(
            mtp_weights=None,
            ws_short=self.config.ws_short,
            ws_long=self.config.ws_long,
        )

    def __call__(self, input_ids):
        """
        input_ids: [B, T] token ids
        returns: [B, T, vocab_size] logits
        """
        B, T = input_ids.shape
        device = input_ids.device

        all_logits = []
        for b in range(B):
            seq = input_ids[b]  # [T]

            # Compute bigram hash
            bigram_seq = get_bigram_hash(seq)

            # Dummy target (we ignore the loss)
            dummy_target = seq.to(torch.int64)

            # seqlens for single sequence
            seqlens = torch.tensor([T], device=device, dtype=torch.int32)

            # Forward pass - returns (loss, logits)
            with torch.no_grad():
                _, logits = self.model(
                    seq.to(torch.int32),
                    dummy_target,
                    seqlens,
                    bigram_seq.to(device),
                    self.schedule_cfg
                )
            all_logits.append(logits)

        # Stack to [B, T, vocab_size]
        return torch.stack(all_logits, dim=0)


def load_for_eval(checkpoint_path, device="cuda"):
    """
    Load model from checkpoint for evaluation.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., "logs/{run_id}/state_step001555.pt")
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        model: InferenceWrapper with __call__(input_ids) -> logits interface
        tokenizer: GPT2Tokenizer
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle torch.compile prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}

    # Create model with hardcoded config (matches train_gpt.py)
    model = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=2048,
    ).to(device)

    model.load_state_dict(model_data)
    model.eval()

    # Wrap for inference interface
    wrapper = InferenceWrapper(model)

    # Return GPT-2 tokenizer (same vocab as training)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return wrapper, tokenizer


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eval_wrapper.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    model, tokenizer = load_for_eval(checkpoint_path)

    # Test inference
    text = "The quick brown fox"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    logits = model(input_ids)
    print(f"Input: {text}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Get next token prediction
    next_token_id = logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"Next token prediction: '{next_token}'")
