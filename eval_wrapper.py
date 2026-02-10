"""
Minimal wrapper for loading checkpoints and running inference.

Usage:
    from eval_wrapper import load_for_eval
    model, tokenizer = load_for_eval("logs/{run_id}/state_step001555.pt")

    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
    logits = model(input_ids)  # [B, T] -> [B, T, vocab_size]
"""
import os

# Set up single-GPU environment before importing train_gpt
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
from dataclasses import dataclass

# Import from train_gpt (now safe since training code is in __main__ block)
from train_gpt import GPT, ForwardScheduleConfig, get_bigram_hash, ParamConfig

# Make ParamConfig available in __main__ for torch.load unpickling
import sys
sys.modules['__main__'].ParamConfig = ParamConfig


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

        # Pad to multiple of 16 (required by attention reshaping)
        pad_len = (16 - T % 16) % 16
        T_padded = T + pad_len

        all_logits = []
        for b in range(B):
            seq = input_ids[b]  # [T]

            # Pad sequence if needed
            if pad_len > 0:
                seq = torch.cat([seq, seq[-1:].expand(pad_len)])

            # Compute bigram hash
            bigram_seq = get_bigram_hash(seq)

            # Dummy target (we ignore the loss)
            dummy_target = seq.to(torch.int64)

            # seqlens for single sequence - cumulative format [0, T] for flash_attn_varlen
            seqlens = torch.tensor([0, T_padded], device=device, dtype=torch.int32)

            # Forward pass - returns (loss, logits)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, logits = self.model(
                    seq.to(torch.int32),
                    dummy_target,
                    seqlens,
                    bigram_seq.to(device),
                    self.schedule_cfg
                )

            # Remove padding from logits - model returns (1, T_padded, vocab)
            logits = logits[0, :T, :]  # -> (T, vocab)
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

    # Convert to bfloat16 to match training
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
    model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
    model.attn_bank.data = model.attn_bank.data.bfloat16()
    model.mlp_bank.data = model.mlp_bank.data.bfloat16()

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

    # Test inference - use longer text to avoid padding issues
    text = "The quick brown fox jumps over the lazy dog "*5 + "The quick brown fox jumps over the lazy "  # ~90 tokens
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    print(f"Input tokens: {input_ids.shape[1]}")

    logits = model(input_ids)
    print(f"Input: {text}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Check top-k predictions
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    top_k = torch.topk(probs, k=10)
    print("\nTop 10 predictions:")
    for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
        print(f"  {i+1}. '{tokenizer.decode([idx])}' ({prob:.4f})")

    # Get next token prediction
    next_token_id = logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\nNext token prediction: '{next_token}'")
