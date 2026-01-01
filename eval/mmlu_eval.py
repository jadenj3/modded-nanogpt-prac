"""MMLU evaluation harness adapter for modded-nanogpt.

This script wraps the custom GPT model defined in ``train_gpt.py`` and exposes
it through EleutherAI's lm-evaluation-harness interface so we can score tasks
such as MMLU without converting checkpoints to Hugging Face format.

Example
-------
python eval/mmlu_eval.py \
    --checkpoint logs/<run_id>/state_step000500.pt \
    --tasks mmlu \
    --num-fewshot 5 \
    --output results/mmlu.json

The script expects ``lm_eval`` (EleutherAI's harness) and ``tiktoken`` to be
installed in the current environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

# Set dummy env vars before importing train_gpt (which expects torchrun env)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Allow running the script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import tiktoken

from lm_eval import evaluator, tasks, utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import train_gpt
from train_gpt import BOS_ID, ForwardScheduleConfig, GPT


@register_model("nanogpt")
class NanoGPTLMEvalAdapter(LM):
    """Expose train_gpt.GPT checkpoints to lm-evaluation-harness."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_gen_toks: int = 128,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but no GPU is visible")
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype.lower(), torch.bfloat16)
        self._max_gen_toks = max_gen_toks
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.bos_token = BOS_ID
        self._max_length = (
            train_gpt.args.val_batch_size // (train_gpt.grad_accum_steps * train_gpt.world_size)
        )
        self._batch_size = 1

        self.model = GPT(
            vocab_size=50257,
            num_layers=11,
            num_heads=6,
            head_dim=128,
            model_dim=768,
            max_seq_len=self._max_length,
        ).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = state.get("model", state)
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

        # Load model weights (allow missing YaRN buffers since they're saved separately)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        expected_missing = {"yarn.cos", "yarn.sin"}
        actual_missing = set(missing) - expected_missing
        if actual_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing keys: {actual_missing}. Unexpected keys: {unexpected}."
            )

        # Load non-persistent state that was saved separately in checkpoint
        self.model.split_embed = state["split_embed"]
        self.model.yarn.cos = state["yarn_cos"].to(self.device)
        self.model.yarn.sin = state["yarn_sin"].to(self.device)
        self.model.yarn.angular_freq = state["yarn_angular_freq"].to(self.device)
        self.model.yarn.attn_scale = state["yarn_attn_scale"]

        for module in self.model.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                module.weight.data = module.weight.data.to(self.dtype)
        self.model.eval()

        mtp = torch.tensor([1.0], device=self.device, dtype=torch.float32)
        self.forward_cfg = ForwardScheduleConfig(
            mtp_weights=mtp,
            ws_short=train_gpt.args.ws_validate_post_yarn_ext // 2,
            ws_long=train_gpt.args.ws_validate_post_yarn_ext,
        )

    # ------------------------------------------------------------------
    # Required LM API
    # ------------------------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        return self.bos_token

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device_name(self) -> str:
        return str(self.device)

    @property
    def tokenizer_name(self) -> str:
        return "gpt2"

    # lm-eval uses these helpers internally for caching
    def tok_encode(self, string: str, **_: object) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: Iterable[int], **_: object) -> str:
        return self.tokenizer.decode(list(tokens))

    # ------------------------------------------------------------------
    # Harness entrypoints
    # ------------------------------------------------------------------
    def loglikelihood(self, requests):
        outputs: list[tuple[float, bool]] = []
        for req in requests:
            cache_val = self.cache_hook.get("loglikelihood", req.args)
            if cache_val is not None:
                outputs.append(cache_val)
                continue

            context, continuation = req.args
            context_tokens = self._encode_with_bos(context)
            continuation_tokens = self.tokenizer.encode(continuation)
            if not continuation_tokens:
                pair = (0.0, True)
                outputs.append(pair)
                self.cache_hook.add_partial("loglikelihood", req.args, pair)
                continue

            tokens = context_tokens + continuation_tokens
            logits = self._run_model(tokens, valid_tokens=len(tokens) - 1)
            start = max(len(context_tokens) - 1, 0)
            end = start + len(continuation_tokens)
            selected = logits[start:end].to(torch.float32)
            target = torch.tensor(continuation_tokens, device=self.device)
            log_probs = torch.log_softmax(selected, dim=-1)
            token_logprobs = log_probs.gather(1, target.unsqueeze(-1)).squeeze(-1)
            total = float(token_logprobs.sum().item())
            greedy = bool(selected.argmax(dim=-1).eq(target).all().item())
            pair = (total, greedy)
            outputs.append(pair)
            self.cache_hook.add_partial("loglikelihood", req.args, pair)
        return outputs

    def loglikelihood_rolling(self, requests):
        results: list[float] = []
        for req in requests:
            (text,) = req.args
            token_ids = self._encode_with_bos(text)
            total = 0.0
            start = 0
            while start + 1 < len(token_ids):
                end = min(len(token_ids), start + self._max_length)
                chunk = token_ids[start:end]
                logits = self._run_model(chunk, valid_tokens=len(chunk) - 1)
                target = torch.tensor(chunk[1:], device=self.device)
                log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
                total += float(log_probs.gather(1, target.unsqueeze(-1)).squeeze(-1).sum().item())
                start = end - 1  # overlap by one token to preserve context
            results.append(total)
        return results

    def generate_until(self, requests):
        generations: list[str] = []
        for req in requests:
            cache_val = self.cache_hook.get("generate_until", req.args)
            if cache_val is not None:
                generations.append(cache_val)
                continue

            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            if isinstance(until, str):
                until = [until]
            max_gen = gen_kwargs.get("max_gen_toks", self._max_gen_toks) if isinstance(gen_kwargs, dict) else self._max_gen_toks

            tokens = self._encode_with_bos(context)
            generated: list[int] = []
            for _ in range(max_gen):
                # include a dummy token so we receive the next-token logits
                logits = self._run_model(tokens + [self.bos_token], valid_tokens=len(tokens))
                next_token = int(logits[-1].argmax().item())
                tokens.append(next_token)
                generated.append(next_token)
                text = self.tokenizer.decode(generated)
                if self._stops_here(text, until):
                    text = self._truncate_until(text, until)
                    generated = self.tokenizer.encode(text)
                    break
                if len(tokens) >= self._max_length:
                    break

            text = self.tokenizer.decode(generated)
            generations.append(text)
            self.cache_hook.add_partial("generate_until", req.args, text)
        return generations

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _encode_with_bos(self, text: str) -> list[int]:
        """Encode text and prepend BOS token."""
        tokens = self.tokenizer.encode(text)
        return [self.bos_token] + tokens

    @torch.no_grad()
    def _run_model(self, tokens: list[int], valid_tokens: int) -> torch.Tensor:
        """Run forward pass and return logits for valid_tokens positions.

        Args:
            tokens: List of token IDs to feed the model
            valid_tokens: Number of token positions to return logits for

        Returns:
            Tensor of shape (valid_tokens, vocab_size) with logit scores
        """
        # Pad to multiple of 16 as required by the model
        pad_len = (16 - len(tokens) % 16) % 16
        padded = tokens + [self.bos_token] * pad_len

        input_ids = torch.tensor(padded, dtype=torch.int32, device=self.device)
        # Dummy targets (required by forward signature, not used for logits)
        target_ids = torch.tensor(padded, dtype=torch.int64, device=self.device)

        # Build sequence length tensor for flash attention: [0, seq_len]
        seqlens = torch.tensor([0, len(padded)], dtype=torch.int32, device=self.device)

        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            # Use the model's built-in return_logits=True to get logits
            _, logits = self.model(
                input_ids,
                target_ids,
                seqlens,
                self.forward_cfg,
                return_logits=True,
            )

        # Return only the valid token positions (exclude padding), remove batch dim
        return logits.squeeze(0)[:valid_tokens]

    def _stops_here(self, text: str, stop_sequences: list[str]) -> bool:
        """Check if any stop sequence appears in text."""
        return any(stop in text for stop in stop_sequences)

    def _truncate_until(self, text: str, stop_sequences: list[str]) -> str:
        """Truncate text at first occurrence of any stop sequence."""
        for stop in stop_sequences:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        return text


def main():
    parser = argparse.ArgumentParser(description="Evaluate modded-nanogpt on lm-eval tasks")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu",
        help="Comma-separated list of tasks (default: mmlu)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for inference (default: bfloat16)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit number of examples per task (for debugging)",
    )
    args = parser.parse_args()

    # Build the model wrapper
    model = NanoGPTLMEvalAdapter(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )

    # Parse tasks
    task_list = [t.strip() for t in args.tasks.split(",")]

    # Run evaluation
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
