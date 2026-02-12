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

The script expects ``lm_eval`` (EleutherAI's harness) to be installed in the
current environment.
"""

from __future__ import annotations

import argparse
import json
import os
import inspect
import sys
from pathlib import Path
from typing import Iterable

# Set dummy env vars before importing train_gpt (which expects torchrun env)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Allow running the script from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from lm_eval import evaluator, tasks, utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from eval_wrapper import load_for_eval


@register_model("nanogpt")
class NanoGPTLMEvalAdapter(LM):
    """Expose train_gpt.GPT checkpoints to lm-evaluation-harness."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_gen_toks: int = 128,
        batch_size: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but no GPU is visible")
        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size or 1

        # Load model and tokenizer via eval_wrapper (handles architecture,
        # checkpoint loading, YaRN replay, bfloat16 conversion, etc.)
        wrapper, hf_tokenizer = load_for_eval(checkpoint_path, device=str(self.device))
        self.model_wrapper = wrapper
        self.tokenizer = hf_tokenizer
        self.bos_token = hf_tokenizer.bos_token_id  # 1
        self._max_length = wrapper.max_seq_len  # 2048

    # ------------------------------------------------------------------
    # Required LM API
    # ------------------------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

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
        return "PleIAs/Baguettotron"

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
            context, continuation = req.args
            context_tokens = self._encode_with_bos(context)
            continuation_tokens = self.tokenizer.encode(continuation)
            if not continuation_tokens:
                pair = (0.0, True)
                outputs.append(pair)
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
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            if isinstance(until, str):
                until = [until]
            max_gen = gen_kwargs.get("max_gen_toks", self._max_gen_toks) if isinstance(gen_kwargs, dict) else self._max_gen_toks

            tokens = self._encode_with_bos(context)
            generated: list[int] = []
            for _ in range(max_gen):
                if len(tokens) >= self._max_length:
                    break
                logits = self._run_model(tokens, valid_tokens=len(tokens))
                next_token = int(logits[-1].argmax().item())
                tokens.append(next_token)
                generated.append(next_token)
                text = self.tokenizer.decode(generated)
                if self._stops_here(text, until):
                    text = self._truncate_until(text, until)
                    generated = self.tokenizer.encode(text)
                    break

            text = self.tokenizer.decode(generated)
            generations.append(text)
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
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        logits = self.model_wrapper(input_ids)  # [1, T, vocab_size]
        return logits[0, :valid_tokens]

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
        "--limit",
        type=float,
        default=None,
        help="Limit number of examples per task (for debugging)",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Write per-sample inputs/outputs to a JSONL file",
    )
    args = parser.parse_args()

    # Build the model wrapper
    model = NanoGPTLMEvalAdapter(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Parse tasks
    task_list = [t.strip() for t in args.tasks.split(",")]

    # Run evaluation
    eval_kwargs = dict(
        model=model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    eval_params = inspect.signature(evaluator.simple_evaluate).parameters
    if args.log_samples and "log_samples" not in eval_params:
        raise RuntimeError("lm_eval simple_evaluate does not support log_samples in this version.")
    if args.output and "output_path" not in eval_params:
        raise RuntimeError("lm_eval simple_evaluate does not support output_path in this version.")
    if "log_samples" in eval_params:
        eval_kwargs["log_samples"] = args.log_samples
    if args.output and "output_path" in eval_params:
        eval_kwargs["output_path"] = args.output

    results = evaluator.simple_evaluate(**eval_kwargs)

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
