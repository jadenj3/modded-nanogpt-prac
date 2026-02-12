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
    """Expose train_gpt.GPT checkpoints to lm-evaluation-harness.

    This adapter bridges the gap between our custom GPT model (loaded via
    eval_wrapper.load_for_eval) and EleutherAI's lm-evaluation-harness. It
    implements the three core evaluation methods the harness expects:
    loglikelihood, loglikelihood_rolling, and generate_until.

    The model uses the PleIAs/Baguettotron tokenizer (vocab_size=65536) and
    has a max sequence length of 2048.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_gen_toks: int = 128,
        batch_size: int | None = None,
        **_: object,
    ) -> None:
        """Initialize the adapter by loading a checkpoint and tokenizer.

        Args:
            checkpoint_path: Filesystem path to a training checkpoint .pt file,
                e.g. "logs/<run_id>/state_step001000.pt". The checkpoint contains
                the model state_dict under the "model" key.
            device: Torch device string, e.g. "cuda" or "cpu".
            dtype: Ignored (kept for interface compat). Model is always loaded
                in bfloat16 via eval_wrapper.
            max_gen_toks: Maximum number of new tokens to generate per request
                in generate_until. Default 128.
            batch_size: Number of requests to process at once. Currently each
                request is processed sequentially regardless of this value.

        Sets up:
            self.model_wrapper: InferenceWrapper that accepts [B, T] int64 input_ids
                and returns [B, T, vocab_size] logits.
            self.tokenizer: HuggingFace AutoTokenizer (PleIAs/Baguettotron).
            self.bos_token: BOS token id (1).
            self._max_length: Max sequence length (2048).
        """
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

    def tok_encode(self, string: str, **_: object) -> list[int]:
        """Tokenize a string into a list of integer token IDs.

        Args:
            string: Raw text to tokenize, e.g. "The capital of France is".

        Uses the Baguettotron tokenizer's encode method. Does NOT prepend BOS
        (use _encode_with_bos for that).

        Returns:
            list[int]: Token IDs, variable length depending on input text.
                Used internally by lm-eval for caching and request construction.
        """
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: Iterable[int], **_: object) -> str:
        """Decode a sequence of token IDs back into a string.

        Args:
            tokens: Iterable of integer token IDs, e.g. [1, 450, 3829, 315].

        Converts token IDs back to human-readable text using Baguettotron.

        Returns:
            str: Decoded text. Used internally by lm-eval for logging and
                stop-sequence detection during generation.
        """
        return self.tokenizer.decode(list(tokens))

    # ------------------------------------------------------------------
    # Harness entrypoints
    # ------------------------------------------------------------------
    def loglikelihood(self, requests):
        """Score how likely the model considers each (context, continuation) pair.

        This is the primary evaluation method for multiple-choice tasks like MMLU.
        For each request, the harness provides a context string (e.g. the question
        + few-shot examples) and a continuation string (e.g. one answer choice).
        We compute the model's log-probability of the continuation given the context.

        Args:
            requests: List of lm_eval Request objects. Each has req.args = (context, continuation)
                where both are strings. For MMLU 5-shot, context is ~1000-2000 tokens
                (few-shot examples + question), continuation is typically 1-5 tokens
                (the answer choice like " A" or " Paris").

        Processing:
            1. Tokenize context (with BOS prepended) and continuation separately.
            2. Concatenate into one sequence: [BOS, context_tokens..., continuation_tokens...].
            3. If the sequence exceeds max_seq_len (2048), truncate from the LEFT,
               keeping all continuation tokens intact (we need those for scoring).
            4. Run a single forward pass through the model to get logits at every position.
            5. Extract logits at positions aligned with the continuation tokens.
               Specifically, logits[i] predicts token[i+1], so we take logits from
               position (context_len - 1) to (context_len - 1 + len(continuation)).
            6. Compute log_softmax over vocab, then gather the log-probs for the
               actual continuation token IDs. Sum these for total log-likelihood.
            7. Also check if the model's greedy prediction matches every continuation
               token (the "greedy" flag).

        Returns:
            list[tuple[float, bool]]: One pair per request:
                - float: Sum of log-probabilities of continuation tokens (always <= 0).
                    More negative = model considers continuation less likely.
                - bool: True if the model's argmax prediction matches every token
                    in the continuation (greedy exact match).
                The harness uses these to pick the highest-scoring continuation
                as the model's answer for multiple-choice tasks.
        """
        outputs: list[tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            context_tokens = self._encode_with_bos(context)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            if not continuation_tokens:
                pair = (0.0, True)
                outputs.append(pair)
                continue

            tokens = context_tokens + continuation_tokens
            # Truncate from left if exceeding max length, keeping all continuation tokens
            if len(tokens) > self._max_length:
                tokens = tokens[-self._max_length:]
                context_len = len(tokens) - len(continuation_tokens)
            else:
                context_len = len(context_tokens)
            logits = self._run_model(tokens, valid_tokens=len(tokens) - 1)
            start = max(context_len - 1, 0)
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
        """Compute total log-likelihood of entire text sequences (no context/continuation split).

        Used by perplexity-style benchmarks (e.g. WikiText) where we want to score
        the model's probability of an entire document, not just an answer choice.

        Args:
            requests: List of lm_eval Request objects. Each has req.args = (text,)
                where text is a full document string, potentially thousands of tokens
                long (much longer than max_seq_len).

        Processing:
            1. Tokenize the full text with BOS prepended.
            2. Process in sliding windows of max_seq_len (2048) tokens. Each window
               overlaps the previous by 1 token to maintain context continuity.
            3. For each window, run a forward pass and compute log-probs of each
               token given its predecessors. Sum all token log-probs.
            4. Accumulate the total across all windows.

        Returns:
            list[float]: One float per request — the total log-likelihood of the
                full text (always <= 0). Used by the harness to compute perplexity
                as exp(-total / num_tokens).
        """
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
        """Autoregressively generate text until a stop sequence is hit or limits are reached.

        Used by open-ended generation benchmarks (e.g. HumanEval, GSM8K) where the
        model must produce a free-form answer rather than just scoring answer choices.

        Args:
            requests: List of lm_eval Request objects. Each has req.args = (context, gen_kwargs)
                where:
                - context (str): The prompt text, e.g. a question or code prefix.
                - gen_kwargs (dict): Generation parameters including:
                    - "until" (list[str]): Stop sequences, e.g. ["\\n", "<|im_end>"].
                        Generation halts when any of these appear in the output.
                    - "max_gen_toks" (int): Override for max tokens to generate.

        Processing:
            1. Tokenize the context with BOS prepended.
            2. Loop up to max_gen_toks times:
               a. Run a forward pass on all tokens so far (prompt + generated).
               b. Take argmax of the last position's logits as the next token.
               c. Append to the token list and decode the generated portion to text.
               d. Check if any stop sequence appears in the decoded text. If so,
                  truncate at the first occurrence and stop.
               e. Also stop if total sequence length reaches max_seq_len (2048).
            3. Greedy decoding only (no sampling, temperature, or top-k/top-p).

        Returns:
            list[str]: One generated string per request, with stop sequences removed.
                Used by the harness for exact-match or functional correctness evaluation.
        """
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
                    generated = self.tokenizer.encode(text, add_special_tokens=False)
                    break

            text = self.tokenizer.decode(generated)
            generations.append(text)
        return generations

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _encode_with_bos(self, text: str) -> list[int]:
        """Tokenize text and prepend the BOS token.

        Args:
            text: Raw text string to tokenize, e.g. "Question: What is 2+2?".

        Encodes the text using Baguettotron, then prepends BOS (token ID 1) as the
        first token, matching the training data format where every document starts
        with BOS.

        Returns:
            list[int]: Token IDs of length (1 + num_text_tokens), starting with
                BOS. Used as input to _run_model for all evaluation methods.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.bos_token] + tokens

    @torch.no_grad()
    def _run_model(self, tokens: list[int], valid_tokens: int) -> torch.Tensor:
        """Run a single forward pass through the model and return logits.

        This is the central inference method that all three harness entrypoints
        (loglikelihood, loglikelihood_rolling, generate_until) call.

        Args:
            tokens: List of token IDs to feed the model, length T. Typically
                starts with BOS (1) followed by encoded text. Max usable length
                is self._max_length (2048); longer sequences are truncated from
                the LEFT to preserve the most recent context.
            valid_tokens: Number of token positions to return logits for, counted
                from the start of the (possibly truncated) sequence. For
                loglikelihood this is T-1 (we don't need logits at the last
                position since there's no next token to predict). For
                generate_until this is T (we need the last position's logits
                to pick the next token).

        Processing:
            1. If len(tokens) > max_length, truncate from the left and clamp
               valid_tokens accordingly.
            2. Reshape tokens into a [1, T] int64 tensor on the model's device.
            3. Call InferenceWrapper which handles padding to multiple of 16,
               constructing seqlens, running the GPT forward pass in bfloat16,
               and removing padding from the output.
            4. Extract logits for the first valid_tokens positions.

        Returns:
            torch.Tensor of shape (valid_tokens, vocab_size) in bfloat16.
                logits[i] contains the model's next-token prediction distribution
                after seeing tokens[0:i+1]. Used by callers to compute log-probs
                (loglikelihood) or to pick the argmax next token (generate_until).
        """
        # Safety truncation from left if sequence exceeds max length
        if len(tokens) > self._max_length:
            tokens = tokens[-self._max_length:]
            valid_tokens = min(valid_tokens, len(tokens))
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        logits = self.model_wrapper(input_ids)  # [1, T, vocab_size]
        return logits[0, :valid_tokens]

    def _stops_here(self, text: str, stop_sequences: list[str]) -> bool:
        """Check whether any stop sequence has appeared in the generated text.

        Args:
            text: The decoded generated text so far, e.g. "The answer is Paris\\n".
            stop_sequences: List of strings to watch for, e.g. ["\\n", "<|im_end>"].

        Simple substring containment check — returns True as soon as any stop
        sequence is found anywhere in text.

        Returns:
            bool: True if at least one stop sequence is present in text. Used by
                generate_until to decide when to stop the generation loop.
        """
        return any(stop in text for stop in stop_sequences)

    def _truncate_until(self, text: str, stop_sequences: list[str]) -> str:
        """Truncate text at the first occurrence of any stop sequence.

        Args:
            text: The decoded generated text containing a stop sequence,
                e.g. "The answer is Paris\\nExtra stuff".
            stop_sequences: List of stop strings, e.g. ["\\n", "<|im_end>"].

        Finds the earliest occurrence of each stop sequence and cuts the text
        at that point, excluding the stop sequence itself.

        Returns:
            str: Truncated text with everything from the first stop sequence
                onward removed, e.g. "The answer is Paris". Used by
                generate_until to clean the final output before returning.
        """
        for stop in stop_sequences:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        return text


def main():
    """CLI entrypoint: parse args, load model, run lm-eval-harness, print results."""
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
