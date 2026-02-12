"""
Generative MMLU eval with <think> support for SYNTH-trained models.
Follows the nanochat evaluation pattern.

Usage:
    python eval/mmlu_gen_eval.py --checkpoint logs/<run_id>/state_step001000.pt
    python eval/mmlu_gen_eval.py --checkpoint <path> --limit 50 --debug
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from eval_wrapper import load_for_eval


LETTERS = ('A', 'B', 'C', 'D')


def render_mc(question, letters, choices):
    lines = [question, ""]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def format_prompt(question, choices, fewshot=None):
    """Format MMLU question in SYNTH chat template with <think> trigger."""
    parts = []
    for ex in (fewshot or []):
        q = render_mc(ex["question"], LETTERS, ex["choices"])
        a = LETTERS[ex["answer"]]
        parts.append(
            f"<|im_start|>user\n{q}\n<|im_end>\n"
            f"<|im_start|>assistant\n\n<think>\nThe answer is {a}.\n</think>\n\n{a}<|im_end>"
        )
    q = render_mc(question, LETTERS, choices)
    parts.append(
        f"<|im_start|>user\n{q}\n<|im_end>\n"
        f"<|im_start|>assistant\n\n<think>\n"
    )
    return "\n".join(parts)


@torch.no_grad()
def generate(model, enc, prompt_text, device, max_new_tokens=300):
    bos = enc.bos_token_id  # 1
    prompt_tokens = [bos] + enc.encode(prompt_text, add_special_tokens=False)
    max_seq_len = model.max_seq_len
    # truncate prompt if it's already too long
    if len(prompt_tokens) >= max_seq_len:
        prompt_tokens = prompt_tokens[-(max_seq_len - max_new_tokens):]
    tokens = list(prompt_tokens)
    generated = []
    for _ in range(max_new_tokens):
        if len(tokens) >= max_seq_len:
            break
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(ids)
        next_id = logits[0, -1].argmax().item()
        tokens.append(next_id)
        generated.append(next_id)
        # Decode with special tokens visible for stop detection
        gen_text = enc.decode(generated, skip_special_tokens=False)
        if "</think>" in gen_text or "<|im_end>" in gen_text or "<|end_of_text|>" in gen_text:
            break
    return generated


def extract_answer(text):
    """Extract first non-whitespace character after the first </think>, or first letter in output."""
    # Decode without special tokens for answer extraction
    if "</think>" in text:
        after = text.split("</think>", 1)[1].strip()
        if after and after[0] in LETTERS:
            return after[0]
    # fallback: first letter found in the output
    for ch in text.strip():
        if ch in LETTERS:
            return ch
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num-fewshot', type=int, default=5)
    parser.add_argument('--max-think-tokens', type=int, default=300)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    enc = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")

    print(f"Loading model from {args.checkpoint}...")
    model, _ = load_for_eval(args.checkpoint, device=args.device)

    print("Loading MMLU...")
    ds_test = load_dataset("cais/mmlu", "all", split="test")
    ds_dev = load_dataset("cais/mmlu", "all", split="dev")

    if args.limit:
        ds_test = ds_test.shuffle(seed=42).select(range(min(args.limit, len(ds_test))))

    # few-shot examples per subject from dev split
    fewshot_by_subject = {}
    for ex in ds_dev:
        subj = ex["subject"]
        if subj not in fewshot_by_subject:
            fewshot_by_subject[subj] = []
        if len(fewshot_by_subject[subj]) < args.num_fewshot:
            fewshot_by_subject[subj].append(ex)

    correct = 0
    total = 0
    subject_stats = {}
    debug_count = {}

    for i, example in enumerate(ds_test):
        subject = example["subject"]
        question = example["question"]
        choices = example["choices"]
        gold = example["answer"]  # 0-3

        fewshot = fewshot_by_subject.get(subject, [])[:args.num_fewshot]
        prompt = format_prompt(question, choices, fewshot=fewshot)
        generated_ids = generate(model, enc, prompt, args.device, max_new_tokens=args.max_think_tokens)
        generated_text = enc.decode(generated_ids, skip_special_tokens=False)
        answer = extract_answer(generated_text)
        is_correct = (answer == LETTERS[gold])

        if is_correct:
            correct += 1
        total += 1

        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1

        if args.debug:
            if subject not in debug_count:
                debug_count[subject] = 0
            if debug_count[subject] < 2:
                print(f"\n[{subject} #{debug_count[subject]}]")
                print(f"  Q: {question[:200]}")
                print(f"  Gold: {LETTERS[gold]} | Pred: {answer} | {'OK' if is_correct else 'WRONG'}")
                print(f"  Extracted answer: '{answer}' from extract_answer()")
                print(f"  First 20 generated token IDs: {generated_ids[:20]}")
                print(f"  Full generated output (with special tokens):\n{generated_text}")
                debug_count[subject] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds_test)}] acc: {correct/total:.4f}")

    print("\n" + "=" * 60)
    print(f"{'Subject':<45} {'Acc':>8} {'N':>6}")
    print("=" * 60)
    for subj in sorted(subject_stats.keys()):
        s = subject_stats[subj]
        acc = s["correct"] / s["total"]
        print(f"{subj:<45} {acc:>8.4f} {s['total']:>6}")
    print("=" * 60)
    print(f"{'OVERALL':<45} {correct/total:>8.4f} {total:>6}")


if __name__ == "__main__":
    main()
