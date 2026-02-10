"""
Evaluate the CORE metric for a trained model.

Usage:
    python run_core_eval.py <checkpoint_path>
    python run_core_eval.py logs/run_id/state_step001555.pt --max-per-task 100
"""
import os
import csv
import time
import json
import yaml
import random
import zipfile
import tempfile
import shutil
import argparse
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from eval_wrapper import load_for_eval
from core_eval import evaluate_task

# Eval bundle URL (~162MB)
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_BUNDLE_DIR = Path(__file__).parent / "eval_bundle"


class TokenizerWrapper:
    """Wrapper to provide the interface core_eval expects."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._bos_token_id = tokenizer.bos_token_id

    def get_bos_token_id(self):
        return self._bos_token_id

    def __call__(self, prompts, prepend=None):
        """Tokenize prompts, optionally prepending a token."""
        result = []
        for prompt in prompts:
            tokens = self._tokenizer.encode(prompt)
            if prepend is not None:
                tokens = [prepend] + tokens
            result.append(tokens)
        return result


def download_eval_bundle():
    """Download and extract eval bundle if not present."""
    if EVAL_BUNDLE_DIR.exists():
        return

    print(f"Downloading eval bundle from {EVAL_BUNDLE_URL}...")
    import urllib.request

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "eval_bundle.zip")
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        extracted = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted, str(EVAL_BUNDLE_DIR))

    print(f"Eval bundle extracted to {EVAL_BUNDLE_DIR}")


def evaluate_model(model, tokenizer, device, max_per_task=-1, chat_template=False):
    """Evaluate model on the CORE benchmark."""
    download_eval_bundle()

    config_path = EVAL_BUNDLE_DIR / "core.yaml"
    data_base_path = EVAL_BUNDLE_DIR / "eval_data"
    eval_meta_path = EVAL_BUNDLE_DIR / "eval_meta_data.csv"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baselines
    random_baselines = {}
    with open(eval_meta_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    results = {}
    centered_results = {}

    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' '),
            'chat_template': chat_template
        }
        print(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot)... ", end='', flush=True)

        # Load data
        data_path = data_base_path / task_meta['dataset_uri']
        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle and optionally limit
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered

        elapsed = time.time() - start_time
        print(f"acc: {accuracy:.4f} | centered: {centered:.4f} | {elapsed:.1f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--max-per-task', type=int, default=-1,
                        help='Max examples per task (-1 = all)')
    parser.add_argument('--chat-template', action='store_true', default=False,
                        help='Wrap eval prompts in chat template (for SYNTH-trained models)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, hf_tokenizer = load_for_eval(args.checkpoint, device=args.device)
    tokenizer = TokenizerWrapper(hf_tokenizer)

    print(f"Evaluating on CORE benchmark...")
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = evaluate_model(model, tokenizer, args.device, max_per_task=args.max_per_task,
                             chat_template=args.chat_template)

    print("\n" + "="*60)
    print(f"{'Task':<35} {'Accuracy':<10} {'Centered':<10}")
    print("="*60)
    for label in out['results']:
        print(f"{label:<35} {out['results'][label]:<10.4f} {out['centered_results'][label]:<10.4f}")
    print("="*60)
    print(f"{'CORE METRIC':<35} {'':<10} {out['core_metric']:<10.4f}")


if __name__ == "__main__":
    main()
