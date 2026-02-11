"""
PleIAs/SYNTH dataset preparation
https://huggingface.co/datasets/PleIAs/SYNTH

Downloads the SYNTH dataset, formats each example with a chat template,
tokenizes with PleIAs/Baguettotron, and writes .bin shards in the same format
as fineweb for the existing data loader.
"""
import os
import sys
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks) # number of tokens
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def filter_synth(example):
    """Keep only examples that have all required fields."""
    return (example.get("query") is not None
            and example.get("synthetic_reasoning") is not None
            and example.get("synthetic_answer") is not None)

def format_synth(example):
    """Format a SYNTH example using a chat template."""
    query = example["query"]
    reasoning = example["synthetic_reasoning"]
    answer = example["synthetic_answer"]
    # Add constraints if this is a RAG exercise
    exercise = example.get("exercise", "")
    if exercise and "rag" in exercise.lower():
        constraints = example.get("constraints", "")
        if constraints:
            query = f"{query}\n{constraints}"
    text = (
        f"<|im_start|>user\n{query}\n<|im_end>\n"
        f"<|im_start|>assistant\n\n<think>\n{reasoning}\n</think>\n\n"
        f"{answer}<|im_end>"
    )
    return text

# --- main ---

shard_size = int(1e8)  # 100M tokens per shard
if len(sys.argv) >= 2:
    shard_size = int(float(sys.argv[1]))

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "synth")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
bos_id = enc.bos_token_id  # 1

# load dataset in streaming mode
print("Loading PleIAs/SYNTH dataset (streaming)...")
ds = load_dataset(
    "PleIAs/SYNTH",
    streaming=True,
    split="train",
    data_files=["synth_*.parquet"],
)

# tokenize and write shards
shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None
total_examples = 0
skipped = 0

for example in ds:
    # filter
    if not filter_synth(example):
        skipped += 1
        continue

    # format and tokenize
    text = format_synth(example)
    tokens = [bos_id] + enc.encode(text)
    tokens_np = np.array(tokens, dtype=np.uint16)
    total_examples += 1

    # shard logic (same as fineweb.py)
    if token_count + len(tokens_np) < shard_size:
        all_tokens_np[token_count:token_count + len(tokens_np)] = tokens_np
        token_count += len(tokens_np)
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens_np))
    else:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"synth_{split}_{shard_index:06d}.bin")
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count + remainder] = tokens_np[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        # leftovers go to next shard
        all_tokens_np[0:len(tokens_np) - remainder] = tokens_np[remainder:]
        token_count = len(tokens_np) - remainder

# write remaining tokens as last shard
if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"synth_{split}_{shard_index:06d}.bin")
    write_datafile(filename, all_tokens_np[:token_count])

print(f"Done. {total_examples:,} examples processed, {skipped:,} skipped, {shard_index + 1} shards written.")
