"""
SYNTH dataset preprocessing (PleIAs/SYNTH).

Creates .bin shards compatible with the training loader:
- 256 int32 header (magic/version/token_count)
- uint16 token IDs (GPT-2 tokenizer)

The first shard is used as validation, remaining shards are training.
"""
import os
import glob
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

_ENC = None
_EOT = None


def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in PyTorch/C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(toks)  # number of tokens after header (uint16)
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


def parse_args():
    parser = argparse.ArgumentParser(description="SYNTH dataset preprocessing")
    parser.add_argument(
        "--data_files",
        type=str,
        default="synth_*.parquet",
        help="Dataset file glob (default: synth_*.parquet)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10**8,
        help="Size of each shard in tokens",
    )
    return parser.parse_args()


def _init_tokenizer():
    global _ENC, _EOT
    _ENC = tiktoken.get_encoding("gpt2")
    _EOT = _ENC._special_tokens["<|endoftext|>"]


def _tokenize(doc):
    text = _format_doc(doc)
    tokens = [_EOT]
    tokens.extend(_ENC.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def _format_doc(doc):
    return (
        "<|im_start|>user\n"
        + doc["query"]
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n\n<think>\n"
        + doc["synthetic_reasoning"]
        + "\n</think>\n\n"
        + doc["synthetic_answer"]
        + "<|im_end|>"
    )


def main():
    args = parse_args()
    local_dir = "synth"
    data_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(data_cache_dir, exist_ok=True)

    local_files = sorted(glob.glob(args.data_files))
    if local_files:
        dataset = load_dataset(
            "parquet",
            data_files=local_files,
            split="train",
        )
    else:
        dataset = load_dataset(
            "PleIAs/SYNTH",
            split="train",
            data_files=[args.data_files],
        )

    dataset = dataset.filter(
        lambda x: "query" in x and "synthetic_reasoning" in x and "synthetic_answer" in x
    )

    nprocs = max(1, os.cpu_count() - 2)
    with mp.Pool(nprocs, initializer=_init_tokenizer) as pool:
        shard_index = 0
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(_tokenize, dataset, chunksize=16):
            if token_count + len(tokens) < args.shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(data_cache_dir, f"synth_{split}_{shard_index:06d}.bin")
                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(data_cache_dir, f"synth_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
