import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

# Download SYNTH parquet shards from Hugging Face into data/synth/.
# This mirrors the FineWeb cached_* scripts but uses snapshot_download
# to fetch all synth_*.parquet files at once.


def parse_args():
    parser = argparse.ArgumentParser(description="Download SYNTH parquet shards")
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Download only the first N synth_*.parquet files (sorted by name)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = os.path.join(os.path.dirname(__file__), "synth")
    os.makedirs(local_dir, exist_ok=True)
    if args.num_files is None:
        snapshot_download(
            repo_id="PleIAs/SYNTH",
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=["synth_*.parquet"],
        )
        return

    files = [
        name for name in list_repo_files("PleIAs/SYNTH", repo_type="dataset")
        if name.startswith("synth_") and name.endswith(".parquet")
    ]
    files = sorted(files)[: args.num_files]
    for name in files:
        hf_hub_download(
            repo_id="PleIAs/SYNTH",
            repo_type="dataset",
            filename=name,
            local_dir=local_dir,
        )


if __name__ == "__main__":
    main()
