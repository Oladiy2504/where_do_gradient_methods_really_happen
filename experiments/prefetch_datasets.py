from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Portable root for all datasets (default ./data). Copy this to the cluster.",
    )
    parser.add_argument(
        "--fineweb-shards",
        type=int,
        default=1,
        help="FineWeb training shards to fetch (0-indexed). PAPER_TASKS uses 1; "
        "each shard is ~200 MB.",
    )
    parser.add_argument(
        "--skip-fineweb",
        action="store_true",
        help="Skip the FineWeb GPT dataset (the only large download).",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    hf_home = os.path.join(data_dir, "hf")
    os.makedirs(hf_home, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)

    from src.models.data import get_cifar10, get_fineweb, get_mnist, get_sst2

    print(f"[prefetch] data-dir = {data_dir}")
    print(f"[prefetch] HF_HOME  = {os.environ['HF_HOME']}")

    print("[prefetch] MNIST ...")
    get_mnist(root=data_dir, num_workers=0)

    print("[prefetch] CIFAR10 ...")
    get_cifar10(root=data_dir, num_workers=0)

    print("[prefetch] SST2 (nyu-mll/glue) ...")
    get_sst2(num_workers=0)

    if args.skip_fineweb:
        print("[prefetch] FineWeb skipped (--skip-fineweb)")
    else:
        print(f"[prefetch] FineWeb: {args.fineweb_shards} shard(s) ...")
        get_fineweb(
            num_shards=args.fineweb_shards,
            cache_dir=os.path.join(data_dir, "fineweb"),
            num_workers=0,
        )

    print(f"\n[prefetch] done. Copy '{data_dir}' to the cluster, then set:")
    print(f"  export HF_HOME='$PWD/{os.path.relpath(hf_home)}'")
    print("  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1")

if __name__ == "__main__":
    main()
