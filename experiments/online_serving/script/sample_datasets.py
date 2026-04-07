"""
Sample from ShareGPT and WildChat datasets.
Generates 6 small datasets: 3 seeds × 2 datasets.
"""

import argparse
import json
import os

from datasets import load_dataset


def sample_sharegpt(seed, n=500):
    """Sample from ShareGPT in vllm-compatible format.

    vllm expects: [{"conversations": [{"from": "human", "value": "..."},
                                       {"from": "gpt", "value": "..."}]}]
    """
    ds = load_dataset("RyokoAI/ShareGPT52K", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n * 2, len(ds))))

    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        convs = row.get("conversations", [])
        # Need at least 2 turns (human + gpt) for vllm sharegpt format
        if len(convs) >= 2:
            human_msg = convs[0].get("value", "").strip()
            if human_msg and len(human_msg) > 10:
                samples.append({"conversations": convs})

    return samples[:n]


def sample_wildchat(seed, n=500):
    """Sample from WildChat, convert to vllm sharegpt format."""
    ds = load_dataset("allenai/WildChat-1M", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n * 2, len(ds))))

    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        convs = row.get("conversation", [])
        if len(convs) >= 2:
            # Convert WildChat format to ShareGPT format
            sharegpt_convs = []
            for msg in convs:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role == "user":
                    sharegpt_convs.append({"from": "human", "value": content})
                elif role == "assistant":
                    sharegpt_convs.append({"from": "gpt", "value": content})
            if len(sharegpt_convs) >= 2 and sharegpt_convs[0]["value"]:
                samples.append({"conversations": sharegpt_convs})

    return samples[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="sampled_datasets")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")

        # ShareGPT
        print(f"Sampling {args.num_samples} from ShareGPT...")
        sharegpt = sample_sharegpt(seed, args.num_samples)
        path = os.path.join(args.output_dir, f"sharegpt_seed{seed}_n{len(sharegpt)}.json")
        with open(path, "w") as f:
            json.dump(sharegpt, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(sharegpt)} samples to {path}")

        # WildChat
        print(f"Sampling {args.num_samples} from WildChat...")
        wildchat = sample_wildchat(seed, args.num_samples)
        path = os.path.join(args.output_dir, f"wildchat_seed{seed}_n{len(wildchat)}.json")
        with open(path, "w") as f:
            json.dump(wildchat, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(wildchat)} samples to {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
