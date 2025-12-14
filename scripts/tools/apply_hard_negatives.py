"""
Augment existing training.json with mined hard negatives.

Usage:
    uv run python scripts/apply_hard_negatives.py \
        --training_path data/ft_data/training.json \
        --hard_negatives_path data/ft_data/hard_negatives.jsonl \
        --output_path data/ft_data/training_with_hard_neg.json
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Add hard negatives to an existing training dataset.")
    parser.add_argument("--training_path", type=str, required=True, help="Path to training.json generated in step 1")
    parser.add_argument("--hard_negatives_path", type=str, required=True, help="JSONL file produced by mine_hard_negatives.py")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save the augmented training file")
    return parser.parse_args()


def load_hard_negatives(path: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry.get("id")
            texts = [item["text"] for item in entry.get("hard_negatives", []) if "text" in item]
            if qid and texts:
                mapping.setdefault(qid, []).extend(texts)
    return mapping


def main():
    args = parse_args()
    dataset = Dataset.from_json(args.training_path)
    hard_neg_map = load_hard_negatives(args.hard_negatives_path)

    if not hard_neg_map:
        raise ValueError("No hard negatives found in the provided file.")

    def inject_hard_neg(example):
        extras = hard_neg_map.get(example["id"])
        if not extras:
            return example
        merged = list(example["neg"])
        for text in extras:
            if text not in merged:
                merged.append(text)
        example["neg"] = merged
        return example

    augmented = dataset.map(inject_hard_neg)

    output_path = args.output_path or args.training_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    augmented.to_json(output_path)
    print(f"Augmented training data saved to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()
