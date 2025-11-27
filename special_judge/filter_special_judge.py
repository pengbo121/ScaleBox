import argparse
import json
from typing import List

import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and summarize special-judge samples from a JSONL dataset")
    parser.add_argument("--input", default="data/classified_deepseek-chat.jsonl", help="Path to input JSONL file to load with datasets.load_dataset(JSON)")
    parser.add_argument("--output-jsonl", default="data/require_special_judge.jsonl", help="Path to write filtered samples (needs_special_judge=True) as JSONL")
    parser.add_argument("--output-ids", default="data/require_special_judge_ids.json", help="Path to write problem ids of filtered samples as JSON array")
    parser.add_argument("--categories", nargs="*", default=["multiple_solutions", "float_comparison"], help="Extra categories to summarize (count and proportion)")
    return parser.parse_args()


def summarize_categories(dataset, categories: List[str]) -> dict:
    summary = {}
    total = len(dataset)
    for cat in categories:
        data = dataset.filter(lambda x, c=cat: c in x.get("categories", []))
        n = len(data)
        summary[cat] = {
            "count": n,
            "proportion": (n / total) if total else 0.0,
        }
    return summary


def main():
    args = parse_args()

    # Load dataset
    dataset = datasets.load_dataset("json", data_files=args.input, split="train")
    total_number = len(dataset)

    # Filter special judge
    special_judge_data = dataset.filter(lambda x: x.get("needs_special_judge", False) is True)
    special_judge_number = len(special_judge_data)

    # Write outputs
    special_judge_data.to_json(args.output_jsonl, lines=True)
    special_judge_ids = special_judge_data["problem_id"]
    with open(args.output_ids, "w") as f:
        json.dump(list(special_judge_ids), f)

    # Summaries
    category_summary = summarize_categories(dataset, args.categories)

    print(f"Total number of samples: {total_number}")
    print(
        f"Number of samples that need special judge: {special_judge_number}"
    )
    proportion = (special_judge_number / total_number) if total_number else 0.0
    print(
        f"Proportion of samples that need special judge: {proportion:.2%}"
    )
    for cat, stats in category_summary.items():
        print(f"Number of samples with '{cat}' category: {stats['count']}")
        print(
            f"Proportion of samples with '{cat}' category: {stats['proportion']:.2%}"
        )


if __name__ == "__main__":
    main()
