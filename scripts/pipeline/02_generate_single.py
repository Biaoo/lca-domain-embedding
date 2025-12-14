"""Generate queries for a single or multiple specified markdown files.

Usage:
    # Single file
    python scripts/step2_generate_single.py data/markdown/xxx.md

    # Multiple files
    python scripts/step2_generate_single.py data/markdown/a.md data/markdown/b.md

    # With custom output file
    python scripts/step2_generate_single.py data/markdown/xxx.md --output-file data/lca_embedding_dataset.csv
"""

import json
import csv
import re
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))

from src.utils.llm_generate_query import generate_query

ROOT_PATH = root_path
DEFAULT_OUTPUT_FILE = ROOT_PATH / "data" / "lca_embedding_dataset.csv"


def extract_metadata_from_markdown(content: str, filename: str) -> dict:
    """Extract UUID and version from markdown content."""
    uuid_match = re.search(r"\*\*UUID:\*\*\s*`([^`]+)`", content)
    uuid = uuid_match.group(1) if uuid_match else filename.replace(".md", "")

    version_match = re.search(r"\*\*Version:\*\*\s*(\S+)", content)
    version = version_match.group(1) if version_match else ""

    return {"uuid": uuid, "version": version}


def process_single_file(md_path: Path, dataset_type: str = "process") -> list[dict]:
    """Process a single markdown file and generate query records."""
    content = md_path.read_text(encoding="utf-8")
    metadata = extract_metadata_from_markdown(content, md_path.name)

    query_result = generate_query(content, dataset_type=dataset_type)
    queries_data = json.loads(query_result)
    queries = queries_data.get("queries", [])

    results = []
    for query in queries:
        results.append(
            {
                "query": query,
                "dataset_uuid": metadata["uuid"],
                "dataset_version": metadata["version"],
                "dataset_content": content,
                "dataset_type": dataset_type,
            }
        )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate queries for specified markdown file(s)"
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="Markdown file(s) to process"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="process",
        help="Source label to store in dataset_type column (e.g., process or flow)",
    )

    args = parser.parse_args()

    # Check if output file exists to determine if we need header
    write_header = not args.output_file.exists()

    success_count = 0
    failed_count = 0
    total_queries = 0

    with open(args.output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "dataset_uuid",
                "dataset_version",
                "dataset_content",
                "dataset_type",
            ]
        )

        if write_header:
            writer.writeheader()

        for md_file in args.files:
            if not md_file.exists():
                print(f"✗ File not found: {md_file}")
                failed_count += 1
                continue

            try:
                print(f"Processing: {md_file.name}...")
                records = process_single_file(md_file, dataset_type=args.dataset_type)
                writer.writerows(records)
                f.flush()
                success_count += 1
                total_queries += len(records)
                print(f"✓ {md_file.name}: generated {len(records)} queries")
            except Exception as e:
                failed_count += 1
                print(f"✗ {md_file.name}: {e}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total queries: {total_queries}")
    print(f"  Output: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
