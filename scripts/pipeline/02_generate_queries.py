"""Step 2: Generate search queries from markdown files using LLM.

Reads markdown files from step 1 (process + sampled flow), generates queries using LLM,
and outputs training dataset with columns: query | dataset_uuid | dataset_version | dataset_content | dataset_type

Features:
- Parallel processing with configurable concurrency (default: 10)
- Batch writing to CSV after each batch completes
- Resumable from a specific index if interrupted
"""

import json
import csv
import re
import random
import signal
import sys
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

root_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))

from src.utils.llm_generate_query import generate_query

ROOT_PATH = root_path
DEFAULT_ROOT_INPUT_DIR = ROOT_PATH / "data" / "supabase_exports_markdown"
DEFAULT_PROCESS_DIR = DEFAULT_ROOT_INPUT_DIR / "processs"
DEFAULT_FLOW_DIR = DEFAULT_ROOT_INPUT_DIR / "flows"
DEFAULT_OUTPUT_FILE = ROOT_PATH / "data" / "tiangong_lca_embedding_dataset.csv"
DEFAULT_CONCURRENCY = 10
DEFAULT_FLOW_RATIO = 1.0
DEFAULT_RANDOM_SEED = 520


class ProgressTracker:
    """Track progress and handle interruption gracefully."""

    def __init__(self, total: int, start_index: int = 0):
        self.total = total
        self.start_index = start_index
        self.current_index = start_index
        self.last_completed_batch_end = start_index
        self.success_count = 0
        self.failed_count = 0
        self.total_queries = 0
        self._interrupted = False

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal and print resume information."""
        self._interrupted = True
        print("\n" + "=" * 60)
        print("INTERRUPTED!")
        print("=" * 60)
        print(f"Progress: {self.current_index}/{self.total}")
        print(f"Successfully processed: {self.success_count}")
        print(f"Failed: {self.failed_count}")
        print(f"Total queries generated: {self.total_queries}")
        print(f"Last completed batch ended at: {self.last_completed_batch_end}")
        print("-" * 60)
        print(f"To resume, run with --start-index {self.last_completed_batch_end}")
        print("=" * 60)
        sys.exit(1)

    def update(self, success: bool, queries_count: int = 0):
        """Update progress after processing a record."""
        self.current_index += 1
        if success:
            self.success_count += 1
            self.total_queries += queries_count
        else:
            self.failed_count += 1

    def mark_batch_complete(self, batch_end_index: int):
        """Mark a batch as fully written to disk."""
        self.last_completed_batch_end = batch_end_index


def extract_metadata_from_markdown(content: str, filename: str) -> dict:
    """Extract UUID and version from markdown content.

    Args:
        content: Markdown content
        filename: Filename (used as fallback for UUID)

    Returns:
        Dict with uuid and version
    """
    uuid_match = re.search(r"\*\*UUID:\*\*\s*`([^`]+)`", content)
    uuid = uuid_match.group(1) if uuid_match else filename.replace(".md", "")

    version_match = re.search(r"\*\*Version:\*\*\s*(\S+)", content)
    version = version_match.group(1) if version_match else ""

    return {"uuid": uuid, "version": version}


def process_single_file(
    md_path: Path, idx: int, dataset_type: str = "process"
) -> tuple[int, list[dict], Optional[str]]:
    """Process a single markdown file and generate query records.

    Args:
        md_path: Path to the markdown file
        idx: Index of the file for ordering
        dataset_type: Source label (process / flow)

    Returns:
        Tuple of (index, records list, error message or None)
    """
    try:
        content = md_path.read_text(encoding="utf-8")
        metadata = extract_metadata_from_markdown(content, md_path.name)

        query_result = generate_query(content, dataset_type=dataset_type)
        queries_data = json.loads(query_result) if query_result else {}
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

        return (idx, results, None)
    except Exception as e:
        return (idx, [], f"{md_path.name}: {e}")


def generate_query_dataset(
    input_dir: Optional[Path] = None,
    process_dir: Optional[Path] = None,
    flow_dir: Optional[Path] = None,
    flow_ratio: float = DEFAULT_FLOW_RATIO,
    flow_limit: Optional[int] = None,
    output_file: Optional[Path] = None,
    start_index: int = 0,
    append_mode: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """Generate query dataset from markdown files with parallel processing.

    Args:
        input_dir: Backward compatible alias for process_dir
        process_dir: Directory containing process markdown files
        flow_dir: Directory containing flow markdown files
        flow_ratio: Flow:process file ratio (1.0 -> 1:1)
        flow_limit: Optional hard cap on flow files to sample
        output_file: Output CSV file path
        start_index: Index to start processing from (for resuming)
        append_mode: If True, append to existing file instead of overwriting
        concurrency: Number of parallel workers (default: 10)
        random_seed: Seed for sampling and order shuffling

    Returns:
        Summary dict with success/failed counts
    """
    if input_dir is None:
        input_dir = DEFAULT_PROCESS_DIR
    if output_file is None:
        output_file = DEFAULT_OUTPUT_FILE

    process_dir = process_dir or input_dir
    if process_dir is None:
        process_dir = DEFAULT_PROCESS_DIR

    if flow_dir is None:
        candidate_flow = process_dir.parent / "flows"
        flow_dir = candidate_flow if candidate_flow.exists() else DEFAULT_FLOW_DIR

    if not process_dir.exists():
        print(f"Error: Process directory not found: {process_dir}")
        print("Please run step1_generate_markdown.py first.")
        return {"success": 0, "failed": 0, "total_queries": 0}

    output_file.parent.mkdir(parents=True, exist_ok=True)

    process_files = sorted(process_dir.glob("*.md"))
    flow_files = sorted(flow_dir.glob("*.md")) if flow_dir and flow_dir.exists() else []

    rng = random.Random(random_seed)
    flow_ratio = max(flow_ratio, 0)
    target_flow = int(len(process_files) * flow_ratio)
    if flow_limit is not None:
        target_flow = min(target_flow, flow_limit)
    target_flow = min(target_flow, len(flow_files))
    sampled_flow_files = (
        rng.sample(flow_files, target_flow)
        if target_flow and len(flow_files) > target_flow
        else flow_files[:target_flow]
    )

    files_with_type = [("process", path) for path in process_files] + [
        ("flow", path) for path in sampled_flow_files
    ]
    rng.shuffle(files_with_type)

    total_files = len(files_with_type)

    print(f"Process markdown dir: {process_dir} ({len(process_files)} files)")
    if flow_dir and flow_dir.exists():
        print(
            f"Flow markdown dir: {flow_dir} "
            f"(available {len(flow_files)}, sampled {len(sampled_flow_files)}, ratio={flow_ratio}, limit={flow_limit})"
        )
    else:
        print("Flow markdown dir: None (skip flow sampling)")
    print(f"Total markdown files to process: {total_files}")
    print(f"Concurrency: {concurrency}")

    if start_index > 0:
        print(f"Resuming from index {start_index}")
        if start_index >= total_files:
            print(f"Error: start_index ({start_index}) >= total files ({total_files})")
            return {"success": 0, "failed": 0, "total_queries": 0}

    tracker = ProgressTracker(total_files, start_index)

    file_mode = "a" if append_mode and start_index > 0 else "w"
    write_header = not (append_mode and start_index > 0 and output_file.exists())

    files_to_process = files_with_type[start_index:]

    with open(output_file, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "dataset_uuid",
                "dataset_version",
                "dataset_content",
                "dataset_type",
            ],
        )

        if write_header:
            writer.writeheader()

        # Process in batches of `concurrency` size
        pbar = tqdm(total=len(files_to_process), desc="Generating queries")

        for batch_start in range(0, len(files_to_process), concurrency):
            batch_end = min(batch_start + concurrency, len(files_to_process))
            batch_files = files_to_process[batch_start:batch_end]
            batch_indices = range(start_index + batch_start, start_index + batch_end)

            # Process batch in parallel
            batch_results = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        process_single_file, md_file, idx, dataset_type
                    ): (idx, dataset_type, md_file)
                    for (dataset_type, md_file), idx in zip(batch_files, batch_indices)
                }

                for future in as_completed(futures):
                    idx, dataset_type, md_file = futures[future]
                    batch_idx, records, error = future.result()
                    batch_results.append((batch_idx, records, error))

                    if error:
                        tracker.update(success=False)
                        tqdm.write(f"✗ [{idx}][{dataset_type}] {error}")
                    else:
                        tracker.update(success=True, queries_count=len(records))

                    pbar.update(1)
                    pbar.set_postfix(
                        {"ok": tracker.success_count, "fail": tracker.failed_count}
                    )

            # Sort by index to maintain order, then write batch
            batch_results.sort(key=lambda x: x[0])
            for idx, records, error in batch_results:
                if records:
                    writer.writerows(records)

            f.flush()
            tracker.mark_batch_complete(start_index + batch_end)

        pbar.close()

    print("\n" + "=" * 60)
    print("Step 2 complete: Query generation")
    print("=" * 60)
    print(f"  Success: {tracker.success_count}")
    print(f"  Failed: {tracker.failed_count}")
    print(f"  Total queries: {tracker.total_queries}")
    print(f"  Output: {output_file}")
    print("=" * 60)

    return {
        "success": tracker.success_count,
        "failed": tracker.failed_count,
        "total_queries": tracker.total_queries,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 2: Generate search queries from markdown files using LLM"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="(Optional) Deprecated alias for --process-dir; kept for backward compatibility.",
    )
    parser.add_argument(
        "--process-dir",
        type=Path,
        default=None,
        help=f"Directory containing process markdown files (default: {DEFAULT_PROCESS_DIR})",
    )
    parser.add_argument(
        "--flow-dir",
        type=Path,
        default=None,
        help="Directory containing flow markdown files (default: sibling 'flows' of process-dir if present)",
    )
    parser.add_argument(
        "--flow-ratio",
        type=float,
        default=DEFAULT_FLOW_RATIO,
        help="Flow:process file ratio (1.0 = sample same number of flows as processes)",
    )
    parser.add_argument(
        "--flow-limit",
        type=int,
        default=None,
        help="Optional hard cap on flow markdown files to include",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index to start processing from, for resuming (default: 0)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file (use with --start-index)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of parallel workers (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for flow sampling and ordering (default: {DEFAULT_RANDOM_SEED})",
    )

    args = parser.parse_args()

    generate_query_dataset(
        input_dir=args.input_dir or args.process_dir,
        process_dir=args.process_dir,
        flow_dir=args.flow_dir,
        flow_ratio=args.flow_ratio,
        flow_limit=args.flow_limit,
        output_file=args.output_file,
        start_index=args.start_index,
        append_mode=args.append,
        concurrency=args.concurrency,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
