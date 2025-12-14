#!/usr/bin/env python3
"""
从 Supabase 的 flows / processes 表导出满足 state_code 条件的 json_ordered 字段。

使用方式示例：
 uv run python scripts/export_supabase_json_ordered.py \
    --state-codes 100 \
    --output-dir data/supabase_exports

依赖环境变量：
  - SUPABASE_URL
  - SUPABASE_SECRET_KEY / SUPABASE_PUBLISHABLE_DEFAULT_KEY / SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY / SUPABASE_KEY
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from supabase import Client, create_client

TABLES: Tuple[str, str] = ("flows", "processes")
DEFAULT_OUTPUT_DIR = Path("data/supabase_exports")
DEFAULT_PAGE_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export json_ordered from Supabase tables."
    )
    parser.add_argument(
        "--supabase-url",
        default=os.getenv("SUPABASE_URL"),
        help="Supabase project URL (默认读取环境变量 SUPABASE_URL)",
    )
    parser.add_argument(
        "--supabase-key",
        default=os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_KEY"),
        help="Supabase API 密钥，默认读取 SUPABASE_SECRET_KEY / SUPABASE_PUBLISHABLE_DEFAULT_KEY / SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY / SUPABASE_KEY",
    )
    parser.add_argument(
        "--state-codes",
        type=int,
        nargs="+",
        help="需要导出的 state_code 值列表。例如: --state-codes 100",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"导出的本地目录 (默认: {DEFAULT_OUTPUT_DIR})，每个表会创建独立子目录",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"分页大小，避免 Supabase API 默认的 1000 行限制 (默认: {DEFAULT_PAGE_SIZE})",
    )
    return parser.parse_args()


def ensure_filters(args: argparse.Namespace) -> None:
    """确保用户提供了 state_code 过滤条件，避免全量扫描。"""
    if args.state_codes is None:
        raise ValueError("请提供 state_code 过滤条件，例如 --state-codes 3 4")


def get_supabase_client(url: str | None, key: str | None) -> Client:
    if not url:
        raise ValueError("缺少 SUPABASE_URL，请通过参数 --supabase-url 或环境变量提供")
    if not key:
        raise ValueError(
            "缺少 Supabase 密钥，请通过 --supabase-key 或 SUPABASE_SECRET_KEY / SUPABASE_PUBLISHABLE_DEFAULT_KEY / "
            "SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY / SUPABASE_KEY 提供"
        )
    return create_client(url, key)


def build_query(
    client: Client,
    table: str,
    state_codes: Optional[List[int]],
) -> Any:
    query = client.table(table).select("id, version, json_ordered, state_code")
    if state_codes:
        query = query.in_("state_code", state_codes)
    return query


def count_rows(
    client: Client,
    table: str,
    state_codes: Optional[List[int]],
) -> Optional[int]:
    """获取满足条件的总行数，用于进度展示。"""
    query = client.table(table).select("id", count="exact")
    if state_codes:
        query = query.in_("state_code", state_codes)
    response = query.limit(1).execute()
    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(f"{table} 计数查询失败: {error}")
    return getattr(response, "count", None)


def fetch_rows(
    client: Client,
    table: str,
    state_codes: Optional[List[int]],
    page_size: int,
) -> Iterable[Dict[str, Any]]:
    offset = 0
    while True:
        query = build_query(client, table, state_codes).range(
            offset, offset + page_size - 1
        )
        response = query.execute()
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"{table} 查询失败: {error}")
        rows = getattr(response, "data", None) or []
        for row in rows:
            yield row
        if len(rows) < page_size:
            break
        offset += page_size


def save_json_ordered(record: Dict[str, Any], dest_dir: Path) -> Optional[Path]:
    record_id = record.get("id")
    version = record.get("version")
    payload = record.get("json_ordered")

    if not record_id or not version:
        return None
    if payload is None:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{record_id}-{version}.json"
    path = dest_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def export_table(
    client: Client,
    table: str,
    output_root: Path,
    state_codes: Optional[List[int]],
    page_size: int,
) -> int:
    dest_dir = output_root / table
    saved = 0
    skipped = 0
    total_rows = count_rows(client, table, state_codes)
    progress = tqdm(
        total=total_rows if total_rows is not None else None,
        desc=f"{table}",
        unit="row",
    )

    for record in fetch_rows(client, table, state_codes, page_size):
        path = save_json_ordered(record, dest_dir)
        if path:
            saved += 1
        else:
            skipped += 1
        progress.update(1)

    progress.close()
    print(
        f"[{table}] 保存 {saved} 条，跳过 {skipped} 条（缺少 id/version/json_ordered） -> {dest_dir}"
    )
    return saved


def main() -> int:
    load_dotenv()
    args = parse_args()
    ensure_filters(args)

    client = get_supabase_client(args.supabase_url, args.supabase_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Supabase URL: {args.supabase_url}")
    print(f"输出目录: {args.output_dir.resolve()}")
    if args.state_codes:
        print(f"state_code in {args.state_codes}")

    total_saved = 0
    for table in TABLES:
        total_saved += export_table(
            client=client,
            table=table,
            output_root=args.output_dir,
            state_codes=args.state_codes,
            page_size=args.page_size,
        )

    print(f"完成：共导出 {total_saved} 条记录到 {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
