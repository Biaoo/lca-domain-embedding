"""
Rendering utilities for HTML reports.
"""
import json
from pathlib import Path
from typing import Dict, List


def load_markdown_sections(md_text: str) -> Dict[str, str]:
    """
    Split markdown by top-level headings (## ) into a dict {heading: content}.
    Assumes headings like '## 摘要', '## 1. 引言', etc.
    """
    sections: Dict[str, str] = {}
    current = None
    buffer: List[str] = []
    lines = md_text.splitlines()
    for line in lines:
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buffer).strip()
            current = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip()
    return sections


def md_paragraphs(text: str) -> str:
    """Convert markdown paragraphs to single-line HTML paragraphs (basic)."""
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return " ".join(parts)


def lookup_metric(metrics: Dict, family: str, k: int):
    """Case/alias tolerant metric lookup."""
    if not isinstance(metrics, dict):
        return None
    if family == "precision":
        candidates = [f"P@{k}", f"Precision@{k}", f"PRECISION@{k}"]
    elif family == "recall":
        candidates = [f"Recall@{k}", f"RECALL@{k}", f"recall@{k}"]
    else:
        candidates = [f"{family.upper()}@{k}", f"{family}@{k}", f"{family.capitalize()}@{k}"]
    for key in candidates:
        if key in metrics:
            return metrics.get(key)
    lower = {k.lower(): v for k, v in metrics.items()}
    for key in candidates:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def format_float(val) -> str:
    try:
        return f"{float(val):.4f}"
    except Exception:
        return "-"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    head_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>"


def build_summary_table(results: List[Dict]) -> str:
    headers = ["模型", "NDCG@10", "Recall@10", "MRR@10", "MAP@10"]
    rows: List[List[str]] = []
    for item in results:
        m = item["metrics"]
        rows.append(
            [
                item["alias"],
                format_float(lookup_metric(m.get("ndcg", {}), "ndcg", 10)),
                format_float(lookup_metric(m.get("recall", {}), "recall", 10)),
                format_float(lookup_metric(m.get("mrr", {}), "mrr", 10)),
                format_float(lookup_metric(m.get("map", {}), "map", 10)),
            ]
        )
    return render_table(headers, rows)


def build_tail_table(results: List[Dict]) -> str:
    headers = ["模型", "NDCG@100", "Recall@100"]
    rows: List[List[str]] = []
    for item in results:
        m = item["metrics"]
        rows.append(
            [
                item["alias"],
                format_float(lookup_metric(m.get("ndcg", {}), "ndcg", 100)),
                format_float(lookup_metric(m.get("recall", {}), "recall", 100)),
            ]
        )
    return render_table(headers, rows)


def build_comparison_table(results: List[Dict], key_metrics: List[str]) -> str:
    raw = next((r for r in results if r["alias"] == "raw"), None)
    headers = ["模型"] + key_metrics
    rows: List[List[str]] = []
    for item in results:
        row = [item["alias"]]
        if raw is None or item["alias"] == "raw":
            row.extend(["-"] * len(key_metrics))
        else:
            for metric in key_metrics:
                family, k = metric.lower().split("@")
                k_int = int(k)
                base = lookup_metric(raw["metrics"].get(family, {}), family, k_int)
                val = lookup_metric(item["metrics"].get(family, {}), family, k_int)
                if base is None or base == 0:
                    row.append("-")
                else:
                    diff = (val - base) / base * 100 if val is not None else 0
                    row.append(f"{diff:+.1f}%")
        rows.append(row)
    return render_table(headers, rows)


def build_metric_table(results: List[Dict], family: str, k_values: List[int]) -> str:
    headers = ["模型"] + [f"{family.upper()}@{k}" for k in k_values]
    rows: List[List[str]] = []
    for item in results:
        metrics = item["metrics"].get(family, {})
        row = [item["alias"]]
        for k in k_values:
            row.append(format_float(lookup_metric(metrics, family, k)))
        rows.append(row)
    return render_table(headers, rows)


def render_detail_tables(results: List[Dict], k_values: List[int]) -> str:
    parts = []
    for family in ["ndcg", "map", "recall", "precision", "mrr"]:
        parts.append(f"<h3>{family.upper()}</h3>")
        parts.append(build_metric_table(results, family, k_values))
    return "\n".join(parts)


def replace_placeholders(template: str, mapping: Dict[str, str]) -> str:
    html = template
    for key, value in mapping.items():
        html = html.replace(key, value)
    return html
