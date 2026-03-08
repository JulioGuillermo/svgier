"""Builds a bootstrap SVG dataset from a source JSONL file.

This script normalizes records, validates SVG basics, filters invalid samples,
and writes train/val/test JSONL splits plus summary artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

ALLOWED_SPLITS = ("train", "val", "test")
DISALLOWED_TAGS = {"script", "foreignObject"}


@dataclass
class Record:
    id: str
    prompt: str
    svg: str
    source: str
    license: str
    category: str
    complexity_level: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bootstrap SVG dataset")
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--rejections-path", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_record(raw: dict[str, Any], fallback_id: str) -> Record:
    return Record(
        id=normalize_text(raw.get("id")) or fallback_id,
        prompt=normalize_text(raw.get("prompt") or raw.get("description")),
        svg=normalize_text(raw.get("svg")),
        source=normalize_text(raw.get("source")) or "unknown",
        license=normalize_text(raw.get("license")) or "unknown",
        category=normalize_text(raw.get("category")) or "unknown",
        complexity_level=normalize_text(raw.get("complexity_level")) or "unknown",
    )


def has_required_fields(record: Record) -> bool:
    return bool(record.id and record.prompt and record.svg)


def extract_tag_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def validate_svg(svg: str) -> tuple[bool, str]:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return False, "xml_parse_failure"

    if extract_tag_name(root.tag) != "svg":
        return False, "missing_svg_root"

    for element in root.iter():
        tag = extract_tag_name(element.tag)
        if tag in DISALLOWED_TAGS:
            return False, f"disallowed_tag:{tag}"

    if len(list(root)) == 0:
        return False, "empty_svg"

    return True, "ok"


def dedupe_key(record: Record) -> str:
    normalized_prompt = " ".join(record.prompt.lower().split())
    payload = f"{normalized_prompt}\n{record.svg.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * 0.90)
    n_val = int(total * 0.05)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return {"train": train, "val": val, "test": test}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_report(
    path: Path,
    total_raw: int,
    total_kept: int,
    rejection_counts: dict[str, int],
    split_counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bootstrap Data Report",
        "",
        f"- Raw samples: {total_raw}",
        f"- Kept samples: {total_kept}",
        f"- Rejected samples: {total_raw - total_kept}",
        "",
        "## Split Counts",
    ]

    for split in ALLOWED_SPLITS:
        lines.append(f"- {split}: {split_counts.get(split, 0)}")

    lines.extend(["", "## Rejection Reasons"])

    if not rejection_counts:
        lines.append("- none")
    else:
        for reason, count in sorted(rejection_counts.items()):
            lines.append(f"- {reason}: {count}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_output_row(record: Record, split: str) -> dict[str, Any]:
    return {
        "id": record.id,
        "prompt": record.prompt,
        "svg": record.svg,
        "source": record.source,
        "license": record.license,
        "split": split,
        "quality_flags": [],
        "category": record.category,
        "complexity_level": record.complexity_level,
    }


def main() -> None:
    args = parse_args()

    raw_rows = load_jsonl(args.input_jsonl)
    seen: set[str] = set()
    cleaned_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    for index, raw in enumerate(raw_rows):
        fallback_id = f"sample_{index}"
        record = normalize_record(raw, fallback_id)

        if not has_required_fields(record):
            reason = "missing_required_fields"
            rejections.append({"id": record.id, "reason": reason})
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        is_valid, reason = validate_svg(record.svg)
        if not is_valid:
            rejections.append({"id": record.id, "reason": reason})
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        key = dedupe_key(record)
        if key in seen:
            reason = "duplicate_record"
            rejections.append({"id": record.id, "reason": reason})
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        seen.add(key)
        cleaned_rows.append(build_output_row(record, split=""))

    splits = split_records(cleaned_rows, seed=args.seed)

    split_counts: dict[str, int] = {}
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
        split_counts[split] = len(rows)
        write_jsonl(args.output_dir / f"bootstrap_{split}.jsonl", rows)

    write_jsonl(args.rejections_path, rejections)
    write_report(
        path=args.report_path,
        total_raw=len(raw_rows),
        total_kept=len(cleaned_rows),
        rejection_counts=rejection_counts,
        split_counts=split_counts,
    )


if __name__ == "__main__":
    main()
