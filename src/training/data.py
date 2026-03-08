"""Dataset loading and prompt formatting utilities for SVG SFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset


@dataclass
class PromptFormatter:
    """Builds a single training text sequence from prompt and SVG."""

    system_instruction: str = "Generate valid and coherent SVG that matches the user request."

    def format_sample(self, prompt: str, svg: str) -> str:
        safe_prompt: str = prompt.strip()
        safe_svg: str = svg.strip()
        return (
            "<|system|>\n"
            f"{self.system_instruction}\n"
            "<|user|>\n"
            f"{safe_prompt}\n"
            "<|assistant|>\n"
            f"{safe_svg}"
        )


class JsonlDatasetLoader:
    """Loads local JSONL files and converts them into HF Dataset objects."""

    def load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text: str = line.strip()
                if not text:
                    continue
                row: dict[str, Any] = json.loads(text)
                rows.append(row)
        return rows

    def to_text_dataset(self, rows: list[dict[str, Any]], formatter: PromptFormatter) -> Dataset:
        text_rows: list[dict[str, str]] = []
        for row in rows:
            prompt: str = str(row.get("prompt", "")).strip()
            svg: str = str(row.get("svg", "")).strip()
            if not prompt or not svg:
                continue
            text: str = formatter.format_sample(prompt=prompt, svg=svg)
            text_rows.append({"text": text})
        return Dataset.from_list(text_rows)
