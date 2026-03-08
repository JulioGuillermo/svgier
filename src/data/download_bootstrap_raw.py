"""Downloads and builds a bootstrap raw JSONL dataset from public HF sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


@dataclass
class DownloadConfig:
    """Config for raw bootstrap download and merge."""

    output_jsonl: Path
    text2svg_limit: int
    instruct_svg_limit: int
    svg_emoji_limit: int


class RowBuilder:
    """Builds normalized raw rows from dataset-specific records."""

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value or "").strip()

    def from_text2svg(self, index: int, row: dict[str, Any]) -> dict[str, str] | None:
        prompt: str = self._clean(
            row.get("caption_llava") or row.get("caption_cogvlm") or row.get("caption_blip2")
        )
        svg: str = self._clean(row.get("Svg"))
        if not prompt or not svg:
            return None
        return {
            "id": f"text2svg_{index}",
            "prompt": prompt,
            "svg": svg,
            "source": "starvector/text2svg-stack",
            "license": "unknown",
            "category": "mixed",
            "complexity_level": "unknown",
        }

    def from_instruct_svg(self, index: int, row: dict[str, Any]) -> dict[str, str] | None:
        prompt: str = self._clean(row.get("input") or row.get("description_1") or row.get("description_0"))
        svg: str = self._clean(row.get("output"))
        if not prompt or not svg:
            return None
        return {
            "id": f"instruct_svg_{index}",
            "prompt": prompt,
            "svg": svg,
            "source": "uwunion/instruct_svg",
            "license": "unknown",
            "category": "illustration",
            "complexity_level": "medium",
        }

    def from_svg_emoji(self, index: int, row: dict[str, Any]) -> dict[str, str] | None:
        filename: str = self._clean(row.get("Filename"))
        prompt: str = filename.replace(".svg", "").replace("_", " ").replace("-", " ").strip()
        svg: str = self._clean(row.get("Svg"))
        if not prompt or not svg:
            return None
        return {
            "id": f"svg_emoji_{index}",
            "prompt": prompt,
            "svg": svg,
            "source": "ServiceNow/svg-emoji",
            "license": "unknown",
            "category": "icon",
            "complexity_level": "low",
        }


class RawBootstrapDownloader:
    """Downloads, transforms, and writes merged raw bootstrap records."""

    def __init__(self, config: DownloadConfig) -> None:
        self.config: DownloadConfig = config
        self.builder: RowBuilder = RowBuilder()

    def _iter_stream(self, dataset_name: str, limit: int) -> Iterable[dict[str, Any]]:
        stream = load_dataset(dataset_name, split="train", streaming=True)
        return islice(stream, limit)

    def _collect_text2svg(self) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for index, row in enumerate(self._iter_stream("starvector/text2svg-stack", self.config.text2svg_limit)):
            item = self.builder.from_text2svg(index=index, row=row)
            if item is not None:
                output.append(item)
        return output

    def _collect_instruct_svg(self) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for index, row in enumerate(self._iter_stream("uwunion/instruct_svg", self.config.instruct_svg_limit)):
            item = self.builder.from_instruct_svg(index=index, row=row)
            if item is not None:
                output.append(item)
        return output

    def _collect_svg_emoji(self) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for index, row in enumerate(self._iter_stream("ServiceNow/svg-emoji", self.config.svg_emoji_limit)):
            item = self.builder.from_svg_emoji(index=index, row=row)
            if item is not None:
                output.append(item)
        return output

    def download(self) -> list[dict[str, str]]:
        text2svg_rows: list[dict[str, str]] = self._collect_text2svg()
        instruct_rows: list[dict[str, str]] = self._collect_instruct_svg()
        emoji_rows: list[dict[str, str]] = self._collect_svg_emoji()
        rows: list[dict[str, str]] = text2svg_rows + instruct_rows + emoji_rows

        self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        print(
            "downloaded_raw="
            f"{len(rows)} text2svg={len(text2svg_rows)} "
            f"instruct_svg={len(instruct_rows)} svg_emoji={len(emoji_rows)} "
            f"output={self.config.output_jsonl}"
        )
        return rows


class ConfigFactory:
    """Builds CLI config for raw download."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Download bootstrap raw SVG dataset")
        parser.add_argument("--output-jsonl", type=Path, default=Path("data/raw/bootstrap_input.jsonl"))
        parser.add_argument("--text2svg-limit", type=int, default=5000)
        parser.add_argument("--instruct-svg-limit", type=int, default=3500)
        parser.add_argument("--svg-emoji-limit", type=int, default=2500)
        return parser.parse_args()

    @staticmethod
    def from_args(args: argparse.Namespace) -> DownloadConfig:
        return DownloadConfig(
            output_jsonl=args.output_jsonl,
            text2svg_limit=args.text2svg_limit,
            instruct_svg_limit=args.instruct_svg_limit,
            svg_emoji_limit=args.svg_emoji_limit,
        )


def main() -> None:
    args = ConfigFactory.parse_args()
    config: DownloadConfig = ConfigFactory.from_args(args)
    downloader = RawBootstrapDownloader(config=config)
    downloader.download()


if __name__ == "__main__":
    main()
