"""Prompt formatting helpers shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptFormatter:
    """Builds a single text sequence from prompt and SVG."""

    system_instruction: str = (
        "Generate valid and coherent SVG that matches the user request."
    )

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
