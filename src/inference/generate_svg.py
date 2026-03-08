"""Generate SVG text from a prompt with optional LoRA checkpoint loading."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from src.common.prompting import PromptFormatter


@dataclass
class GenerateConfig:
    """Runtime config for one-off SVG generation."""

    model_name: str
    output_file: Path
    prompt: str
    checkpoint: Path | None
    checkpoints_dir: Path
    max_new_tokens: int
    temperature: float
    top_p: float


class DeviceResolver:
    """Resolves the best available torch device."""

    @staticmethod
    def detect() -> str:
        if torch.cuda.is_available():
            return "cuda"
        xpu_backend = getattr(torch, "xpu", None)
        if xpu_backend is not None and xpu_backend.is_available():
            return "xpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def dtype_for_device(device: str) -> torch.dtype:
        if device == "cuda":
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported) and is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if device in {"xpu", "mps"}:
            return torch.float16
        return torch.float32


class CheckpointResolver:
    """Resolves explicit checkpoint path or latest checkpoint in a directory."""

    _checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")

    def resolve(self, explicit_path: Path | None, checkpoints_dir: Path) -> Path | None:
        if explicit_path is not None:
            if not explicit_path.exists():
                raise FileNotFoundError(f"Checkpoint path not found: {explicit_path}")
            return explicit_path

        if not checkpoints_dir.exists():
            return None

        candidates: list[tuple[int, Path]] = []
        for item in checkpoints_dir.iterdir():
            if not item.is_dir():
                continue
            match = self._checkpoint_pattern.match(item.name)
            if match is None:
                continue
            step = int(match.group(1))
            candidates.append((step, item))

        if not candidates:
            return None

        candidates.sort(key=lambda pair: pair[0])
        return candidates[-1][1]


class ModelLoader:
    """Loads tokenizer and model, optionally applying a LoRA adapter checkpoint."""

    def __init__(self, config: GenerateConfig, device: str, dtype: torch.dtype) -> None:
        self.config: GenerateConfig = config
        self.device: str = device
        self.dtype: torch.dtype = dtype

    def _load_tokenizer(self, checkpoint: Path | None):
        if checkpoint is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(checkpoint), use_fast=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
            except Exception:
                pass

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_base_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=self.dtype,
        )
        return model

    def load(self, checkpoint: Path | None):
        tokenizer = self._load_tokenizer(checkpoint=checkpoint)

        if checkpoint is not None and (checkpoint / "adapter_config.json").exists():
            base_model = self._load_base_model()
            model = PeftModel.from_pretrained(base_model, str(checkpoint))
        elif checkpoint is not None:
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint), dtype=self.dtype
            )
        else:
            model = self._load_base_model()

        model = model.to(self.device)
        model.eval()
        return tokenizer, model


class SvgStopCriteria(StoppingCriteria):
    """Stops generation when the closing SVG tag is present."""

    def __init__(self, tokenizer: Any, prompt_token_count: int) -> None:
        self.tokenizer: Any = tokenizer
        self.prompt_token_count: int = prompt_token_count

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any
    ) -> bool:
        generated_ids = input_ids[0, self.prompt_token_count :]
        decoded_text: str = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return "</svg>" in decoded_text.lower()


class SvgOutputExtractor:
    """Extracts only the first valid SVG block from generated text."""

    _pattern = re.compile(r"(<svg\\b[^>]*>.*?</svg>)", flags=re.IGNORECASE | re.DOTALL)

    @classmethod
    def extract(cls, text: str) -> str:
        match = cls._pattern.search(text)
        if match is None:
            raise ValueError(
                "No complete <svg>...</svg> block found in generated output."
            )
        return match.group(1).strip()


class SvgGenerator:
    """Streams generation to console and saves final output to file."""

    def __init__(self, config: GenerateConfig) -> None:
        self.config: GenerateConfig = config
        self.device: str = DeviceResolver.detect()
        self.dtype: torch.dtype = DeviceResolver.dtype_for_device(self.device)
        self.checkpoint_resolver: CheckpointResolver = CheckpointResolver()
        self.prompt_formatter: PromptFormatter = PromptFormatter()

    def _build_inputs(self, tokenizer):
        prompt_text: str = self.prompt_formatter.format_sample(
            prompt=self.config.prompt,
            svg="",
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in encoded.items()}

    @staticmethod
    def _resolve_max_new_tokens(value: int) -> int:
        if value <= 0:
            return 65536
        return value

    def run(self) -> None:
        checkpoint = self.checkpoint_resolver.resolve(
            explicit_path=self.config.checkpoint,
            checkpoints_dir=self.config.checkpoints_dir,
        )

        if checkpoint is not None:
            print(f"[generate] checkpoint={checkpoint}")
        else:
            print("[generate] checkpoint=none (using base model)")

        print(f"[generate] device={self.device} dtype={self.dtype}")

        loader = ModelLoader(config=self.config, device=self.device, dtype=self.dtype)
        tokenizer, model = loader.load(checkpoint=checkpoint)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        inputs = self._build_inputs(tokenizer=tokenizer)
        do_sample: bool = self.config.temperature > 0.0
        prompt_token_count: int = int(inputs["input_ids"].shape[-1])
        stop_criteria = StoppingCriteriaList(
            [
                SvgStopCriteria(
                    tokenizer=tokenizer, prompt_token_count=prompt_token_count
                )
            ]
        )
        max_new_tokens: int = self._resolve_max_new_tokens(self.config.max_new_tokens)

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": do_sample,
            "streamer": streamer,
            "stopping_criteria": stop_criteria,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": None,
        }

        generation_thread = Thread(target=model.generate, kwargs=generation_kwargs)
        generation_thread.start()

        chunks: list[str] = []
        for piece in streamer:
            print(piece, end="", flush=True)
            chunks.append(piece)

        generation_thread.join()
        print()

        output_text: str = "".join(chunks).strip()
        svg_text: str = SvgOutputExtractor.extract(output_text)
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_file.write_text(svg_text + "\n", encoding="utf-8")
        print(f"[generate] saved={self.config.output_file}")


class ConfigFactory:
    """Builds GenerateConfig from CLI arguments."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Generate SVG from a single prompt"
        )
        parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
        parser.add_argument("--checkpoint", type=Path, default=None)
        parser.add_argument(
            "--checkpoints-dir",
            type=Path,
            default=Path("checkpoints/qwen35_0_8b_lora_bootstrap"),
        )
        parser.add_argument("--output-file", type=Path, required=True)
        parser.add_argument("--max-new-tokens", type=int, default=0)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--top-p", type=float, default=0.9)
        parser.add_argument("prompt", nargs=argparse.REMAINDER)
        return parser.parse_args()

    @staticmethod
    def from_args(args: argparse.Namespace) -> GenerateConfig:
        raw_prompt: str = " ".join(args.prompt).strip()
        if raw_prompt.startswith("--"):
            raw_prompt = raw_prompt[2:].strip()
        if not raw_prompt:
            raise ValueError("Prompt is required at the end of the command.")

        return GenerateConfig(
            model_name=args.model_name,
            output_file=args.output_file,
            prompt=raw_prompt,
            checkpoint=args.checkpoint,
            checkpoints_dir=args.checkpoints_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


def main() -> None:
    args = ConfigFactory.parse_args()
    config = ConfigFactory.from_args(args)
    generator = SvgGenerator(config=config)
    generator.run()


if __name__ == "__main__":
    main()
