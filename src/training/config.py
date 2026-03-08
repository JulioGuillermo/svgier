"""Typed configuration objects for LoRA training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Input dataset paths used for SFT."""

    train_jsonl: Path = Path("data/processed/bootstrap_train.jsonl")
    val_jsonl: Path = Path("data/processed/bootstrap_val.jsonl")


@dataclass
class AdapterConfig:
    """LoRA adapter settings."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class RunConfig:
    """Runtime settings for the training execution."""

    model_name: str = "Qwen/Qwen3.5-0.8B"
    output_dir: Path = Path("checkpoints/qwen35_0_8b_lora_bootstrap")
    max_length: int = 1024
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    gradient_checkpointing: bool = True


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    run: RunConfig = field(default_factory=RunConfig)
