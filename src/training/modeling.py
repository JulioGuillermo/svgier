"""Model and tokenizer builders for LoRA-based causal LM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.training.config import AdapterConfig, RunConfig


@dataclass
class TokenizerBuilder:
    """Creates and adjusts tokenizer settings."""

    run_config: RunConfig

    def build(self) -> PreTrainedTokenizerBase:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.run_config.model_name,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


@dataclass
class ModelBuilder:
    """Loads base model and applies LoRA adapter."""

    run_config: RunConfig
    adapter_config: AdapterConfig

    def build_base_model(self) -> PreTrainedModel:
        model_kwargs: dict[str, Any] = {}
        if self.run_config.bf16:
            model_kwargs["dtype"] = torch.bfloat16

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.run_config.model_name,
            **model_kwargs,
        )

        if self.run_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model

    def apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        lora_cfg: LoraConfig = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.adapter_config.r,
            lora_alpha=self.adapter_config.alpha,
            lora_dropout=self.adapter_config.dropout,
            target_modules=list(self.adapter_config.target_modules),
            inference_mode=False,
        )
        peft_model: PreTrainedModel = get_peft_model(model, lora_cfg)
        return peft_model
