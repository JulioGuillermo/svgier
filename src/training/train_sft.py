"""SFT training entry point for Qwen3.5-0.8B LoRA viability runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.training.config import AdapterConfig, DataConfig, RunConfig, TrainConfig
from src.training.data import JsonlDatasetLoader, PromptFormatter
from src.training.modeling import ModelBuilder, TokenizerBuilder


class TokenizationPipeline:
    """Tokenizes text datasets for causal language model training."""

    def __init__(self, tokenizer_builder: TokenizerBuilder, max_length: int) -> None:
        self.tokenizer_builder: TokenizerBuilder = tokenizer_builder
        self.max_length: int = max_length
        self.tokenizer = tokenizer_builder.build()

    def encode_batch(self, batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded: dict[str, Any] = self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded


class TrainingOrchestrator:
    """Coordinates loading data, building model, and running Trainer."""

    def __init__(self, config: TrainConfig) -> None:
        self.config: TrainConfig = config
        self.formatter: PromptFormatter = PromptFormatter()
        self.dataset_loader: JsonlDatasetLoader = JsonlDatasetLoader()

    def _load_datasets(self) -> tuple[Dataset, Dataset]:
        train_rows = self.dataset_loader.load_jsonl(self.config.data.train_jsonl)
        val_rows = self.dataset_loader.load_jsonl(self.config.data.val_jsonl)

        train_ds: Dataset = self.dataset_loader.to_text_dataset(train_rows, self.formatter)
        val_ds: Dataset = self.dataset_loader.to_text_dataset(val_rows, self.formatter)
        return train_ds, val_ds

    def _training_arguments(self) -> TrainingArguments:
        run: RunConfig = self.config.run
        return TrainingArguments(
            output_dir=str(run.output_dir),
            learning_rate=run.learning_rate,
            per_device_train_batch_size=run.per_device_train_batch_size,
            per_device_eval_batch_size=run.per_device_eval_batch_size,
            gradient_accumulation_steps=run.gradient_accumulation_steps,
            num_train_epochs=run.num_train_epochs,
            warmup_ratio=run.warmup_ratio,
            logging_steps=run.logging_steps,
            save_steps=run.save_steps,
            eval_steps=run.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            report_to="none",
            save_total_limit=run.save_total_limit,
            bf16=run.bf16,
            fp16=run.fp16,
            seed=run.seed,
            remove_unused_columns=False,
            gradient_checkpointing=run.gradient_checkpointing,
            do_train=True,
            do_eval=True,
        )

    def run(self) -> None:
        train_ds, val_ds = self._load_datasets()

        tokenizer_builder = TokenizerBuilder(run_config=self.config.run)
        token_pipeline = TokenizationPipeline(
            tokenizer_builder=tokenizer_builder,
            max_length=self.config.run.max_length,
        )

        train_tok: Dataset = train_ds.map(
            token_pipeline.encode_batch,
            batched=True,
            remove_columns=["text"],
        )
        val_tok: Dataset = val_ds.map(
            token_pipeline.encode_batch,
            batched=True,
            remove_columns=["text"],
        )

        model_builder = ModelBuilder(
            run_config=self.config.run,
            adapter_config=self.config.adapter,
        )
        model = model_builder.build_base_model()
        model = model_builder.apply_lora(model)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=token_pipeline.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=self._training_arguments(),
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator,
            processing_class=token_pipeline.tokenizer,
        )

        trainer.train()
        trainer.save_model()
        token_pipeline.tokenizer.save_pretrained(str(self.config.run.output_dir))


class ConfigFactory:
    """Builds TrainConfig from command line arguments."""

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train Qwen3.5-0.8B with LoRA on bootstrap dataset")
        parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-0.8B")
        parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/bootstrap_train.jsonl"))
        parser.add_argument("--val-jsonl", type=Path, default=Path("data/processed/bootstrap_val.jsonl"))
        parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen35_0_8b_lora_bootstrap"))
        parser.add_argument("--max-length", type=int, default=1024)
        parser.add_argument("--learning-rate", type=float, default=2e-4)
        parser.add_argument("--train-batch-size", type=int, default=2)
        parser.add_argument("--eval-batch-size", type=int, default=2)
        parser.add_argument("--grad-accum", type=int, default=8)
        parser.add_argument("--epochs", type=int, default=2)
        parser.add_argument("--bf16", action="store_true")
        parser.add_argument("--fp16", action="store_true")
        return parser.parse_args()

    @staticmethod
    def from_args(args: argparse.Namespace) -> TrainConfig:
        data_cfg = DataConfig(train_jsonl=args.train_jsonl, val_jsonl=args.val_jsonl)
        adapter_cfg = AdapterConfig()
        run_cfg = RunConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            bf16=args.bf16,
            fp16=args.fp16,
        )
        return TrainConfig(data=data_cfg, adapter=adapter_cfg, run=run_cfg)


def main() -> None:
    args = ConfigFactory.parse_args()
    config = ConfigFactory.from_args(args)

    orchestrator = TrainingOrchestrator(config=config)
    orchestrator.run()


if __name__ == "__main__":
    main()
