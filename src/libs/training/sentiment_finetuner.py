"""Fine-tune a sequence-classification model on a Hugging Face dataset.

All datasets and models come from the Hugging Face Hub by explicit repo IDs and
revisions (AGENTS.md sections 25, 31). Heavy dependencies (torch, transformers,
datasets, numpy) are imported lazily inside ``finetune`` so this module imports
on machines without them (e.g. for CPU-only architecture tests); the actual run
happens on a GPU runtime.
"""
from __future__ import annotations

from pathlib import Path

from .training_reporter import TrainingReporter


class SentimentFinetuner:
    """Fine-tune a HF classifier on a small labeled dataset and save weights."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        dataset_id: str,
        dataset_revision: str,
        dataset_config: str,
        text_field: str,
        label_field: str,
        num_labels: int,
        train_samples: int,
        eval_samples: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        max_length: int,
        precision: str,
        seed: int,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._dataset_id = dataset_id
        self._dataset_revision = dataset_revision
        self._dataset_config = dataset_config
        self._text_field = text_field
        self._label_field = label_field
        self._num_labels = num_labels
        self._train_samples = train_samples
        self._eval_samples = eval_samples
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._max_length = max_length
        self._precision = precision
        self._seed = seed

    def finetune(self, output_dir: Path, reporter: TrainingReporter) -> dict:
        import numpy as np
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        output_dir = Path(output_dir)
        checkpoints = output_dir / "checkpoints"

        reporter.report_message(
            f"loading dataset {self._dataset_id}/{self._dataset_config}@{self._dataset_revision}"
        )
        dataset = load_dataset(
            self._dataset_id, self._dataset_config, revision=self._dataset_revision
        )
        train_split = dataset["train"].select(range(self._train_samples))
        eval_split = dataset["validation"].select(range(self._eval_samples))

        reporter.report_message(f"loading model {self._model_id}@{self._model_revision}")
        tokenizer = AutoTokenizer.from_pretrained(self._model_id, revision=self._model_revision)

        def tokenize(batch):
            return tokenizer(batch[self._text_field], truncation=True, max_length=self._max_length)

        keep = {"input_ids", "attention_mask", "labels"}

        def prepare(split):
            split = split.map(tokenize, batched=True)
            split = split.rename_column(self._label_field, "labels")
            return split.remove_columns([c for c in split.column_names if c not in keep])

        train_split = prepare(train_split)
        eval_split = prepare(eval_split)

        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_id, revision=self._model_revision, num_labels=self._num_labels
        )

        class _ReporterCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    numeric = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
                    reporter.report_step(int(state.global_step), numeric)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": float((preds == labels).mean())}

        use_fp16 = self._precision == "fp16" and torch.cuda.is_available()
        args = TrainingArguments(
            output_dir=str(checkpoints),
            num_train_epochs=self._num_epochs,
            per_device_train_batch_size=self._batch_size,
            per_device_eval_batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=use_fp16,
            seed=self._seed,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_split,
            eval_dataset=eval_split,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[_ReporterCallback()],
        )

        reporter.report_message("training started")
        trainer.train()
        metrics = trainer.evaluate()

        final_dir = checkpoints / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        reporter.report_message(f"training complete; weights at {final_dir}")

        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
