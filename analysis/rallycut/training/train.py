"""Training script for VideoMAE fine-tuning on beach volleyball."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

from rallycut.training.config import LABEL_MAP, TrainingConfig
from rallycut.training.dataset import BeachVolleyballDataset, create_data_collator
from rallycut.training.sampler import TrainingSample


def load_model_and_processor(
    config: TrainingConfig,
) -> tuple[VideoMAEForVideoClassification, VideoMAEImageProcessor]:
    """Load VideoMAE model and processor for fine-tuning.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, processor)
    """
    model_path = config.base_model_path

    if model_path.exists():
        # Load from local weights
        processor = VideoMAEImageProcessor.from_pretrained(str(model_path))
        model = VideoMAEForVideoClassification.from_pretrained(
            str(model_path),
            num_labels=3,
            id2label=LABEL_MAP,
            label2id={v: k for k, v in LABEL_MAP.items()},
            ignore_mismatched_sizes=True,
        )
    else:
        # Load from HuggingFace Hub
        processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=3,
            id2label=LABEL_MAP,
            label2id={v: k for k, v in LABEL_MAP.items()},
            ignore_mismatched_sizes=True,
        )

    return model, processor


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute metrics for evaluation.

    Args:
        eval_pred: Predictions from trainer

    Returns:
        Dictionary of metrics
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    acc_val = accuracy["accuracy"] if accuracy else 0.0
    f1_val = f1["f1"] if f1 else 0.0

    return {
        "accuracy": acc_val,
        "f1": f1_val,
    }


def train(
    train_samples: list[TrainingSample],
    val_samples: list[TrainingSample],
    video_paths: dict[str, Path],
    config: TrainingConfig | None = None,
    resume_from_checkpoint: str | None = None,
) -> Path:
    """Train VideoMAE model on beach volleyball data.

    Args:
        train_samples: Training samples
        val_samples: Validation samples
        video_paths: Mapping from video_id to local file path
        config: Training configuration
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Path to the trained model
    """
    if config is None:
        config = TrainingConfig()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and processor
    model, processor = load_model_and_processor(config)

    # Determine device
    if config.use_mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Training on device: {device}")
    model.to(device)  # type: ignore[arg-type]

    # Create datasets
    train_dataset = BeachVolleyballDataset(
        samples=train_samples,
        video_paths=video_paths,
        processor=processor,
        config=config,
        augment=True,
    )

    val_dataset = BeachVolleyballDataset(
        samples=val_samples,
        video_paths=video_paths,
        processor=processor,
        config=config,
        augment=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_strategy="epoch" if config.save_strategy == "epoch" else "steps",
        eval_steps=config.eval_steps if config.save_strategy == "steps" else None,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy == "steps" else None,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        dataloader_num_workers=config.dataloader_num_workers,
        fp16=False,  # MPS doesn't support FP16 well
        remove_unused_columns=False,
        label_smoothing_factor=config.label_smoothing,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=create_data_collator(),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            )
        ],
    )

    # Train
    print(f"Starting training with {len(train_samples)} train samples, {len(val_samples)} val")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save best model
    best_model_path = config.output_dir / "best"
    trainer.save_model(str(best_model_path))
    processor.save_pretrained(str(best_model_path))

    # Save training info
    info = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "config": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.num_epochs,
            "effective_batch_size": config.effective_batch_size,
        },
        "best_metrics": trainer.state.best_metric,
    }
    with open(config.output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Training complete! Model saved to: {best_model_path}")
    print(f"Best F1: {trainer.state.best_metric:.4f}")

    # Clean up datasets
    train_dataset.close()
    val_dataset.close()

    return best_model_path
