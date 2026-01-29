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
from rallycut.training.dataset import (
    BeachVolleyballDataset,
    PreExtractedFramesDataset,
    create_data_collator,
)
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

    Uses macro F1 as primary metric to avoid majority class (NO_PLAY) inflating scores.
    Also reports per-class F1 for debugging class imbalance issues.

    Args:
        eval_pred: Predictions from trainer

    Returns:
        Dictionary of metrics including macro F1, weighted F1, and per-class F1
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1_per_class = f1_metric.compute(predictions=predictions, references=labels, average=None)

    acc_val = accuracy["accuracy"] if accuracy else 0.0
    f1_macro_val = f1_macro["f1"] if f1_macro else 0.0
    f1_weighted_val = f1_weighted["f1"] if f1_weighted else 0.0

    # Per-class F1: 0=NO_PLAY, 1=PLAY, 2=SERVICE
    f1_classes = f1_per_class["f1"] if f1_per_class else [0.0, 0.0, 0.0]

    return {
        "accuracy": acc_val,
        "f1": f1_macro_val,  # Primary metric: macro F1 (treats all classes equally)
        "f1_weighted": f1_weighted_val,  # Keep for reference
        "f1_no_play": float(f1_classes[0]) if len(f1_classes) > 0 else 0.0,
        "f1_play": float(f1_classes[1]) if len(f1_classes) > 1 else 0.0,
        "f1_service": float(f1_classes[2]) if len(f1_classes) > 2 else 0.0,
    }


def train(
    train_samples: list[TrainingSample],
    val_samples: list[TrainingSample],
    video_paths: dict[str, Path] | None = None,
    config: TrainingConfig | None = None,
    resume_from_checkpoint: str | None = None,
    train_frames_dir: Path | None = None,
    val_frames_dir: Path | None = None,
) -> Path:
    """Train VideoMAE model on beach volleyball data.

    Supports two modes:
    1. Video mode: Pass video_paths to load frames from video files on-the-fly
    2. Pre-extracted mode: Pass train_frames_dir and val_frames_dir with .npy files
       (10x faster, enables multiprocessing)

    Args:
        train_samples: Training samples (used for labels in pre-extracted mode)
        val_samples: Validation samples (used for labels in pre-extracted mode)
        video_paths: Mapping from video_id to local file path (video mode)
        config: Training configuration
        resume_from_checkpoint: Path to checkpoint to resume from
        train_frames_dir: Directory with pre-extracted train frame .npy files
        val_frames_dir: Directory with pre-extracted val frame .npy files

    Returns:
        Path to the trained model
    """
    if config is None:
        config = TrainingConfig()

    # Determine mode
    use_preextracted = train_frames_dir is not None and val_frames_dir is not None
    if not use_preextracted and video_paths is None:
        raise ValueError("Must provide either video_paths or (train_frames_dir, val_frames_dir)")

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
    train_dataset: BeachVolleyballDataset | PreExtractedFramesDataset
    val_dataset: BeachVolleyballDataset | PreExtractedFramesDataset

    if use_preextracted:
        print("Using pre-extracted frames (multiprocessing enabled)")
        train_labels = [s.label for s in train_samples]
        val_labels = [s.label for s in val_samples]
        assert train_frames_dir is not None
        assert val_frames_dir is not None

        train_dataset = PreExtractedFramesDataset(
            frames_dir=train_frames_dir,
            labels=train_labels,
            processor=processor,
            config=config,
            augment=True,
        )

        val_dataset = PreExtractedFramesDataset(
            frames_dir=val_frames_dir,
            labels=val_labels,
            processor=processor,
            config=config,
            augment=False,
        )
    else:
        print("Using video loading (sequential, slower)")
        assert video_paths is not None
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
        max_grad_norm=config.max_grad_norm,  # Clip gradients to prevent explosion
        logging_steps=config.logging_steps,
        eval_strategy="epoch" if config.save_strategy == "epoch" else "steps",
        eval_steps=config.eval_steps if config.save_strategy == "steps" else None,
        save_strategy=config.save_strategy,
        save_steps=float(config.save_steps) if config.save_strategy == "steps" else 500.0,
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

    # Train with proper cleanup
    print(f"Starting training with {len(train_samples)} train samples, {len(val_samples)} val")
    try:
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

        return best_model_path
    finally:
        # Clean up datasets (releases video file handles)
        train_dataset.close()
        val_dataset.close()
