#!/usr/bin/env python3
"""Train a non-temporal baseline classifier on frozen VideoMAE features.

This tests whether the extracted features are discriminative for rally detection
without any temporal modeling. If AUC is near 0.5, features/labels are misaligned.
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.training import video_level_split
from rallycut.training.sampler import generate_sequence_labels
from rallycut.core.proxy import ProxyGenerator


def main():
    print("=" * 70)
    print("NON-TEMPORAL BASELINE CLASSIFIER")
    print("=" * 70)

    # Load data
    videos = load_evaluation_videos()
    cache = FeatureCache(cache_dir=Path("training_data/features"))
    stride = 48

    print(f"\nLoaded {len(videos)} videos")

    # Split videos
    train_videos, val_videos = video_level_split(videos, train_ratio=0.8, seed=42)
    print(f"Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")

    # Collect features and labels
    X_train, y_train = [], []
    X_val, y_val = [], []

    for video in train_videos:
        cached = cache.get(video.content_hash, stride)
        if cached is None:
            continue
        features, metadata = cached

        # Generate labels
        original_fps = video.fps or 30.0
        fps = 30.0 if original_fps > ProxyGenerator.FPS_NORMALIZE_THRESHOLD else original_fps
        labels = generate_sequence_labels(
            rallies=video.ground_truth_rallies,
            video_duration_ms=video.duration_ms or 0,
            fps=fps,
            stride=stride,
        )
        # Truncate to minimum length
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        X_train.extend(features)
        y_train.extend(labels)

    for video in val_videos:
        cached = cache.get(video.content_hash, stride)
        if cached is None:
            continue
        features, metadata = cached

        original_fps = video.fps or 30.0
        fps = 30.0 if original_fps > ProxyGenerator.FPS_NORMALIZE_THRESHOLD else original_fps
        labels = generate_sequence_labels(
            rallies=video.ground_truth_rallies,
            video_duration_ms=video.duration_ms or 0,
            fps=fps,
            stride=stride,
        )
        # Truncate to minimum length
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        X_val.extend(features)
        y_val.extend(labels)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}, labels: {y_train.shape}")
    print(f"  Val: {X_val.shape}, labels: {y_val.shape}")
    print(f"  Train class balance: NO_RALLY={1-y_train.mean():.1%}, RALLY={y_train.mean():.1%}")
    print(f"  Val class balance: NO_RALLY={1-y_val.mean():.1%}, RALLY={y_val.mean():.1%}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train Logistic Regression
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION")
    print("=" * 70)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train_scaled, y_train)

    # Predictions
    y_train_proba = lr.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = lr.predict_proba(X_val_scaled)[:, 1]
    y_val_pred = lr.predict(X_val_scaled)

    # Metrics
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_pr_auc = average_precision_score(y_val, y_val_proba)
    val_f1 = f1_score(y_val, y_val_pred)

    print(f"\nTrain ROC-AUC: {train_roc_auc:.4f}")
    print(f"Val ROC-AUC:   {val_roc_auc:.4f}")
    print(f"Val PR-AUC:    {val_pr_auc:.4f}")
    print(f"Val F1:        {val_f1:.4f}")

    print("\nConfusion Matrix (Val):")
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    print("\nClassification Report (Val):")
    print(classification_report(y_val, y_val_pred, target_names=["NO_RALLY", "RALLY"]))

    # Train MLP
    print("\n" + "=" * 70)
    print("MLP CLASSIFIER (2 hidden layers)")
    print("=" * 70)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    mlp.fit(X_train_scaled, y_train)

    # Predictions
    y_train_proba_mlp = mlp.predict_proba(X_train_scaled)[:, 1]
    y_val_proba_mlp = mlp.predict_proba(X_val_scaled)[:, 1]
    y_val_pred_mlp = mlp.predict(X_val_scaled)

    # Metrics
    train_roc_auc_mlp = roc_auc_score(y_train, y_train_proba_mlp)
    val_roc_auc_mlp = roc_auc_score(y_val, y_val_proba_mlp)
    val_pr_auc_mlp = average_precision_score(y_val, y_val_proba_mlp)
    val_f1_mlp = f1_score(y_val, y_val_pred_mlp)

    print(f"\nTrain ROC-AUC: {train_roc_auc_mlp:.4f}")
    print(f"Val ROC-AUC:   {val_roc_auc_mlp:.4f}")
    print(f"Val PR-AUC:    {val_pr_auc_mlp:.4f}")
    print(f"Val F1:        {val_f1_mlp:.4f}")

    print("\nConfusion Matrix (Val):")
    cm_mlp = confusion_matrix(y_val, y_val_pred_mlp)
    print(f"  TN={cm_mlp[0,0]:4d}  FP={cm_mlp[0,1]:4d}")
    print(f"  FN={cm_mlp[1,0]:4d}  TP={cm_mlp[1,1]:4d}")

    print("\nClassification Report (Val):")
    print(classification_report(y_val, y_val_pred_mlp, target_names=["NO_RALLY", "RALLY"]))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nLogistic Regression: ROC-AUC={val_roc_auc:.4f}, PR-AUC={val_pr_auc:.4f}, F1={val_f1:.4f}")
    print(f"MLP Classifier:      ROC-AUC={val_roc_auc_mlp:.4f}, PR-AUC={val_pr_auc_mlp:.4f}, F1={val_f1_mlp:.4f}")

    if val_roc_auc < 0.6:
        print("\n[WARNING] ROC-AUC < 0.6 suggests features are not discriminative!")
        print("          Check feature extraction or label alignment.")
    elif val_roc_auc < 0.7:
        print("\n[INFO] ROC-AUC 0.6-0.7 suggests features have weak discriminative power.")
        print("       Temporal modeling may help, but check for data issues first.")
    else:
        print("\n[OK] ROC-AUC >= 0.7 suggests features are discriminative.")
        print("     Temporal modeling should be able to improve on this baseline.")


if __name__ == "__main__":
    main()
