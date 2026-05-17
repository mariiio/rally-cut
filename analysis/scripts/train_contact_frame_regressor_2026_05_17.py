#!/usr/bin/env python3
"""Phase 2: train the contact-frame-refinement regressor.

Input: training_data.csv from extract_contact_frame_training_data_2026_05_17.py
Output: trained model at weights/contact_frame_regressor/best_model.joblib + LOO CV report

Strategy:
  - LOO CV by video (leave-one-video-out) to honestly measure generalization
  - LightGBM regressor (handles missing-pose sentinel values gracefully)
  - Loss: Huber (robust to outliers in the ±15 frame range)
  - Report: MAE, RMSE, within-±5 recovery, regression on already-correct cases
  - Train final model on all data
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUT_DIR = Path("reports/contact_frame_regressor_2026_05_17")
WEIGHTS_DIR = Path("weights/contact_frame_regressor")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names — everything except metadata."""
    metadata = {"video", "rally_id", "gt_frame", "gt_action", "gt_tid",
                "cand_frame", "target_offset"}
    return [c for c in df.columns if c not in metadata]


def evaluate_loo(df: pd.DataFrame) -> dict:
    """Leave-one-video-out CV. Returns metrics dict."""
    from sklearn.ensemble import GradientBoostingRegressor

    feat_cols = feature_columns(df)
    videos = sorted(df["video"].unique())
    print(f"\nLOO CV across {len(videos)} videos, {len(feat_cols)} features", flush=True)

    all_preds = []
    all_targets = []
    per_video_metrics = []

    for video in videos:
        train_df = df[df["video"] != video]
        val_df = df[df["video"] == video]
        if len(val_df) == 0:
            continue

        X_train = train_df[feat_cols].values
        y_train = train_df["target_offset"].values
        X_val = val_df[feat_cols].values
        y_val = val_df["target_offset"].values

        # Sentinel -1.0 values in feature matrix represent "missing pose"
        # — sklearn GBR handles them as just another numeric value, which is
        # functionally fine since the tree splits will isolate them.
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=10,
            loss="huber",
            alpha=0.9,  # huber threshold (residual cutoff)
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        all_preds.extend(preds)
        all_targets.extend(y_val)

        mae = float(np.mean(np.abs(preds - y_val)))
        per_video_metrics.append({
            "video": video,
            "n": len(y_val),
            "mae_raw": float(np.mean(np.abs(y_val))),  # do-nothing baseline
            "mae_pred": mae,
            "improvement": float(np.mean(np.abs(y_val)) - mae),
        })

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Snap recommendation: round prediction to nearest integer frame
    snap_preds = np.round(all_preds).astype(int)

    # Compute "after snap" delta from GT — pre-snap delta is target_offset,
    # post-snap delta is target_offset - snap_pred (we shift by snap_pred)
    pre_delta = np.abs(all_targets)
    post_delta = np.abs(all_targets - snap_preds)

    pre_within_5 = int(np.sum(pre_delta <= 5))
    post_within_5 = int(np.sum(post_delta <= 5))
    pre_within_2 = int(np.sum(pre_delta <= 2))
    post_within_2 = int(np.sum(post_delta <= 2))

    # Count GAINED vs LOST (cases that crossed the ±5 boundary)
    gained = int(np.sum((pre_delta > 5) & (post_delta <= 5)))
    lost = int(np.sum((pre_delta <= 5) & (post_delta > 5)))

    return {
        "n_total": len(all_targets),
        "mae_raw": float(np.mean(pre_delta)),       # do-nothing
        "mae_pred": float(np.mean(np.abs(all_targets - all_preds))),  # continuous prediction
        "mae_snap": float(np.mean(post_delta)),     # after applying rounded snap
        "rmse_pred": float(np.sqrt(np.mean((all_targets - all_preds) ** 2))),
        "pre_within_5": pre_within_5,
        "post_within_5": post_within_5,
        "pre_within_2": pre_within_2,
        "post_within_2": post_within_2,
        "gained_at_5": gained,
        "lost_at_5": lost,
        "net_at_5": gained - lost,
        "per_video": per_video_metrics,
    }


def main() -> int:
    csv_path = OUT_DIR / "training_data.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path}; run extract_contact_frame_training_data_2026_05_17.py first")
        return 1
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} training rows from {csv_path}", flush=True)
    print(f"Target distribution: mean={df['target_offset'].mean():.2f}, "
          f"std={df['target_offset'].std():.2f}, "
          f"range=[{df['target_offset'].min()}, {df['target_offset'].max()}]")

    metrics = evaluate_loo(df)

    print(f"\n{'='*60}")
    print(f"LOO CV RESULTS ({metrics['n_total']} examples)")
    print(f"{'='*60}")
    print(f"Do-nothing baseline   MAE: {metrics['mae_raw']:.3f}")
    print(f"Model prediction      MAE: {metrics['mae_pred']:.3f}  ({metrics['mae_raw']-metrics['mae_pred']:+.3f})")
    print(f"After rounded snap    MAE: {metrics['mae_snap']:.3f}  ({metrics['mae_raw']-metrics['mae_snap']:+.3f})")
    print(f"RMSE (model):              {metrics['rmse_pred']:.3f}")
    print()
    print(f"Within ±5 of GT:  {metrics['pre_within_5']} → {metrics['post_within_5']}  "
          f"(GAINED {metrics['gained_at_5']}, LOST {metrics['lost_at_5']}, NET {metrics['net_at_5']:+d})")
    print(f"Within ±2 of GT:  {metrics['pre_within_2']} → {metrics['post_within_2']}")

    # Per-video table
    print(f"\nPer-video LOO improvement (top regressors at bottom):")
    print(f"  {'video':<8s}{'n':>5s}{'MAE_raw':>10s}{'MAE_pred':>10s}{'Δ':>9s}")
    sorted_videos = sorted(metrics["per_video"], key=lambda v: -v["improvement"])
    for v in sorted_videos:
        marker = "▲" if v["improvement"] > 0.1 else ("▼" if v["improvement"] < -0.1 else " ")
        print(f"  {v['video']:<8s}{v['n']:>5d}{v['mae_raw']:>10.3f}{v['mae_pred']:>10.3f}  "
              f"{v['improvement']:+8.3f} {marker}")

    # Train final model on all data
    print(f"\nTraining final model on all {len(df)} examples...")
    from sklearn.ensemble import GradientBoostingRegressor
    feat_cols = feature_columns(df)
    X = df[feat_cols].values
    y = df["target_offset"].values
    final_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        min_samples_leaf=10, loss="huber", alpha=0.9, random_state=42,
    )
    final_model.fit(X, y)

    model_path = WEIGHTS_DIR / "best_model.joblib"
    joblib.dump({
        "model": final_model,
        "feature_names": feat_cols,
        "loo_metrics": metrics,
        "training_date": "2026-05-17",
        "n_training_examples": len(df),
    }, model_path)
    print(f"Saved final model to {model_path}")

    # Feature importance
    print(f"\nTop 15 features by importance (Friedman gain):")
    importances = sorted(
        zip(feat_cols, final_model.feature_importances_, strict=True),
        key=lambda x: -x[1],
    )
    for name, imp in importances[:15]:
        print(f"  {name:<20s}{imp:>12.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
