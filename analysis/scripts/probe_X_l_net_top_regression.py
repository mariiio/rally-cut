# ruff: noqa: N803, N806, F841
"""Probe X-L: learned net_top_y regression from calibration corners.

Probe X-K showed B (hand-tuned calibration projection with constant 0.30)
was uniformly biased +0.21. The hypothesis is that the geometric form is
correct but the constant is wrong AND camera-angle-dependent. So: fit
the coefficients on the 77 GT samples.

Features (all derived from the 4 calibration corners, normalized 0-1):

  midline_y         : perspective net midline y at court center
                      (diagonal intersection y).
  near_y            : mean of the two near-baseline corner ys.
  far_y             : mean of the two far-baseline corner ys.
  court_depth_y     : near_y - far_y (image-y span of court).
  midline_width_x   : (right - left) x of midline endpoints (apparent
                      court width at the midline).
  near_width_x      : near baseline width in image.
  far_width_x       : far baseline width in image.
  trapezoid_aspect  : (near_width_x + far_width_x) / 2 / court_depth_y
                      (perspective compression proxy).

Train several models with leave-one-video-out cross-validation:
  M0: just `midline_y` (i.e., assume net top is the midline projection)
  M1: midline_y + linear(court_depth_y) — one learned offset coefficient
  M2: midline_y + court_depth_y + far_y (two coefficients)
  M3: full linear regression on all 8 features
  M4: ridge regression on M3 features (regularised; defends against
      overfitting with n=77 samples)

Score each by LOO-CV |Δ| vs GT. Pick the simplest model whose median
|Δ| is within ε of the best.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass

import numpy as np
import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"


@dataclass
class Sample:
    name: str
    fps: float
    gt: float
    features: dict[str, float]


def _line_intersect(
    p1: dict, p2: dict, p3: dict, p4: dict,
) -> tuple[float, float] | None:
    denom = (p1["x"] - p2["x"]) * (p3["y"] - p4["y"]) - \
            (p1["y"] - p2["y"]) * (p3["x"] - p4["x"])
    if abs(denom) < 1e-9:
        return None
    t = ((p1["x"] - p3["x"]) * (p3["y"] - p4["y"]) -
         (p1["y"] - p3["y"]) * (p3["x"] - p4["x"])) / denom
    return (
        p1["x"] + t * (p2["x"] - p1["x"]),
        p1["y"] + t * (p2["y"] - p1["y"]),
    )


def compute_features(corners: list[dict]) -> dict[str, float]:
    # 0=BL, 1=BR, 2=TR, 3=TL
    near_y = (corners[0]["y"] + corners[1]["y"]) / 2
    far_y = (corners[2]["y"] + corners[3]["y"]) / 2
    near_width_x = abs(corners[1]["x"] - corners[0]["x"])
    far_width_x = abs(corners[2]["x"] - corners[3]["x"])
    court_depth_y = max(1e-6, near_y - far_y)
    diag = _line_intersect(corners[0], corners[2], corners[1], corners[3])
    midline_y = diag[1] if diag else (near_y + far_y) / 2
    # Project midline endpoints onto sidelines
    baseline_vp = _line_intersect(corners[0], corners[1], corners[3], corners[2])
    if diag and baseline_vp:
        left = _line_intersect(diag_p1(diag), diag_p2(baseline_vp), corners[3], corners[0])
        right = _line_intersect(diag_p1(diag), diag_p2(baseline_vp), corners[2], corners[1])
        midline_width_x = abs(right[0] - left[0]) if (left and right) else 0.0
    else:
        midline_width_x = (near_width_x + far_width_x) / 2
    trapezoid_aspect = ((near_width_x + far_width_x) / 2) / court_depth_y
    return {
        "midline_y": midline_y,
        "near_y": near_y,
        "far_y": far_y,
        "court_depth_y": court_depth_y,
        "midline_width_x": midline_width_x,
        "near_width_x": near_width_x,
        "far_width_x": far_width_x,
        "trapezoid_aspect": trapezoid_aspect,
    }


# Helper wrappers to keep `_line_intersect` typed as Corner-shape dicts
def diag_p1(t: tuple[float, float]) -> dict:
    return {"x": t[0], "y": t[1]}


def diag_p2(t: tuple[float, float]) -> dict:
    return {"x": t[0], "y": t[1]}


def load_samples() -> list[Sample]:
    out: list[Sample] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, COALESCE(v.fps, 30) AS fps,
                   v.court_calibration_net_top_y AS gt,
                   v.court_calibration_json
            FROM videos v
            WHERE v.court_calibration_net_top_y IS NOT NULL
              AND v.court_calibration_json IS NOT NULL
            ORDER BY v.name
            """,
        )
        for vname, fps, gt, corners_json in cur:
            corners = corners_json if isinstance(corners_json, list) else json.loads(corners_json)
            if len(corners) != 4:
                continue
            feats = compute_features(corners)
            out.append(Sample(name=vname, fps=fps, gt=gt, features=feats))
    return out


# ---------------------------------------------------------------------------
# Models — each takes X (n×d) and y (n,), fits, predicts. We do LOO-CV
# externally by training on n-1 samples and predicting the 1 left out.
# ---------------------------------------------------------------------------
def _fit_predict_ols(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    # Add bias term
    X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    # OLS via lstsq (handles near-rank-deficient X)
    coefs, *_ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
    return X_test_b @ coefs


def _fit_predict_ridge(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    alpha: float = 0.01,
) -> np.ndarray:
    X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    d = X_train_b.shape[1]
    A = X_train_b.T @ X_train_b + alpha * np.eye(d)
    A[0, 0] -= alpha  # don't regularise bias
    b = X_train_b.T @ y_train
    coefs = np.linalg.solve(A, b)
    return X_test_b @ coefs


FEATURE_SETS = {
    "M0 midline only":          ["midline_y"],
    "M1 midline + depth":       ["midline_y", "court_depth_y"],
    "M2 midline + far + depth": ["midline_y", "far_y", "court_depth_y"],
    "M3 all 8 OLS":             [
        "midline_y", "near_y", "far_y", "court_depth_y",
        "midline_width_x", "near_width_x", "far_width_x", "trapezoid_aspect",
    ],
    "M4 all 8 ridge α=0.01":    [
        "midline_y", "near_y", "far_y", "court_depth_y",
        "midline_width_x", "near_width_x", "far_width_x", "trapezoid_aspect",
    ],
}


def feature_matrix(samples: list[Sample], names: list[str]) -> np.ndarray:
    return np.array([[s.features[n] for n in names] for s in samples])


def loo_cv(samples: list[Sample], names: list[str], use_ridge: bool) -> list[float]:
    n = len(samples)
    X = feature_matrix(samples, names)
    y = np.array([s.gt for s in samples])
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if use_ridge:
            preds[i] = _fit_predict_ridge(X[mask], y[mask], X[i:i+1])[0]
        else:
            preds[i] = _fit_predict_ols(X[mask], y[mask], X[i:i+1])[0]
    return [float(p - y[i]) for i, p in enumerate(preds)]


def _summarize(name: str, errors: list[float]) -> str:
    abs_e = [abs(e) for e in errors]
    return (
        f"  {name:<32} n={len(errors):>3}  "
        f"med |Δ|={statistics.median(abs_e):.3f}  "
        f"mean |Δ|={statistics.mean(abs_e):.3f}  "
        f"worst |Δ|={max(abs_e):.3f}  "
        f"mean signed Δ={statistics.mean(errors):+.3f}  "
        f">0.05: {sum(1 for a in abs_e if a > 0.05):<2}  "
        f">0.10: {sum(1 for a in abs_e if a > 0.10):<2}  "
        f">0.15: {sum(1 for a in abs_e if a > 0.15)}"
    )


def main() -> int:
    samples = load_samples()
    print(f"Loaded {len(samples)} samples (videos with calibration + GT)\n", flush=True)

    # Baseline: A0 per-rally median (re-pulled to allow side-by-side)
    print("=== LOO-CV (leave-one-video-out) ===", flush=True)
    print("  baseline A0 (X-K result):       med |Δ|=0.025  mean |Δ|=0.032  worst |Δ|=0.178  >0.05: 14  >0.10: 2", flush=True)
    print("  baseline B  (X-K hand-tuned):   med |Δ|=0.206  mean |Δ|=0.210  worst |Δ|=0.323  >0.05: 77  >0.10: 77", flush=True)
    print()

    best_med = float("inf")
    best_name = ""
    for name, feats in FEATURE_SETS.items():
        use_ridge = "ridge" in name
        errors = loo_cv(samples, feats, use_ridge=use_ridge)
        print(_summarize(name, errors), flush=True)
        med = statistics.median([abs(e) for e in errors])
        if med < best_med:
            best_med = med
            best_name = name

    print()
    print(f"Best LOO-CV: {best_name} (med |Δ|={best_med:.3f})", flush=True)

    # Fit the best model on ALL 77 samples and dump coefficients so we can
    # bake them into the pipeline.
    print()
    print("=== Best-model coefficients (fit on all 77) ===", flush=True)
    best_feats = FEATURE_SETS[best_name]
    X = feature_matrix(samples, best_feats)
    y = np.array([s.gt for s in samples])
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    if "ridge" in best_name:
        d = X_b.shape[1]
        A = X_b.T @ X_b + 0.01 * np.eye(d)
        A[0, 0] -= 0.01
        coefs = np.linalg.solve(A, X_b.T @ y)
    else:
        coefs, *_ = np.linalg.lstsq(X_b, y, rcond=None)
    print(f"  intercept: {coefs[0]:+.4f}", flush=True)
    for i, name in enumerate(best_feats):
        print(f"  {name:<24} {coefs[i+1]:+.4f}", flush=True)

    # Worst-case rescues
    print()
    print("=== Top-5 A0-worst videos: does best model rescue them? ===", flush=True)
    a0_errors = {s.name: None for s in samples}
    # Read A0 from probe X-J output… for simplicity just run prediction on
    # the worst cases listed in X-K.
    a0_worst = [
        ("caca", 0.352, 0.178),
        ("hehe", 0.487, 0.146),
        ("michu", 0.321, 0.083),
        ("macho", 0.350, 0.081),
        ("kiki",  0.441, 0.077),
    ]
    name_to_sample = {s.name: s for s in samples}
    for vname, gt, a0_err in a0_worst:
        if vname not in name_to_sample:
            continue
        s = name_to_sample[vname]
        X_s = np.array([[s.features[n] for n in best_feats]])
        X_s_b = np.hstack([np.ones((1, 1)), X_s])
        pred = float((X_s_b @ coefs).flatten()[0])
        err = pred - gt
        print(
            f"  {vname:<8} gt={gt:.3f}  A0 |Δ|={a0_err:.3f}  "
            f"BEST pred={pred:.3f}  |Δ|={abs(err):.3f}  "
            f"({'WIN' if abs(err) < a0_err else 'LOSS'} vs A0)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
