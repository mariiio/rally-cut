"""CLI entry point for within-team ReID training infrastructure.

Run from analysis/ directory:
    uv run python -m training.within_team_reid.cli build-eval-cache
    uv run python -m training.within_team_reid.cli train --variant v1
    uv run python -m training.within_team_reid.cli eval-only --variant v1
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(help="Within-team ReID head training (Session 3)")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("within_team_reid.cli")


@app.command("build-eval-cache")
def build_eval_cache(
    output_npz: Path = typer.Option(
        Path("reports/within_team_reid/eval_cache/eval_cache.npz"),
        "--output-npz",
        help="Path for stacked DINOv2 feature .npz",
    ),
    output_meta: Path = typer.Option(
        Path("reports/within_team_reid/eval_cache/metadata.json"),
        "--output-meta",
        help="Path for event/entry metadata sidecar",
    ),
    reid_debug_dir: Path = typer.Option(
        Path("reports/tracking_audit/reid_debug"),
        "--reid-debug-dir",
        help="Directory containing per-rally audit JSONs",
    ),
    cross_rally_ref: Path = typer.Option(
        Path("reports/tracking_audit/reid_debug/sota_probe_cross_rally.json"),
        "--cross-rally-ref",
        help="Reference probe gallery JSON (provides video set for cross-rally guard)",
    ),
    held_out_start: int = typer.Option(34, "--held-out-start"),
    held_out_end: int = typer.Option(58, "--held-out-end"),
) -> None:
    """Extract DINOv2 features for held-out swap events + cross-rally gallery."""
    from .eval.cache import build_cache

    cache = build_cache(
        cache_npz=output_npz,
        cache_meta=output_meta,
        reid_debug_dir=reid_debug_dir,
        held_out_start=held_out_start,
        held_out_end=held_out_end,
        cross_rally_entries_json=cross_rally_ref if cross_rally_ref.exists() else None,
    )
    logger.info(
        "Eval cache built: %d held-out events (%d scorable), %d cross-rally entries",
        len(cache.held_out_events),
        sum(1 for m in cache.held_out_events if m.is_scorable()),
        len(cache.cross_rally_entries),
    )


@app.command("verify-baseline")
def verify_baseline(
    cache_npz: Path = typer.Option(
        Path("reports/within_team_reid/eval_cache/eval_cache.npz"),
        "--cache-npz",
    ),
    cache_meta: Path = typer.Option(
        Path("reports/within_team_reid/eval_cache/metadata.json"),
        "--cache-meta",
    ),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Smoke test: identity head reproduces DINOv2-S zero-shot baselines.

    Held-out within-team rank-1 ≈ 10% (probe baseline).
    Cross-rally rank-1 ≈ 0.703 (probe baseline, used as 0.683 CF guard).
    """
    import torch

    from .eval.cache import load_cache
    from .eval.cross_rally import evaluate as evaluate_cross_rally
    from .eval.held_out import evaluate_identity_baseline

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    torch_device = torch.device(device)

    cache = load_cache(cache_npz, cache_meta)
    logger.info(
        "Loaded cache: %d held-out events, %d cross-rally entries",
        len(cache.held_out_events), len(cache.cross_rally_entries),
    )

    held_out_result = evaluate_identity_baseline(cache, torch_device)
    logger.info(
        "Held-out (identity head): rank1=%.3f scored=%d/%d positive=%d tm_mean=%.4f",
        held_out_result.rank1, held_out_result.n_scored, held_out_result.n_total,
        held_out_result.n_positive, held_out_result.teammate_margin_mean,
    )
    expected_lo, expected_hi = 0.06, 0.18  # 10% ± a few events of noise
    if not (expected_lo <= held_out_result.rank1 <= expected_hi):
        logger.warning(
            "  ⚠ held-out rank1 %.3f outside expected range [%.2f, %.2f] for DINOv2-S",
            held_out_result.rank1, expected_lo, expected_hi,
        )

    from .eval.held_out import IdentityHead
    cross_result = evaluate_cross_rally(cache, IdentityHead(), torch_device)
    logger.info(
        "Cross-rally (identity head): rank1=%.4f n_queries=%d (probe baseline=0.703)",
        cross_result.rank1, cross_result.n_queries,
    )
    if abs(cross_result.rank1 - 0.703) > 0.02:
        logger.warning(
            "  ⚠ cross-rally rank1 %.4f drifts > 2pp from probe baseline 0.703",
            cross_result.rank1,
        )
    else:
        logger.info("  ✓ cross-rally matches probe baseline within ±2pp")


@app.command("train")
def train_variant(
    variant: str = typer.Option("v1", "--variant", help="v1, v2, or v3"),
    device: str = typer.Option("auto", "--device"),
    smoke: int = typer.Option(0, "--smoke", help="If >0, run N batches for smoke test then exit"),
    train_pairs: int = typer.Option(0, "--train-pairs", help="If >0, cap training set to this many pairs (overfit test)"),
    epochs: int | None = typer.Option(None, "--epochs", help="Override config.epochs"),
) -> None:
    """Train an MLP head variant on the within-team-reid corpus."""
    from .config import VARIANT_CONFIGS
    from .train import run_variant

    cfg = VARIANT_CONFIGS[variant]
    if epochs is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "epochs": epochs})
    if device != "auto":
        cfg = type(cfg)(**{**cfg.__dict__, "device": device})

    run_variant(cfg, smoke=smoke, train_pairs_cap=train_pairs)


if __name__ == "__main__":
    app()
