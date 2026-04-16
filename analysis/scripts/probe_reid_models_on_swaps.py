"""SOTA ReID probe on within-team swap events.

Tests whether any off-the-shelf model (OSNet / DINOv2-S / DINOv2-L / CLIP)
has within-team identity signal beyond HSV. Reads the 58 audit-produced
swap events from `reports/tracking_audit/reid_debug/<rally>.json`, sorts
by (rallyId, swapFrame), splits 34 ranking / 24 held-out, then runs
Steps 0–5 from the plan (calibration → probe → ranking → cross-rally →
held-out).

See /Users/mario/.claude/plans/fancy-drifting-rainbow.md for full design.
No training, no harvest, no integration — probe only.

Usage:
    uv run python scripts/probe_reid_models_on_swaps.py \
        --output-dir reports/tracking_audit/reid_debug
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
    load_rallies_for_video,
)
from rallycut.tracking.player_features import extract_bbox_crop
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.swap_reid_probe import (
    get_rally_track_to_player,
    load_player_profiles_from_match_analysis,
    probe_swap,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sota_probe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_FRAMES = 15                 # ±15 frames around swap (matches swap_reid_probe default)
MAX_CROPS_PER_WINDOW = 10          # Cap crops per pre/post window to bound cost
MIN_VALID_CROPS_PER_ANCHOR = 3     # Abstain if any anchor has fewer valid crops
CROP_MIN_HEIGHT_FRAC = 0.05        # 5% of frame height minimum bbox height
CROP_EDGE_MARGIN_FRAC = 0.02       # 2% edge margin
CROP_OCCLUSION_IOU = 0.30          # IoU with another primary above which crop is occluded
SIGMA_K = 1.0                      # σ-relative threshold for non-HSV models
HSV_CALIBRATION_TOLERANCE = 1e-5   # cost match tolerance for Step 0 calibration
HSV_CALIBRATION_BIN_TOLERANCE = 3  # ±3 events noise budget on aggregate bins
RANKING_SIZE = 34                  # plan non-negotiable
HSV_DELTA_HAS_SIGNAL = 0.08        # matches swap_reid_probe
HSV_DELTA_BLIND = 0.05

# Winner-gate constants (from plan)
RANKING_NORMALIZED_MARGIN_BAR = 1.5     # ≥ 1.5 × OSNet mean on ranking set
HELDOUT_NORMALIZED_MARGIN_BAR = 1.2     # ≥ 1.2 × OSNet mean on held-out
CROSS_RALLY_REGRESSION_BAR = -0.02      # rank-1 must be within 2pp of OSNet
POSITIVE_PROP_BAR = 0.50                # ≥ 50% events with positive signal

BIN_HAD_SIGNAL = "had_signal"
BIN_BLIND = "blind"
BIN_WRONG = "wrong_preference"
BIN_ABSTAIN = "abstain"

MODEL_HSV = "hsv"
MODEL_OSNET = "osnet_x1_0_supcon"
MODEL_DINOV2_S = "dinov2_vits14"
MODEL_DINOV2_L = "dinov2_vitl14"
MODEL_CLIP_B32 = "clip_vit_b_32"

CANDIDATES_ORDER = [
    MODEL_HSV,
    MODEL_OSNET,
    MODEL_DINOV2_S,
    MODEL_DINOV2_L,
    MODEL_CLIP_B32,
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SwapEvent:
    """One swap event loaded from an audit JSON + canonical mapping derived."""

    rally_id: str
    video_id: str
    swap_frame: int
    gt_track_id: int
    gt_label: str
    pred_old: int
    pred_new: int
    prior_gt_of_new: int
    # priorGtOfNew → canonical (the player pred_new SHOULD have kept tracking)
    correct_player_id: int | None
    # gt_track_id → canonical (what the swap says pred_new is now)
    wrong_player_id: int | None

    @property
    def key(self) -> tuple[str, int]:
        """Stable sort key: (rallyId, swapFrame)."""
        return (self.rally_id, self.swap_frame)


@dataclass
class ProbeOutcome:
    """Per-event per-model probe output."""

    rally_id: str
    swap_frame: int
    pred_old: int
    pred_new: int
    model: str
    # signal = cos(query, anchor_correct) - cos(query, anchor_wrong)
    # where anchor_correct = pred_old pre-swap (physically same human as query),
    # anchor_wrong = pred_new pre-swap (teammate). Positive = model had signal.
    signal: float | None
    n_crops_query: int
    n_crops_correct: int
    n_crops_wrong: int
    bin: str
    abstain_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelStats:
    """Aggregate statistics for a candidate on one event set."""

    model: str
    n_events_total: int
    n_abstained: int
    mean_signal: float
    std_signal: float
    normalized_mean: float  # vs OSNet mean on same set
    bin_counts: dict[str, int]
    proportion_positive: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


def _player_id_from_gt_label(label: str) -> int | None:
    if not label or not label.startswith("player_"):
        return None
    try:
        pid = int(label.split("_", 1)[1])
        return pid if 1 <= pid <= 4 else None
    except ValueError:
        return None


def _load_events_from_audit(audit_path: Path) -> list[SwapEvent]:
    """Re-derive swap events from a rally-level audit JSON.

    Mirrors `scripts/debug_reid_at_swaps.py::_load_swap_events_from_audit`
    so the loaded set matches the stored per-rally reid_debug JSON exactly.
    """
    audit = json.loads(audit_path.read_text())
    gt_label_by_id: dict[int, str] = {
        int(g["gtTrackId"]): g["gtLabel"] for g in audit.get("perGt", [])
    }

    pred_history: dict[int, list[tuple[int, int, int]]] = {}
    for g in audit.get("perGt", []):
        for s, e, pid in g.get("predIdSpans", []):
            pred_history.setdefault(pid, []).append((g["gtTrackId"], s, e))
    for h in pred_history.values():
        h.sort(key=lambda t: t[1])

    def prior_gt_of(pid: int, before: int) -> int | None:
        last_gt: int | None = None
        for gt_id, s, _e in pred_history.get(pid, []):
            if s >= before:
                break
            last_gt = gt_id
        return last_gt

    out: list[SwapEvent] = []
    for g in audit.get("perGt", []):
        spans = g.get("predIdSpans", [])
        for prev, cur in zip(spans, spans[1:]):
            _, _prev_end, prev_pred = prev
            cur_start, _, cur_pred = cur
            if prev_pred == cur_pred:
                continue
            if prev_pred < 0 or cur_pred < 0:
                continue
            incoming = prior_gt_of(cur_pred, cur_start)
            if incoming is None or incoming == g["gtTrackId"]:
                continue
            correct = _player_id_from_gt_label(gt_label_by_id.get(incoming, ""))
            wrong = _player_id_from_gt_label(g["gtLabel"])
            out.append(SwapEvent(
                rally_id=audit["rallyId"],
                video_id=audit["videoId"],
                swap_frame=cur_start,
                gt_track_id=g["gtTrackId"],
                gt_label=g["gtLabel"],
                pred_old=prev_pred,
                pred_new=cur_pred,
                prior_gt_of_new=incoming,
                correct_player_id=correct,
                wrong_player_id=wrong,
            ))
    return out


def load_all_events(reid_debug_dir: Path) -> list[SwapEvent]:
    """Load every swap event across the 18 rally JSONs, sorted by (rally_id, swap_frame).

    Uses the audit-level rally JSONs (reports/tracking_audit/<rally>.json) as the
    source of truth — not the stored reid_debug JSONs — because the audit has
    full `perGt/predIdSpans` needed to re-derive pred-exchange events. The
    reid_debug dir tells us which rallies have swap events (each rally with
    stored HSV output = rally with ≥1 swap event).
    """
    audit_dir = reid_debug_dir.parent  # reports/tracking_audit/
    rally_ids: list[str] = []
    for p in sorted(reid_debug_dir.glob("*.json")):
        # _summary.json convention varies; _load_events handles only per-rally files.
        name = p.name
        if name.startswith("_") or "sota_probe" in name:
            continue
        rally_ids.append(name.removesuffix(".json"))

    events: list[SwapEvent] = []
    for rid in rally_ids:
        audit_path = audit_dir / f"{rid}.json"
        if not audit_path.exists():
            logger.warning("  audit missing for rally %s, skipping", rid[:8])
            continue
        events.extend(_load_events_from_audit(audit_path))

    events.sort(key=lambda e: e.key)
    return events


def load_stored_reid_debug(reid_debug_dir: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    """Load each stored per-rally reid_debug JSON into a lookup by
    (rally_id, swap_frame, pred_old, pred_new) — since the same (rally, frame)
    can have multiple events for different GT tracks."""
    store: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for p in sorted(reid_debug_dir.glob("*.json")):
        if p.name.startswith("_") or "sota_probe" in p.name:
            continue
        data = json.loads(p.read_text())
        for ev in data:
            key = (ev["rallyId"], int(ev["swapFrame"]), int(ev["predOld"]), int(ev["predNew"]))
            store[key] = ev
    return store


# ---------------------------------------------------------------------------
# Rally context loader — video, FPS, predictions, primary tracks, profiles
# ---------------------------------------------------------------------------


@dataclass
class RallyContext:
    rally_id: str
    video_id: str
    video_path: Path
    start_ms: int
    video_fps: float
    frame_width: int
    frame_height: int
    predictions: list[PlayerPosition]
    primary_track_ids: list[int]
    match_analysis: dict[str, Any]


def _fetch_rally_context(rally_id: str) -> RallyContext | None:
    """Fetch everything needed to probe one rally: video, predictions, match_analysis.

    Falls back to raw DB query for fields load_labeled_rallies doesn't expose
    cleanly (primary_track_ids, frame_width/height, full match_analysis_json).
    """
    # Use load_labeled_rallies for predictions (already handles FPS scaling etc.)
    rallies = load_labeled_rallies(rally_id=rally_id)
    if not rallies:
        logger.warning("  rally %s not labelled", rally_id[:8])
        return None
    r = rallies[0]
    if r.predictions is None or not r.predictions.positions:
        logger.warning("  rally %s has no predictions", rally_id[:8])
        return None

    video_path = get_video_path(r.video_id)
    if video_path is None:
        logger.warning("  can't fetch video for rally %s", rally_id[:8])
        return None

    # Pull match_analysis + frame dims.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json, width, height FROM videos WHERE id = %s",
                [r.video_id],
            )
            row = cur.fetchone()
            if not row or not row[0]:
                logger.warning("  no match_analysis for video %s", r.video_id[:8])
                return None
            match_analysis = cast(dict[str, Any], row[0])
            frame_w = int(cast(Any, row[1]) or r.video_width or 1920)
            frame_h = int(cast(Any, row[2]) or r.video_height or 1080)

    return RallyContext(
        rally_id=r.rally_id,
        video_id=r.video_id,
        video_path=video_path,
        start_ms=r.start_ms,
        video_fps=r.video_fps,
        frame_width=frame_w,
        frame_height=frame_h,
        predictions=list(r.predictions.positions),
        primary_track_ids=list(r.predictions.primary_track_ids or []),
        match_analysis=match_analysis,
    )


# ---------------------------------------------------------------------------
# Frame + crop extraction
# ---------------------------------------------------------------------------


def _read_rally_frames(
    video_path: Path,
    start_ms: int,
    video_fps: float,
    rally_frames: set[int],
) -> dict[int, NDArray[np.uint8]]:
    """Read a set of rally-relative frames from video.

    Mirrors swap_reid_probe._iterate_rally_frames but returns the same
    rally_frame → BGR ndarray mapping. Sequential read after seek to first.
    """
    if not rally_frames:
        return {}
    start_frame = int(round(start_ms / 1000.0 * video_fps))
    abs_needed = sorted(start_frame + f for f in rally_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("  cannot open %s", video_path)
        return {}

    out: dict[int, NDArray[np.uint8]] = {}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_needed[0])
        cur = abs_needed[0]
        idx = 0
        while idx < len(abs_needed):
            target = abs_needed[idx]
            while cur < target:
                cap.read()
                cur += 1
                if cur > target + 10_000:
                    break
            ok, frame = cap.read()
            cur += 1
            if not ok:
                break
            if target == abs_needed[idx]:
                rf = target - start_frame
                out[rf] = np.asarray(frame, dtype=np.uint8)
                idx += 1
    finally:
        cap.release()
    return out


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU on normalized (cx, cy, w, h) bboxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _is_quality_crop(
    pos: PlayerPosition,
    primary_positions_at_frame: list[PlayerPosition],
) -> bool:
    """Apply the crop-quality filter: size, edge margin, occlusion."""
    if pos.height < CROP_MIN_HEIGHT_FRAC:
        return False
    left = pos.x - pos.width / 2
    right = pos.x + pos.width / 2
    top = pos.y - pos.height / 2
    bottom = pos.y + pos.height / 2
    if (left < CROP_EDGE_MARGIN_FRAC or right > 1.0 - CROP_EDGE_MARGIN_FRAC
            or top < CROP_EDGE_MARGIN_FRAC or bottom > 1.0 - CROP_EDGE_MARGIN_FRAC):
        return False
    for other in primary_positions_at_frame:
        if other.track_id == pos.track_id:
            continue
        iou = _bbox_iou(
            (pos.x, pos.y, pos.width, pos.height),
            (other.x, other.y, other.width, other.height),
        )
        if iou > CROP_OCCLUSION_IOU:
            return False
    return True


def _collect_crops_for_pred(
    pred_id: int,
    frame_range: range,
    predictions: list[PlayerPosition],
    primary_set: set[int],
    frames_by_rally_frame: dict[int, NDArray[np.uint8]],
    frame_width: int,
    frame_height: int,
) -> list[NDArray[np.uint8]]:
    """Collect up to MAX_CROPS_PER_WINDOW quality-filtered BGR crops for pred_id."""
    by_frame: dict[int, PlayerPosition] = {}
    primary_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in predictions:
        if p.track_id == pred_id:
            by_frame[p.frame_number] = p
        if p.track_id in primary_set:
            primary_by_frame[p.frame_number].append(p)

    candidates: list[tuple[int, PlayerPosition]] = []
    for f in frame_range:
        pos = by_frame.get(f)
        if pos is None:
            continue
        if frames_by_rally_frame.get(f) is None:
            continue
        if not _is_quality_crop(pos, primary_by_frame.get(f, [])):
            continue
        candidates.append((f, pos))

    # Subsample to at most MAX_CROPS_PER_WINDOW, evenly spaced.
    if len(candidates) > MAX_CROPS_PER_WINDOW:
        idxs = np.linspace(0, len(candidates) - 1, MAX_CROPS_PER_WINDOW).astype(int)
        candidates = [candidates[i] for i in idxs]

    crops: list[NDArray[np.uint8]] = []
    for f, pos in candidates:
        frame = frames_by_rally_frame[f]
        crop = extract_bbox_crop(
            frame, (pos.x, pos.y, pos.width, pos.height), frame_width, frame_height,
        )
        if crop is not None:
            crops.append(crop)
    return crops


# ---------------------------------------------------------------------------
# Embedding backbones
# ---------------------------------------------------------------------------


def _default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbedBackbone:
    """Thin wrapper around a ReID/foundation embedding model.

    Each subclass owns its native preprocessor; all produce L2-normalized
    float32 embeddings of shape (N, D).
    """

    name: str = "<abstract>"
    dim: int = 0
    status: str = "untested"  # hot | download | missing | failed

    def is_available(self) -> bool:  # best-effort check without loading weights
        return False

    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.floating]:
        raise NotImplementedError


class OSNetBackbone(EmbedBackbone):
    name = MODEL_OSNET
    dim = 128

    def __init__(self) -> None:
        self._model: Any = None
        self._weights_path: Path | None = None

    def is_available(self) -> bool:
        from rallycut.tracking.reid_general import WEIGHTS_PATH
        self._weights_path = Path(WEIGHTS_PATH)
        if not self._weights_path.exists():
            self.status = "missing"
            return False
        self.status = "hot"
        return True

    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.floating]:
        from rallycut.tracking.reid_general import GeneralReIDModel
        if self._model is None:
            assert self._weights_path is not None
            self._model = GeneralReIDModel(weights_path=self._weights_path)
        result: NDArray[np.floating] = self._model.extract_embeddings(crops)
        return result


class DINOv2Backbone(EmbedBackbone):
    """DINOv2 ViT-S or ViT-L via torch.hub."""

    def __init__(self, variant: str) -> None:
        self.name = variant  # "dinov2_vits14" | "dinov2_vitl14"
        self.dim = 384 if variant == "dinov2_vits14" else 1024
        self._model: Any = None
        self._mean: Any = None
        self._std: Any = None
        self._device: str | None = None

    def is_available(self) -> bool:
        # Heuristic: torch.hub caches under ~/.cache/torch/hub; if the repo
        # dir + any vit weights exist, we call it hot. Otherwise "download".
        import os
        hub_dir = Path(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"))
        weight_pat = "dinov2_vitl14_pretrain.pth" if "vitl14" in self.name else "dinov2_vits14_pretrain.pth"
        weights = Path(os.path.expanduser(f"~/.cache/torch/hub/checkpoints/{weight_pat}"))
        if hub_dir.exists() and weights.exists():
            self.status = "hot"
        else:
            self.status = "download"
        return True

    def _load(self) -> None:
        import torch
        device = _default_device()
        logger.info("  loading %s on %s...", self.name, device)
        model = torch.hub.load("facebookresearch/dinov2", self.name)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self._model = model
        self._mean = mean
        self._std = std
        self._device = device

    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.floating]:
        import torch
        import torch.nn.functional as functional  # noqa: N812
        if self._model is None:
            self._load()
        if not crops:
            return np.empty((0, self.dim), dtype=np.float32)
        batch = []
        for c in crops:
            img = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch.append(t)
        x = torch.stack(batch).to(self._device)
        x = (x - self._mean) / self._std
        with torch.inference_mode():
            feats = self._model(x)
        feats = functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)


class CLIPBackbone(EmbedBackbone):
    name = MODEL_CLIP_B32
    dim = 512

    def __init__(self) -> None:
        self._model: Any = None
        self._preprocess: Any = None
        self._device: str | None = None

    def is_available(self) -> bool:
        try:
            import open_clip  # noqa: F401
        except ImportError:
            self.status = "missing"
            return False
        # open_clip caches under ~/.cache/huggingface/hub or ~/.cache/clip
        import os
        for base in (
            "~/.cache/huggingface/hub",
            "~/.cache/clip",
            "~/.cache/open_clip",
        ):
            if any(Path(os.path.expanduser(base)).glob("**/*laion2b*") if Path(os.path.expanduser(base)).exists() else []):
                self.status = "hot"
                return True
        self.status = "download"
        return True

    def _load(self) -> None:
        import open_clip
        self._device = _default_device()
        logger.info("  loading CLIP ViT-B/32 on %s...", self._device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k",
        )
        model = model.to(self._device).eval()
        self._model = model
        self._preprocess = preprocess

    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.floating]:
        import torch
        from PIL import Image
        if self._model is None:
            self._load()
        if not crops:
            return np.empty((0, self.dim), dtype=np.float32)
        tensors = []
        for c in crops:
            img = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
            tensors.append(self._preprocess(pil))
        x = torch.stack(tensors).to(self._device)
        with torch.inference_mode():
            feats = self._model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        result: NDArray[np.floating] = feats.cpu().numpy().astype(np.float32)
        return result


def make_backbones() -> dict[str, EmbedBackbone]:
    """Instantiate every candidate. Availability not yet queried."""
    return {
        MODEL_OSNET: OSNetBackbone(),
        MODEL_DINOV2_S: DINOv2Backbone("dinov2_vits14"),
        MODEL_DINOV2_L: DINOv2Backbone("dinov2_vitl14"),
        MODEL_CLIP_B32: CLIPBackbone(),
    }


# ---------------------------------------------------------------------------
# Signal computation (per-event per-model)
# ---------------------------------------------------------------------------


def _cos(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _mean_normalize(embeds: NDArray[np.floating]) -> NDArray[np.floating] | None:
    if embeds.shape[0] == 0:
        return None
    mean = embeds.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return None
    result: NDArray[np.floating] = mean / norm
    return result


def compute_event_signal(
    event: SwapEvent,
    ctx: RallyContext,
    frames: dict[int, NDArray[np.uint8]],
    backbone: EmbedBackbone,
) -> ProbeOutcome:
    """Compute signal = cos(query, anchor_correct) - cos(query, anchor_wrong).

    anchor_correct = pred_old pre-swap crops (physically same human as query).
    anchor_wrong   = pred_new pre-swap crops (teammate who was tracked by pred_new).
    query          = pred_new post-swap crops.
    """
    pre_range = range(max(0, event.swap_frame - WINDOW_FRAMES), event.swap_frame)
    post_range = range(event.swap_frame, event.swap_frame + WINDOW_FRAMES)

    primary_set = set(ctx.primary_track_ids) or {event.pred_old, event.pred_new}
    # Even if pred_old/pred_new aren't in primary set, include them so occlusion
    # filter considers them relative to primaries.
    primary_set = primary_set | {event.pred_old, event.pred_new}

    correct_crops = _collect_crops_for_pred(
        event.pred_old, pre_range, ctx.predictions, primary_set, frames,
        ctx.frame_width, ctx.frame_height,
    )
    wrong_crops = _collect_crops_for_pred(
        event.pred_new, pre_range, ctx.predictions, primary_set, frames,
        ctx.frame_width, ctx.frame_height,
    )
    query_crops = _collect_crops_for_pred(
        event.pred_new, post_range, ctx.predictions, primary_set, frames,
        ctx.frame_width, ctx.frame_height,
    )

    n_q, n_c, n_w = len(query_crops), len(correct_crops), len(wrong_crops)
    if n_q < MIN_VALID_CROPS_PER_ANCHOR:
        return ProbeOutcome(event.rally_id, event.swap_frame, event.pred_old, event.pred_new,
                            backbone.name, None, n_q, n_c, n_w, BIN_ABSTAIN,
                            abstain_reason=f"query<{MIN_VALID_CROPS_PER_ANCHOR}")
    if n_c < MIN_VALID_CROPS_PER_ANCHOR:
        return ProbeOutcome(event.rally_id, event.swap_frame, event.pred_old, event.pred_new,
                            backbone.name, None, n_q, n_c, n_w, BIN_ABSTAIN,
                            abstain_reason=f"anchor_correct<{MIN_VALID_CROPS_PER_ANCHOR}")
    if n_w < MIN_VALID_CROPS_PER_ANCHOR:
        return ProbeOutcome(event.rally_id, event.swap_frame, event.pred_old, event.pred_new,
                            backbone.name, None, n_q, n_c, n_w, BIN_ABSTAIN,
                            abstain_reason=f"anchor_wrong<{MIN_VALID_CROPS_PER_ANCHOR}")

    q_embs = backbone.embed(query_crops)
    c_embs = backbone.embed(correct_crops)
    w_embs = backbone.embed(wrong_crops)

    q = _mean_normalize(q_embs)
    c = _mean_normalize(c_embs)
    w = _mean_normalize(w_embs)
    if q is None or c is None or w is None:
        return ProbeOutcome(event.rally_id, event.swap_frame, event.pred_old, event.pred_new,
                            backbone.name, None, n_q, n_c, n_w, BIN_ABSTAIN,
                            abstain_reason="zero-norm mean embedding")

    signal = _cos(q, c) - _cos(q, w)
    # Bin assignment happens later after σ is known across events.
    return ProbeOutcome(event.rally_id, event.swap_frame, event.pred_old, event.pred_new,
                        backbone.name, float(signal), n_q, n_c, n_w, bin=BIN_BLIND)


# ---------------------------------------------------------------------------
# Rally-scoped probe loop
# ---------------------------------------------------------------------------


def _group_events_by_rally(events: list[SwapEvent]) -> dict[str, list[SwapEvent]]:
    grouped: dict[str, list[SwapEvent]] = defaultdict(list)
    for e in events:
        grouped[e.rally_id].append(e)
    return grouped


def probe_events_for_rally(
    rally_events: list[SwapEvent],
    ctx: RallyContext,
    backbones: list[EmbedBackbone],
) -> dict[str, list[ProbeOutcome]]:
    """Extract frames once for the rally, then run all backbones on all events."""
    needed: set[int] = set()
    for ev in rally_events:
        pre = range(max(0, ev.swap_frame - WINDOW_FRAMES), ev.swap_frame)
        post = range(ev.swap_frame, ev.swap_frame + WINDOW_FRAMES)
        needed.update(pre)
        needed.update(post)
    frames = _read_rally_frames(ctx.video_path, ctx.start_ms, ctx.video_fps, needed)

    out: dict[str, list[ProbeOutcome]] = {b.name: [] for b in backbones}
    for ev in rally_events:
        for bb in backbones:
            try:
                res = compute_event_signal(ev, ctx, frames, bb)
            except Exception as exc:  # noqa: BLE001 — probes must not crash the run
                logger.warning(
                    "    %s failed on %s@%d: %s", bb.name, ev.rally_id[:8], ev.swap_frame, exc,
                )
                res = ProbeOutcome(ev.rally_id, ev.swap_frame, ev.pred_old, ev.pred_new,
                                   bb.name, None, 0, 0, 0, BIN_ABSTAIN,
                                   abstain_reason=f"exception: {exc.__class__.__name__}")
            out[bb.name].append(res)
    return out


# ---------------------------------------------------------------------------
# Step 0 — HSV calibration via probe_swap vs stored reid_debug JSON
# ---------------------------------------------------------------------------


def run_hsv_calibration(
    ranking_events: list[SwapEvent],
    stored: dict[tuple[str, int, int, int], dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Re-run probe_swap on ranking events; compare to stored JSON.

    Returns (passed, report). `passed` is True iff per-event classifications
    match 100% OR aggregate bin counts are within ±3.
    """
    logger.info("Step 0 — HSV calibration on %d ranking events", len(ranking_events))
    n_exact_match = 0
    n_missing = 0
    per_event: list[dict[str, Any]] = []
    new_counts: dict[str, int] = defaultdict(int)
    stored_counts: dict[str, int] = defaultdict(int)
    max_cost_delta = 0.0

    events_by_rally = _group_events_by_rally(ranking_events)
    for rally_id, rally_events in events_by_rally.items():
        ctx = _fetch_rally_context(rally_id)
        if ctx is None:
            for ev in rally_events:
                per_event.append({
                    "rally_id": rally_id, "swap_frame": ev.swap_frame,
                    "status": "missing_context",
                })
                n_missing += 1
            continue
        profiles = load_player_profiles_from_match_analysis(ctx.match_analysis)
        for ev in rally_events:
            key = (ev.rally_id, ev.swap_frame, ev.pred_old, ev.pred_new)
            stored_rec = stored.get(key)
            if stored_rec is None:
                per_event.append({
                    "rally_id": rally_id, "swap_frame": ev.swap_frame,
                    "status": "no_stored_record",
                })
                n_missing += 1
                continue
            result = probe_swap(
                rally_id=ev.rally_id,
                swap_frame=ev.swap_frame,
                gt_track_id=ev.gt_track_id,
                pred_old=ev.pred_old,
                pred_new=ev.pred_new,
                prior_gt_of_new=ev.prior_gt_of_new,
                video_path=ctx.video_path,
                rally_start_ms=float(ctx.start_ms),
                video_fps=ctx.video_fps,
                player_profiles=profiles,
                correct_player_id=ev.correct_player_id,
                wrong_player_id=ev.wrong_player_id,
                predictions=ctx.predictions,
                window=WINDOW_FRAMES,
            )
            new_counts[result.classification] += 1
            stored_counts[stored_rec["classification"]] += 1
            exact = result.classification == stored_rec["classification"]
            if exact:
                n_exact_match += 1
            # Compare cost deltas to float tolerance.
            cost_delta = 0.0
            for pid_str, stored_cost in (stored_rec.get("playerCostsPostSwap") or {}).items():
                pid = int(pid_str)
                new_cost = result.player_costs_post_swap.get(pid)
                if new_cost is None:
                    continue
                cost_delta = max(cost_delta, abs(new_cost - stored_cost))
            max_cost_delta = max(max_cost_delta, cost_delta)
            per_event.append({
                "rally_id": rally_id,
                "swap_frame": ev.swap_frame,
                "pred_old": ev.pred_old,
                "pred_new": ev.pred_new,
                "new_class": result.classification,
                "stored_class": stored_rec["classification"],
                "exact_match": exact,
                "max_cost_delta": cost_delta,
            })
            logger.info(
                "  %s@%d pred%d→%d: new=%s stored=%s exact=%s max|Δcost|=%.2e",
                rally_id[:8], ev.swap_frame, ev.pred_old, ev.pred_new,
                result.classification, stored_rec["classification"], exact, cost_delta,
            )

    matched_total = n_exact_match
    n_scored = len(ranking_events) - n_missing
    exact_rate = matched_total / n_scored if n_scored else 0.0
    bin_deltas = {
        cls: abs(new_counts.get(cls, 0) - stored_counts.get(cls, 0))
        for cls in {*new_counts, *stored_counts}
    }
    max_bin_delta = max(bin_deltas.values()) if bin_deltas else 0
    aggregate_within_tolerance = max_bin_delta <= HSV_CALIBRATION_BIN_TOLERANCE

    passed = exact_rate == 1.0 and max_cost_delta <= HSV_CALIBRATION_TOLERANCE
    # Relaxed fallback gate.
    fallback_passed = exact_rate >= (1 - HSV_CALIBRATION_BIN_TOLERANCE / max(n_scored, 1)) \
        and aggregate_within_tolerance

    report = {
        "n_events": len(ranking_events),
        "n_scored": n_scored,
        "n_missing": n_missing,
        "n_exact_match": matched_total,
        "exact_match_rate": exact_rate,
        "new_counts": dict(new_counts),
        "stored_counts": dict(stored_counts),
        "max_bin_delta": max_bin_delta,
        "max_cost_delta": max_cost_delta,
        "strict_passed": passed,
        "fallback_passed": fallback_passed,
        "per_event": per_event,
    }
    logger.info(
        "Step 0 result: exact_match=%d/%d max_bin_delta=%d max|Δcost|=%.2e "
        "strict=%s fallback=%s",
        matched_total, n_scored, max_bin_delta, max_cost_delta,
        passed, fallback_passed,
    )
    return (passed or fallback_passed), report


# ---------------------------------------------------------------------------
# Step 1 — Model discovery
# ---------------------------------------------------------------------------


def discover_models(backbones: dict[str, EmbedBackbone]) -> dict[str, str]:
    """Query availability for each backbone. Returns status dict."""
    status: dict[str, str] = {}
    for name in CANDIDATES_ORDER:
        if name == MODEL_HSV:
            status[name] = "hot"
            continue
        bb = backbones[name]
        ok = bb.is_available()
        status[name] = bb.status if ok else "missing"
        logger.info("  %-24s %s (dim=%d)", name, status[name], bb.dim)
    return status


# ---------------------------------------------------------------------------
# Step 3 — Ranking: σ-relative bins + normalized mean
# ---------------------------------------------------------------------------


def _classify_signal_with_sigma(signal: float, sigma: float) -> str:
    threshold = SIGMA_K * sigma
    if signal >= threshold:
        return BIN_HAD_SIGNAL
    if signal <= -threshold:
        return BIN_WRONG
    return BIN_BLIND


def aggregate_stats(
    model: str,
    outcomes: list[ProbeOutcome],
    osnet_mean: float | None,
) -> ModelStats:
    valid = [o.signal for o in outcomes if o.signal is not None]
    n_abstained = sum(1 for o in outcomes if o.signal is None)
    if not valid:
        return ModelStats(
            model=model, n_events_total=len(outcomes), n_abstained=n_abstained,
            mean_signal=0.0, std_signal=0.0, normalized_mean=0.0,
            bin_counts={BIN_HAD_SIGNAL: 0, BIN_BLIND: 0, BIN_WRONG: 0, BIN_ABSTAIN: n_abstained},
            proportion_positive=0.0,
        )
    mean = statistics.fmean(valid)
    std = statistics.pstdev(valid) if len(valid) >= 2 else 0.0
    sigma = std if std > 1e-6 else 1e-6
    bin_counts = {BIN_HAD_SIGNAL: 0, BIN_BLIND: 0, BIN_WRONG: 0, BIN_ABSTAIN: n_abstained}
    for o in outcomes:
        if o.signal is None:
            continue
        b = _classify_signal_with_sigma(o.signal, sigma)
        o.bin = b
        bin_counts[b] += 1
    prop_pos = sum(1 for s in valid if s > 0) / len(valid)
    normalized = mean / osnet_mean if osnet_mean and abs(osnet_mean) > 1e-6 else 0.0
    return ModelStats(
        model=model, n_events_total=len(outcomes), n_abstained=n_abstained,
        mean_signal=mean, std_signal=std, normalized_mean=normalized,
        bin_counts=bin_counts, proportion_positive=prop_pos,
    )


# ---------------------------------------------------------------------------
# Step 4 — Adversarial cross-rally rank-1
# ---------------------------------------------------------------------------


@dataclass
class GalleryEntry:
    video_id: str
    rally_id: str
    canonical_id: int
    embedding: NDArray[np.floating]


def build_cross_rally_gallery(
    video_ids: list[str],
    backbones: list[EmbedBackbone],
) -> dict[str, list[GalleryEntry]]:
    """For each candidate, embed each (video, rally, canonical_id) as one mean vector.

    Uses primary-track predictions from player_tracks + playerProfiles canonical
    IDs from match_analysis_json. Skips videos with <2 tracked rallies (can't
    leave-one-out).
    """
    per_model: dict[str, list[GalleryEntry]] = {bb.name: [] for bb in backbones}

    for vid in video_ids:
        # Load match_analysis for canonical mapping.
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT match_analysis_json, width, height, fps FROM videos WHERE id = %s",
                    [vid],
                )
                row = cur.fetchone()
        if not row or not row[0]:
            logger.info("  skip %s: no match_analysis_json", vid[:8])
            continue
        match_analysis = cast(dict[str, Any], row[0])
        frame_w = int(cast(Any, row[1]) or 1920)
        frame_h = int(cast(Any, row[2]) or 1080)
        default_fps = float(cast(Any, row[3]) or 30.0)

        rally_tracks = load_rallies_for_video(vid)
        if len(rally_tracks) < 2:
            logger.info("  skip %s: <2 tracked rallies (%d)", vid[:8], len(rally_tracks))
            continue
        video_path = get_video_path(vid)
        if video_path is None:
            logger.info("  skip %s: no video", vid[:8])
            continue

        for r in rally_tracks:
            t2p = get_rally_track_to_player(match_analysis, r.rally_id)
            if not t2p:
                continue
            primary_set = set(r.primary_track_ids) or set(t2p.keys())
            canonical_to_track: dict[int, int] = {}
            for track_id, canonical in t2p.items():
                if track_id in primary_set:
                    canonical_to_track[canonical] = track_id

            # Collect a small frame batch from the middle third of the rally for stability.
            if not r.positions:
                continue
            frames_per_track: dict[int, list[int]] = defaultdict(list)
            for p in r.positions:
                frames_per_track[p.track_id].append(p.frame_number)
            for t, fs in frames_per_track.items():
                fs.sort()

            # Sample up to MAX_CROPS_PER_WINDOW frames per track, evenly spread.
            track_frames: dict[int, list[int]] = {}
            for canonical, track_id in canonical_to_track.items():
                fs = frames_per_track.get(track_id, [])
                if len(fs) < MIN_VALID_CROPS_PER_ANCHOR:
                    continue
                idxs = np.linspace(0, len(fs) - 1, MAX_CROPS_PER_WINDOW).astype(int)
                track_frames[track_id] = [fs[i] for i in idxs]

            all_needed: set[int] = set()
            for fs in track_frames.values():
                all_needed.update(fs)
            frames_by_rf = _read_rally_frames(video_path, r.start_ms, default_fps, all_needed)

            # Build quality-filtered crops per track.
            pos_by_key: dict[tuple[int, int], PlayerPosition] = {
                (p.track_id, p.frame_number): p for p in r.positions
            }
            primary_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
            for p in r.positions:
                if p.track_id in primary_set:
                    primary_by_frame[p.frame_number].append(p)

            crops_by_canonical: dict[int, list[NDArray[np.uint8]]] = {}
            for canonical, track_id in canonical_to_track.items():
                if track_id not in track_frames:
                    continue
                selected: list[NDArray[np.uint8]] = []
                for f in track_frames[track_id]:
                    pos = pos_by_key.get((track_id, f))
                    frame = frames_by_rf.get(f)
                    if pos is None or frame is None:
                        continue
                    if not _is_quality_crop(pos, primary_by_frame.get(f, [])):
                        continue
                    crop = extract_bbox_crop(
                        frame, (pos.x, pos.y, pos.width, pos.height), frame_w, frame_h,
                    )
                    if crop is not None:
                        selected.append(crop)
                if len(selected) >= MIN_VALID_CROPS_PER_ANCHOR:
                    crops_by_canonical[canonical] = selected

            for bb in backbones:
                for canonical, crops in crops_by_canonical.items():
                    try:
                        embs = bb.embed(crops)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "    %s failed on %s/%s/P%d: %s",
                            bb.name, vid[:8], r.rally_id[:8], canonical, exc,
                        )
                        continue
                    mean = _mean_normalize(embs)
                    if mean is None:
                        continue
                    per_model[bb.name].append(GalleryEntry(
                        video_id=vid,
                        rally_id=r.rally_id,
                        canonical_id=canonical,
                        embedding=mean.astype(np.float32),
                    ))
    return per_model


def compute_rank1(entries: list[GalleryEntry]) -> tuple[float, int]:
    """Leave-one-out per (video, rally): gallery = other rallies in same video.

    Returns (rank1, n_queries).
    """
    if not entries:
        return 0.0, 0
    # Group by video.
    by_video: dict[str, list[GalleryEntry]] = defaultdict(list)
    for e in entries:
        by_video[e.video_id].append(e)

    correct = 0
    total = 0
    for vid, video_entries in by_video.items():
        # Need ≥2 distinct rallies.
        rallies = {e.rally_id for e in video_entries}
        if len(rallies) < 2:
            continue
        for query in video_entries:
            gallery = [g for g in video_entries if g.rally_id != query.rally_id]
            if not gallery:
                continue
            best: GalleryEntry | None = None
            best_sim = -2.0
            for g in gallery:
                sim = _cos(query.embedding, g.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best = g
            total += 1
            if best is not None and best.canonical_id == query.canonical_id:
                correct += 1
    return (correct / total if total else 0.0, total)


# ---------------------------------------------------------------------------
# Verdict + report
# ---------------------------------------------------------------------------


def decide_winner(
    ranking_stats: dict[str, ModelStats],
    heldout_stats: dict[str, ModelStats],
    cross_rally: dict[str, tuple[float, int]],
) -> dict[str, Any]:
    """Apply the winner-gate (plan §Decision rule)."""
    osnet_r = ranking_stats.get(MODEL_OSNET)
    osnet_xrank1 = cross_rally.get(MODEL_OSNET, (0.0, 0))[0]
    if osnet_r is None:
        return {"winner": None, "reason": "OSNet baseline missing — cannot normalize."}

    decisions: dict[str, dict[str, Any]] = {}
    winners: list[str] = []
    for model, stats in ranking_stats.items():
        if model in (MODEL_HSV, MODEL_OSNET):
            decisions[model] = {"verdict": "baseline", "reason": "reference only"}
            continue
        norm_rank = stats.normalized_mean
        norm_held = heldout_stats.get(model)
        xrank1 = cross_rally.get(model, (0.0, 0))[0]
        reasons: list[str] = []
        pass1 = norm_rank >= RANKING_NORMALIZED_MARGIN_BAR
        pass2 = (xrank1 - osnet_xrank1) >= CROSS_RALLY_REGRESSION_BAR
        pass3 = stats.proportion_positive >= POSITIVE_PROP_BAR
        pass_held = False
        if norm_held is not None:
            pass_held = (norm_held.normalized_mean >= HELDOUT_NORMALIZED_MARGIN_BAR
                         and norm_held.proportion_positive >= POSITIVE_PROP_BAR)
        if not pass1:
            reasons.append(f"rank_norm={norm_rank:.2f}<{RANKING_NORMALIZED_MARGIN_BAR}")
        if not pass2:
            reasons.append(f"xrank1={xrank1:.3f} regresses >{-CROSS_RALLY_REGRESSION_BAR}pp vs OSNet={osnet_xrank1:.3f}")
        if not pass3:
            reasons.append(f"pos_prop={stats.proportion_positive:.2f}<{POSITIVE_PROP_BAR}")
        ranking_pass = pass1 and pass2 and pass3
        if ranking_pass and pass_held:
            verdict = "winner"
            winners.append(model)
        elif ranking_pass and not pass_held:
            verdict = "overfit"
            reasons.append("held-out gate failed — candidate for harvest+fine-tune path")
        else:
            verdict = "reject"
        decisions[model] = {
            "verdict": verdict,
            "normalized_rank": norm_rank,
            "normalized_heldout": norm_held.normalized_mean if norm_held else None,
            "cross_rally_rank1": xrank1,
            "proportion_positive_rank": stats.proportion_positive,
            "proportion_positive_held": norm_held.proportion_positive if norm_held else None,
            "reasons": reasons,
        }

    if not winners:
        return {"winner": None, "decisions": decisions,
                "reason": "no candidate cleared all three gates on both sets"}
    # If multiple winners, pick by normalized_rank.
    winners.sort(key=lambda m: ranking_stats[m].normalized_mean, reverse=True)
    return {"winner": winners[0], "decisions": decisions, "tied_winners": winners}


def _fmt_bin(bin_counts: dict[str, int]) -> str:
    return (f"hs={bin_counts.get(BIN_HAD_SIGNAL, 0)} "
            f"bl={bin_counts.get(BIN_BLIND, 0)} "
            f"wr={bin_counts.get(BIN_WRONG, 0)} "
            f"ab={bin_counts.get(BIN_ABSTAIN, 0)}")


def write_report(
    output_dir: Path,
    events_total: int,
    ranking_size: int,
    heldout_size: int,
    discovery: dict[str, str],
    calibration: dict[str, Any],
    ranking_stats: dict[str, ModelStats],
    heldout_stats: dict[str, ModelStats],
    cross_rally: dict[str, tuple[float, int]],
    verdict: dict[str, Any],
    wall_seconds: float,
    skipped_models: dict[str, str],
) -> Path:
    """Emit sota_probe.md + sibling JSONs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "sota_probe.md"

    lines: list[str] = []
    lines.append("# SOTA ReID Probe — within-team swap events")
    lines.append("")
    lines.append(
        f"Wall: {wall_seconds / 60:.1f} min · Events: {events_total} "
        f"(ranking {ranking_size} / held-out {heldout_size})."
    )
    lines.append("")
    lines.append("## Discovery")
    lines.append("| Model | Status |")
    lines.append("|---|---|")
    for name in CANDIDATES_ORDER:
        status_line = skipped_models.get(name) or discovery.get(name, "unknown")
        lines.append(f"| `{name}` | {status_line} |")
    lines.append("")
    lines.append("## Step 0 — HSV Calibration")
    lines.append(f"- Events scored: {calibration['n_scored']} / {calibration['n_events']} "
                 f"({calibration['n_missing']} missing context)")
    lines.append(f"- Exact classification match: "
                 f"{calibration['n_exact_match']}/{calibration['n_scored']} "
                 f"({calibration['exact_match_rate']:.1%})")
    lines.append(f"- Max |Δcost|: {calibration['max_cost_delta']:.2e} "
                 f"(tolerance {HSV_CALIBRATION_TOLERANCE:.0e})")
    lines.append(f"- Aggregate bin max Δ: {calibration['max_bin_delta']} "
                 f"(tolerance ±{HSV_CALIBRATION_BIN_TOLERANCE})")
    lines.append(f"- Strict gate: **{'PASS' if calibration['strict_passed'] else 'FAIL'}** · "
                 f"Fallback gate: **{'PASS' if calibration['fallback_passed'] else 'FAIL'}**")
    new_c = calibration["new_counts"]
    stored_c = calibration["stored_counts"]
    lines.append("- Bin comparison (new vs stored):")
    for cls in sorted({*new_c, *stored_c}):
        lines.append(f"  - `{cls}`: new={new_c.get(cls, 0)} stored={stored_c.get(cls, 0)}")
    lines.append("")
    lines.append("## Step 3 — Ranking (events 1–34)")
    lines.append("| Model | emb | mean | σ | norm/OSNet | bins (hs/bl/wr/ab) | pos% |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for name in CANDIDATES_ORDER:
        s = ranking_stats.get(name)
        if s is None:
            lines.append(f"| `{name}` | — | — | — | — | — | — |")
            continue
        bb = make_backbones().get(name)
        dim = bb.dim if bb else 0
        lines.append(
            f"| `{name}` | {dim} | {s.mean_signal:+.3f} | {s.std_signal:.3f} | "
            f"{s.normalized_mean:+.2f} | {_fmt_bin(s.bin_counts)} | "
            f"{s.proportion_positive:.1%} |"
        )
    lines.append("")
    lines.append("## Step 4 — Cross-rally rank-1")
    lines.append("| Model | rank-1 | n_queries |")
    lines.append("|---|---:|---:|")
    for name in CANDIDATES_ORDER:
        r1, n = cross_rally.get(name, (float("nan"), 0))
        lines.append(f"| `{name}` | {r1:.3f} | {n} |" if n else f"| `{name}` | — | 0 |")
    lines.append("")
    lines.append("## Step 5 — Held-out (events 35–58)")
    lines.append("| Model | mean | σ | norm/OSNet | bins | pos% |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for name in CANDIDATES_ORDER:
        s = heldout_stats.get(name)
        if s is None:
            lines.append(f"| `{name}` | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` | {s.mean_signal:+.3f} | {s.std_signal:.3f} | "
            f"{s.normalized_mean:+.2f} | {_fmt_bin(s.bin_counts)} | "
            f"{s.proportion_positive:.1%} |"
        )
    lines.append("")
    lines.append("## Verdict")
    winner = verdict.get("winner")
    if winner:
        lines.append(f"**Winner:** `{winner}` — clears ranking gates and held-out gate.")
        lines.append("")
        lines.append("Next: Session 4 (integration). Wire as additive cost term in "
                     "`_compute_assignment_cost`, weight sweep {0.10, 0.15, 0.20}.")
    else:
        lines.append(f"**No winner.** Reason: {verdict.get('reason', 'unspecified')}")
        lines.append("")
        lines.append("Next: Session 2 (harvest + visual verification), per "
                     "`memory/within_team_reid_project_2026_04_16.md`.")
    lines.append("")
    lines.append("### Per-model verdict detail")
    for name, det in verdict.get("decisions", {}).items():
        lines.append(f"- `{name}` → **{det.get('verdict', '?')}**"
                     + (f" ({', '.join(det.get('reasons', []))})" if det.get("reasons") else ""))
    lines.append("")
    md_path.write_text("\n".join(lines))
    logger.info("Wrote %s", md_path)
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/tracking_audit/reid_debug"),
        help="Where sota_probe.md lands (plus supporting JSONs).",
    )
    p.add_argument(
        "--limit-rallies",
        type=int,
        default=0,
        help="For smoke testing: probe only the first N rallies (0 = all).",
    )
    p.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated candidate names to skip (e.g. dinov2_vitl14).",
    )
    p.add_argument(
        "--calibration-only",
        action="store_true",
        help="Stop after Step 0 (HSV calibration).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    skips = {s.strip() for s in args.skip.split(",") if s.strip()}
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load all events ----------
    events = load_all_events(output_dir)
    logger.info("Loaded %d swap events from %s", len(events), output_dir)
    if not events:
        logger.error("No events found — aborting.")
        return 2

    # Optionally restrict to first N rallies (dev smoke test).
    if args.limit_rallies > 0:
        seen: set[str] = set()
        limited: list[SwapEvent] = []
        for e in events:
            if len(seen) >= args.limit_rallies and e.rally_id not in seen:
                continue
            seen.add(e.rally_id)
            limited.append(e)
        events = limited
        logger.info("  limited to %d rallies → %d events", args.limit_rallies, len(events))

    stored = load_stored_reid_debug(output_dir)
    ranking = events[:RANKING_SIZE]
    heldout = events[RANKING_SIZE:]
    logger.info("Split: %d ranking / %d held-out", len(ranking), len(heldout))

    # ---------- Step 0 — HSV calibration ----------
    passed, calibration = run_hsv_calibration(ranking, stored)
    if not passed:
        # Emit a minimal failure report so the run trail is preserved.
        out_path = output_dir / "sota_probe.md"
        out_path.write_text(
            "# SOTA ReID Probe — FAILED at Step 0 (HSV calibration)\n\n"
            f"Strict pass: {calibration['strict_passed']} · "
            f"Fallback pass: {calibration['fallback_passed']}\n\n"
            f"Aggregate new vs stored bin counts:\n"
            + json.dumps({"new": calibration["new_counts"],
                          "stored": calibration["stored_counts"]}, indent=2)
            + "\n\nSee log for per-event detail. Subsequent steps skipped."
        )
        (output_dir / "sota_probe_calibration.json").write_text(json.dumps(calibration, indent=2))
        return 1

    if args.calibration_only:
        logger.info("--calibration-only requested, stopping.")
        return 0

    # ---------- Step 1 — Model discovery ----------
    logger.info("Step 1 — Model discovery")
    backbones_map = make_backbones()
    discovery = discover_models(backbones_map)
    active_backbones: list[EmbedBackbone] = []
    skipped_models: dict[str, str] = {}
    for name in CANDIDATES_ORDER:
        if name == MODEL_HSV:
            continue
        if name in skips:
            skipped_models[name] = "skipped via --skip"
            continue
        bb = backbones_map[name]
        if discovery.get(name) == "missing":
            skipped_models[name] = "unavailable"
            continue
        active_backbones.append(bb)

    # ---------- Step 2 — Per-candidate probe on all 58 events ----------
    logger.info("Step 2 — Per-candidate probe on %d events × %d candidates",
                len(events), len(active_backbones))
    outcomes_by_model: dict[str, list[ProbeOutcome]] = {bb.name: [] for bb in active_backbones}
    events_by_rally = _group_events_by_rally(events)

    rally_count = len(events_by_rally)
    for idx, (rally_id, rally_events) in enumerate(events_by_rally.items(), start=1):
        logger.info("  [%d/%d] rally %s (%d events)",
                    idx, rally_count, rally_id[:8], len(rally_events))
        ctx = _fetch_rally_context(rally_id)
        if ctx is None:
            for ev in rally_events:
                for bb in active_backbones:
                    outcomes_by_model[bb.name].append(ProbeOutcome(
                        ev.rally_id, ev.swap_frame, ev.pred_old, ev.pred_new,
                        bb.name, None, 0, 0, 0, BIN_ABSTAIN,
                        abstain_reason="rally context missing",
                    ))
            continue
        rally_results = probe_events_for_rally(rally_events, ctx, active_backbones)
        for name, outs in rally_results.items():
            outcomes_by_model[name].extend(outs)

    # ---------- Step 3 — Ranking + σ bins ----------
    logger.info("Step 3 — Compute per-model statistics")
    # Outcomes are appended in the same order as `events` (sorted by rally_id,
    # swap_frame). Slice positionally so duplicates at the same (rally, frame)
    # key aren't collapsed.
    ranking_slice = slice(0, RANKING_SIZE)
    heldout_slice = slice(RANKING_SIZE, None)

    def slice_outcomes(model: str, evs: list[SwapEvent]) -> list[ProbeOutcome]:
        all_outs = outcomes_by_model[model]
        if len(all_outs) != len(events):
            # Defensive fallback: align by key (may collapse duplicates).
            by_key: dict[tuple[str, int, int, int], ProbeOutcome] = {
                (o.rally_id, o.swap_frame, o.pred_old, o.pred_new): o
                for o in all_outs
            }
            return [by_key[(e.rally_id, e.swap_frame, e.pred_old, e.pred_new)]
                    for e in evs
                    if (e.rally_id, e.swap_frame, e.pred_old, e.pred_new) in by_key]
        if evs is ranking:
            return all_outs[ranking_slice]
        if evs is heldout:
            return all_outs[heldout_slice]
        # Fallback: align by key.
        by_key = {(o.rally_id, o.swap_frame, o.pred_old, o.pred_new): o
                  for o in all_outs}
        return [by_key[(e.rally_id, e.swap_frame, e.pred_old, e.pred_new)]
                for e in evs
                if (e.rally_id, e.swap_frame, e.pred_old, e.pred_new) in by_key]

    # OSNet first to get the normalization baseline.
    osnet_rank_stats: ModelStats | None = None
    if MODEL_OSNET in outcomes_by_model:
        osnet_rank_stats = aggregate_stats(MODEL_OSNET, slice_outcomes(MODEL_OSNET, ranking), None)
    osnet_mean_rank = osnet_rank_stats.mean_signal if osnet_rank_stats else None
    osnet_mean_held: float | None = None

    ranking_stats: dict[str, ModelStats] = {}
    heldout_stats: dict[str, ModelStats] = {}
    for bb in active_backbones:
        ranking_stats[bb.name] = aggregate_stats(
            bb.name, slice_outcomes(bb.name, ranking), osnet_mean_rank,
        )
    if MODEL_OSNET in ranking_stats:
        osnet_held_stats = aggregate_stats(
            MODEL_OSNET, slice_outcomes(MODEL_OSNET, heldout), None,
        )
        osnet_mean_held = osnet_held_stats.mean_signal
    for bb in active_backbones:
        heldout_stats[bb.name] = aggregate_stats(
            bb.name, slice_outcomes(bb.name, heldout), osnet_mean_held,
        )

    # ---------- Step 4 — Cross-rally rank-1 ----------
    logger.info("Step 4 — Adversarial cross-rally rank-1")
    video_ids = sorted({e.video_id for e in events})
    gallery = build_cross_rally_gallery(video_ids, active_backbones)
    cross_rally: dict[str, tuple[float, int]] = {}
    for bb in active_backbones:
        r1, n = compute_rank1(gallery.get(bb.name, []))
        cross_rally[bb.name] = (r1, n)
        logger.info("  %s: rank-1=%.3f (n=%d)", bb.name, r1, n)

    # ---------- Step 6 — Verdict + Step 5 baked into decide_winner ----------
    verdict = decide_winner(ranking_stats, heldout_stats, cross_rally)

    # ---------- Persist ----------
    all_outcomes = [o.to_dict() for outs in outcomes_by_model.values() for o in outs]
    (output_dir / "sota_probe_events.json").write_text(json.dumps(all_outcomes, indent=2))
    (output_dir / "sota_probe_cross_rally.json").write_text(json.dumps(
        {
            name: {
                "rank1": r1,
                "n_queries": n,
                "entries": [
                    {
                        "video_id": e.video_id,
                        "rally_id": e.rally_id,
                        "canonical_id": e.canonical_id,
                    } for e in gallery.get(name, [])
                ],
            }
            for name, (r1, n) in cross_rally.items()
        },
        indent=2,
    ))
    (output_dir / "sota_probe_calibration.json").write_text(json.dumps(calibration, indent=2))

    wall = time.time() - t0
    write_report(
        output_dir=output_dir,
        events_total=len(events),
        ranking_size=len(ranking),
        heldout_size=len(heldout),
        discovery=discovery,
        calibration=calibration,
        ranking_stats=ranking_stats,
        heldout_stats=heldout_stats,
        cross_rally=cross_rally,
        verdict=verdict,
        wall_seconds=wall,
        skipped_models=skipped_models,
    )
    logger.info("Total wall time: %.1f min", wall / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
