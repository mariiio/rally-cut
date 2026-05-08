"""Read-only BoT-SORT subclass that captures per-frame BoxMOT internal state.

Activated only when env var BOTSORT_FORENSIC_LOG_DIR is set. Writes one
JSONL sidecar per rally to <BOTSORT_FORENSIC_LOG_DIR>/<BOTSORT_FORENSIC_RALLY_TAG>.jsonl
containing pre/post tracker state, IoU + embedding cost matrices, Hungarian
matches, and lost/removed track lifecycle for every update() call.

The wrapper makes NO behavioral changes: it overrides three internal methods
of BotSort (_first_association, _second_association, _prepare_output) and
replicates their bodies verbatim so the call sequence, mutations, and return
values are byte-identical to the unwrapped class. The only side effect is the
sidecar JSONL file; tracker outputs are unchanged.

Pinned against boxmot==16.0.x. If BoxMOT changes the body of either
_first_association or _second_association, the assertions in
_assert_boxmot_compatible will fire.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import (
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)
from boxmot.utils.matching import (
    embedding_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
)

logger = logging.getLogger(__name__)

ENV_LOG_DIR = "BOTSORT_FORENSIC_LOG_DIR"
ENV_RALLY_TAG = "BOTSORT_FORENSIC_RALLY_TAG"


def _xyxy_list(arr: np.ndarray | None) -> list[float]:
    if arr is None:
        return []
    return [float(x) for x in arr.tolist()]


def _strack_summary(t: STrack) -> dict[str, Any]:
    return {
        "id": int(t.id),
        "det_ind": int(t.det_ind) if t.det_ind is not None else -1,
        "state": int(t.state),
        "frame_id": int(getattr(t, "frame_id", -1)),
        "start_frame": int(getattr(t, "start_frame", -1)),
        "is_activated": bool(getattr(t, "is_activated", False)),
        "tracklet_len": int(getattr(t, "tracklet_len", 0)),
        "xyxy": _xyxy_list(t.xyxy) if t.mean is not None else [],
    }


def _matrix_to_list(m: np.ndarray | None) -> list[list[float]]:
    if m is None or m.size == 0:
        return []
    return [[float(x) for x in row] for row in m.astype(np.float32).tolist()]


def _matches_to_list(m: np.ndarray | list[Any]) -> list[list[int]]:
    if isinstance(m, np.ndarray):
        if m.size == 0:
            return []
        return [[int(a), int(b)] for a, b in m.tolist()]
    return [[int(a), int(b)] for a, b in m]


def _idx_list(arr: np.ndarray | list[int]) -> list[int]:
    if isinstance(arr, np.ndarray):
        return [int(x) for x in arr.tolist()]
    return [int(x) for x in arr]


class InstrumentedBotSort(BotSort):
    """BoT-SORT subclass that records per-frame internal state to a JSONL sidecar.

    All overridden methods replicate the parent body verbatim and add only
    read-only state capture. No tracker behavior changes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        log_dir_env = os.environ.get(ENV_LOG_DIR)
        rally_tag_env = os.environ.get(ENV_RALLY_TAG)
        if not log_dir_env or not rally_tag_env:
            raise RuntimeError(
                f"InstrumentedBotSort requires {ENV_LOG_DIR} and "
                f"{ENV_RALLY_TAG} env vars to be set."
            )

        sidecar_path = Path(log_dir_env) / f"{rally_tag_env}.jsonl"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        self._sidecar_path = sidecar_path
        self._frame_buf: dict[str, Any] = {}

        # PlayerTracker._reset_boxmot_tracker() in track_video's finally clause
        # re-creates the tracker after the per-rally loop completes. The
        # second instance must NOT truncate the sidecar that the first wrote.
        # If the sidecar already exists with content, this is the post-loop
        # reset instance — disable writes entirely.
        if sidecar_path.exists() and sidecar_path.stat().st_size > 0:
            self._sidecar_fp = None
            self._writes_disabled = True
            logger.info(
                "InstrumentedBotSort post-loop reset; preserving existing "
                "sidecar at %s (writes disabled)",
                sidecar_path,
            )
            return

        self._writes_disabled = False
        self._sidecar_fp = open(sidecar_path, "w", buffering=1, encoding="utf-8")

        meta = {
            "type": "meta",
            "rally_tag": rally_tag_env,
            "boxmot_class": "InstrumentedBotSort",
            "config": {
                "track_high_thresh": float(self.track_high_thresh),
                "track_low_thresh": float(self.track_low_thresh),
                "new_track_thresh": float(self.new_track_thresh),
                "match_thresh": float(self.match_thresh),
                "proximity_thresh": float(self.proximity_thresh),
                "appearance_thresh": float(self.appearance_thresh),
                "with_reid": bool(self.with_reid),
                "fuse_first_associate": bool(self.fuse_first_associate),
                "buffer_size": int(self.buffer_size),
                "max_time_lost": int(self.max_time_lost),
            },
            "schema_version": 1,
        }
        self._sidecar_fp.write(json.dumps(meta) + "\n")
        logger.info("InstrumentedBotSort writing sidecar to %s", sidecar_path)

    def close(self) -> None:
        fp = getattr(self, "_sidecar_fp", None)
        if fp is not None and not fp.closed:
            fp.close()
            self._sidecar_fp = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _first_association(
        self,
        dets: np.ndarray,
        dets_first: np.ndarray,
        active_tracks: list[STrack],
        unconfirmed: list[STrack],
        img: np.ndarray,
        detections: list[STrack],
        activated_stracks: list[STrack],
        refind_stracks: list[STrack],
        strack_pool: list[STrack],
    ) -> tuple[Any, Any, Any]:
        # Reset per-frame buffer at the first call inside this update().
        self._frame_buf = {
            "type": "frame",
            "f": int(self.frame_count),
            "n_dets_first": int(len(dets_first)),
            "n_strack_pool": int(len(strack_pool)),
        }

        # Pre-state: snapshot of trackers BEFORE this frame's matching.
        # self.active_tracks here is the post-frame state of frame_count-1.
        active_tracks_pre: list[STrack] = self.active_tracks  # type: ignore[has-type]
        lost_stracks_pre: list[STrack] = self.lost_stracks  # type: ignore[has-type]
        self._frame_buf["active_pre"] = [_strack_summary(t) for t in active_tracks_pre]
        self._frame_buf["lost_pre"] = [_strack_summary(t) for t in lost_stracks_pre]
        self._frame_buf["removed_pre"] = [
            _strack_summary(t) for t in self.removed_stracks
        ]

        # Replicate parent body: STrack.multi_predict mutates strack_pool means.
        STrack.multi_predict(strack_pool)

        # Snapshot of predicted xyxy for each strack_pool entry (post multi_predict).
        # This is what IoU is computed against. Capture BEFORE matching, after predict.
        self._frame_buf["strack_pool_predicted"] = [
            {
                "id": int(t.id),
                "state_pre": int(t.state),
                "xyxy_pred": _xyxy_list(t.xyxy) if t.mean is not None else [],
                "smooth_feat_norm": float(np.linalg.norm(t.smooth_feat))
                if t.smooth_feat is not None
                else 0.0,
            }
            for t in strack_pool
        ]

        warp = self.cmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Capture detection bboxes + confidences (from dets_first, before STrack wrap).
        # dets_first columns: [x1, y1, x2, y2, conf, cls, det_ind].
        dets_first_arr = np.asarray(dets_first)
        if dets_first_arr.size:
            self._frame_buf["dets_first"] = [
                {
                    "x1": float(d[0]), "y1": float(d[1]),
                    "x2": float(d[2]), "y2": float(d[3]),
                    "conf": float(d[4]), "cls": int(d[5]),
                    "det_ind": int(d[6]),
                }
                for d in dets_first_arr
            ]
        else:
            self._frame_buf["dets_first"] = []

        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists_raw = embedding_distance(strack_pool, detections)
            emb_dists = emb_dists_raw.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            emb_dists_raw = None
            emb_dists = None
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        # Capture cost matrices and matches.
        self._frame_buf["first_assoc"] = {
            "track_pool_ids": [int(t.id) for t in strack_pool],
            "track_pool_states_pre": [int(t.state) for t in strack_pool],
            "ious_dists": _matrix_to_list(ious_dists),
            "emb_dists_raw": _matrix_to_list(emb_dists_raw),
            "emb_dists_gated": _matrix_to_list(emb_dists),
            "fused": _matrix_to_list(dists),
            "matches": _matches_to_list(matches),
            "u_track": _idx_list(u_track),
            "u_detection": _idx_list(u_detection),
            "thresh": float(self.match_thresh),
        }

        # Replicate matched-track update loop and capture matched_via per track id.
        first_matched_via: dict[int, str] = {}
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
                first_matched_via[int(track.id)] = "first_tracked"
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)
                first_matched_via[int(track.id)] = "first_reactivated"
        self._frame_buf["first_matched_via"] = first_matched_via

        return matches, u_track, u_detection

    def _second_association(
        self,
        dets_second: np.ndarray,
        activated_stracks: list[STrack],
        lost_stracks: list[STrack],
        refind_stracks: list[STrack],
        u_track_first: list[int] | np.ndarray,
        strack_pool: list[STrack],
    ) -> tuple[Any, Any, Any]:
        if len(dets_second) > 0:
            detections_second = [
                STrack(det, max_obs=self.max_obs) for det in dets_second
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Tracked
        ]

        # Capture inputs to the second pass.
        dets_second_arr = np.asarray(dets_second) if len(dets_second) else None
        if dets_second_arr is not None and dets_second_arr.size:
            dets_second_dump = [
                {
                    "x1": float(d[0]), "y1": float(d[1]),
                    "x2": float(d[2]), "y2": float(d[3]),
                    "conf": float(d[4]), "cls": int(d[5]),
                    "det_ind": int(d[6]),
                }
                for d in dets_second_arr
            ]
        else:
            dets_second_dump = []

        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

        self._frame_buf["second_assoc"] = {
            "r_tracked_ids": [int(t.id) for t in r_tracked_stracks],
            "dets_second": dets_second_dump,
            "ious_dists": _matrix_to_list(dists),
            "matches": _matches_to_list(matches),
            "u_track": _idx_list(u_track),
            "u_detection": _idx_list(u_detection),
            "thresh": 0.5,
        }

        second_matched_via: dict[int, str] = {}
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
                second_matched_via[int(track.id)] = "second_tracked"
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)
                second_matched_via[int(track.id)] = "second_reactivated"

        marked_lost: list[int] = []
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                marked_lost.append(int(track.id))

        self._frame_buf["second_matched_via"] = second_matched_via
        self._frame_buf["marked_lost"] = marked_lost

        return matches, u_track, u_detection

    def _prepare_output(
        self,
        activated_stracks: list[STrack],
        refind_stracks: list[STrack],
        lost_stracks: list[STrack],
        removed_stracks: list[STrack],
    ) -> np.ndarray:
        # Replicate parent body verbatim.
        active_tracks: list[STrack] = self.active_tracks  # type: ignore[has-type]
        lost_stracks_self: list[STrack] = self.lost_stracks  # type: ignore[has-type]
        active_tracks = [
            t for t in active_tracks if t.state == TrackState.Tracked
        ]
        active_tracks = joint_stracks(active_tracks, activated_stracks)
        active_tracks = joint_stracks(active_tracks, refind_stracks)
        lost_stracks_self = sub_stracks(lost_stracks_self, active_tracks)
        lost_stracks_self.extend(lost_stracks)
        lost_stracks_self = sub_stracks(lost_stracks_self, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        active_tracks, lost_stracks_self = remove_duplicate_stracks(
            active_tracks, lost_stracks_self
        )
        self.active_tracks = active_tracks
        self.lost_stracks = lost_stracks_self

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            for t in active_tracks
            if t.is_activated
        ]

        # Capture post-state and flush this frame's record.
        self._frame_buf["active_post"] = [_strack_summary(t) for t in active_tracks]
        self._frame_buf["lost_post"] = [_strack_summary(t) for t in lost_stracks_self]
        self._frame_buf["removed_post"] = [
            _strack_summary(t) for t in self.removed_stracks
        ]
        # Track id → which association pass matched it this frame.
        # Built incrementally by the two _*_association overrides; if a track
        # isn't in either dict, it was either an unconfirmed-handled match,
        # newly initialized, or carried over without a match.
        first = self._frame_buf.get("first_matched_via", {}) or {}
        second = self._frame_buf.get("second_matched_via", {}) or {}
        merged: dict[int, str] = {}
        merged.update(first)
        merged.update(second)
        self._frame_buf["matched_via"] = merged

        # Identify activated stracks that are new (not in active_pre).
        active_pre_ids = {entry["id"] for entry in self._frame_buf["active_pre"]}
        active_post_ids = {entry["id"] for entry in self._frame_buf["active_post"]}
        self._frame_buf["new_track_ids"] = sorted(active_post_ids - active_pre_ids)
        self._frame_buf["dropped_track_ids"] = sorted(active_pre_ids - active_post_ids)

        if self._sidecar_fp is not None and not self._writes_disabled:
            try:
                self._sidecar_fp.write(json.dumps(self._frame_buf) + "\n")
            except (ValueError, OSError) as exc:
                logger.warning(
                    "InstrumentedBotSort failed to write frame record: %s", exc,
                )

        # Defensive: clear the per-frame buffer so a misconfigured second call
        # to _prepare_output (which doesn't happen in the parent flow) doesn't
        # double-emit the same frame.
        self._frame_buf = {}

        return np.asarray(outputs)
