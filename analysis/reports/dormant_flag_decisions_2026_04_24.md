# Dormant Flag Decisions — 2026-04-24

Phase 0.3 of the attribution primitive-first rebuild. Inventory of every env-gated
research flag currently in `analysis/rallycut/`, its origin, the memory/report that
closed it, and a ship-time decision (REMOVE / KEEP-DORMANT / PROMOTE).

| Flag | Default | Files | Verdict | Decision |
|---|---|---|---|---|
| `MATCH_TRACKER_GLOBAL_SEED` | `0` | `tracking/match_tracker.py:113` | Phase-3 global seed probe. Superseded by Day-4 ref-crop solution (95.22% identity); the hypothesis the flag was trying to synthesize is now delivered by ref crops. | **REMOVE** |
| `WEIGHT_LEARNED_REID` | `0.0` | `tracking/global_identity.py:69`, `tracking/player_tracker.py:2091`, `cli/commands/evaluate_tracking.py:477`, `scripts/probe_anchor_viability.py:34`, `scripts/debug_learned_reid_trace.py:28`, `scripts/eval_occlusion_resolver.py:114` | Session-4 within-team ReID additive cost. NO-GO (memory `within_team_reid_project_2026_04_16.md`). | **REMOVE** |
| `ENABLE_OCCLUSION_RESOLVER` | `0` | `tracking/player_tracker.py:1741`, `cli/commands/evaluate_tracking.py:498`, `scripts/probe_anchor_viability.py:36` | Session-5 post-hoc occlusion resolver. NO-GO (feature-set insufficient at N=1 positive; `within_team_reid_project_2026_04_16.md`). | **REMOVE** |
| `LEARNED_MERGE_VETO_COS` | `0.0` | `tracking/merge_veto.py:28`, `cli/commands/evaluate_tracking.py:508` | Session-6 learned-head cosine veto. Underpowered vs ≥50% kill gate (best 25%). NO-GO. | **REMOVE** |
| `SKIP_ALL_MERGE_PASSES` | `0` | `tracking/player_tracker.py:1535` | Session-7 minimal-merge experiment. Confirmed merge-chain KEEP (HOTA dropped −5.2pp under minimal). Flag only existed to run the A/B. | **REMOVE** |
| `ENABLE_COURT_VELOCITY_GATE` | `0` | `tracking/player_tracker.py:1630` | Court-plane velocity gate. NO-GO 2026-04-16 (no sweep cell cleared 0.5pp regression gate). | **REMOVE** |
| `RALLYCUT_DISABLE_COURT_VELOCITY_GATE` | `0` | `tracking/tracklet_link.py:181` | Companion disable switch to above. | **REMOVE** (with the gate itself) |
| `RALLYCUT_COURT_GATE_ADDITIVE` | `0` | `tracking/tracklet_link.py:189` | Additive-cost mode for court-velocity gate. | **REMOVE** (with the gate itself) |
| `RALLYCUT_MAX_MERGE_VELOCITY_METERS` | `2.5` | `tracking/tracklet_link.py:60` | Active tracklet-link merge velocity cap. Not dormant — production config. | **KEEP** (production) |

## Aspirational flags named in memory but never merged

Memory's dormant-flag index lists `TEAM_GATED_ATTRIBUTION`, `JOINT_DECODE_IDENTITY`, and `CROP_HEAD_*` env vars.
Grep confirms none of these are present in the codebase as environment-variable gates:

- **`TEAM_GATED_ATTRIBUTION`** — never landed as an env flag. Attribution team-gate
  experiment (2026-04-24 NO-GO) was ad-hoc code paths removed same session.
- **`JOINT_DECODE_IDENTITY`** — same; never env-gated.
- **`CROP_HEAD_*`** — crop-head code lives at `rallycut/ml/crop_head/` + `rallycut/tracking/crop_head_emitter.py` but is invoked via explicit CLI flag (`--emitter=crop_head`), not env var. Phase-2 NO-GO 2026-04-20.

**Action:** update memory to drop these from the dormant-flag list. They were never in code.

## Code removal summary

Pending user sign-off, 8 env flags above marked REMOVE will be deleted in a cleanup
commit before Phase 1 starts. Net impact:

- Net lines deleted: ~150–250 (flag check + guarded code paths).
- No production behavior change (all flags default-off in current ship).
- Simplifies Phase-1 audit: remove noise so `tracking/match_tracker.py`,
  `tracking/global_identity.py`, `tracking/player_tracker.py`,
  `tracking/tracklet_link.py`, and `tracking/merge_veto.py` read cleanly.

## Not in scope for removal

- `AWS_*`, `S3_*`, `DATABASE_URL`, `MINIO_*`, `MODAL_*` — infra config.
- `ONNX_NUM_THREADS`, `ULTRALYTICS_*`, `YOLO_AUTOCHECK` — runtime tuning.
- `LABEL_STUDIO_API_KEY` — tooling.
- `TRAINING_S3_*` — dataset backup.

These are infrastructure, not research dormancy.
