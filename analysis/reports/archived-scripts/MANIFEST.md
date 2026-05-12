# Archived diagnostic scripts

Scripts moved here from `analysis/scripts/` because they belong to closed investigations and are no longer part of any current pipeline. Live scripts in `analysis/scripts/` should remain focused on canonical / routinely-invoked workflows.

> If a script needs to come back, restore it with `git mv reports/archived-scripts/<dir>/<file>.py scripts/<file>.py` — git history is preserved.

## ball_3d_abandoned/

Status: ABANDONED. The 3D-ball-trajectory workstream was closed in April 2026 after multiple sessions failed to clear the gating criteria.

See: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` topic-files entry for ball ("3D abandoned"), and `MEMORY_ARCHIVE.md` "Earlier dead-ends and abandoned research" for the linked NO-GO memos (`ball_3d_phase_c_audit_2026_04_11.md`, `ball_3d_sota_research_brief_2026_04_11.md`, etc.).

| Script | Purpose |
|---|---|
| `analyze_ball_3d_by_camera_height.py` | Stratified 3D-trajectory error by camera height |
| `analyze_ball_3d_session_short.py` | Per-session 3D feasibility summary |
| `audit_ball_3d_gt_contacts.py` | GT-contact audit for 3D ball workstream |
| `audit_ball_3d_tier1.py` | Tier-1 calibration audit (values referenced as a comment in `landing_detector.py`) |
| `audit_flight_time_vs_fitter.py` | Flight-time vs fitter agreement check |
| `debug_ball_3d.py` | Single-rally 3D debug |
| `diagnose_ball_3d_fitter.py` | Per-rally fitter diagnostic |
| `eval_ball_3d.py` | 3D ball eval harness |
| `select_ball_3d_audit_rallies.py` | Audit-rally selection tool |
| `verify_camera_height_from_players.py` | Camera-height inference probe |
| `visualize_ball_3d_rig.py` | Rig visualization for 3D ball workstream |

## botsort_forensic_2026_04_30/

Status: Investigation closed; chimera-stitching 2-layer fix SHIPPED 2026-05-05 (commits `987f572` + `5ae71f8`, MATCHER_VERSION→v7). The forensic harness was used to capture per-frame BoxMOT internal state during the diagnosis but is not part of any current pipeline.

Activated only when env var `BOTSORT_FORENSIC_LOG_DIR` is set; otherwise behavior is byte-identical to stock BoT-SORT.

See: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/chimera_stitching_dd042609_2026_05_04.md`.

| File | Purpose |
|---|---|
| `botsort_instrumented.py` | Read-only `BotSort` subclass capturing IoU + embedding cost matrices, Hungarian matches, lost/removed track lifecycle to JSONL sidecars. |
| `analyze_botsort_forensic.py` | Aggregates the JSONL sidecars across rallies into investigation summaries. |
| `forensic_unit_test.py` | Synthetic-detection unit test for the instrumented wrapper. |
| `forensic_smoke_test.py` / `forensic_capture_panel.py` / `forensic_panel_ground_truth.py` / `forensic_user_chimera_cases.py` | Capture and reproduction harnesses across panel + user-reported chimera cases. |

## occlusion_resolver_session5/

Status: NO-GO 2026-04-17 (Session 5 of within-team ReID workstream). The post-hoc per-convergence within-team swap resolver couldn't separate the one labelled positive from the no-swap negatives at the required precision (≥0.95 / ≥0.5 recall) given the available feature set. Module + tests deleted on 2026-05-08; these reproduce the labelling pipeline if anyone wants to revisit with a richer feature set or larger labelled corpus.

See: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY_ARCHIVE.md` "Within-Team ReID Sessions 4–9" section, plus the in-tree `analysis/reports/occlusion_resolver/session5_gate_report.md`.

| Script | Purpose |
|---|---|
| `enumerate_convergence_events.py` | Walks GT rallies → same-team primary convergences → emits `events.json` |
| `render_occlusion_labeller.py` | Generates the static HTML labelling UI from `events.json` |
| `eval_occlusion_resolver.py` | Scores labelled events against the resolver, grid-searches thresholds, optional 43-rally A/B sweep |

## class_a_investigation/

Status: NO-GO 2026-05-06. Class A panel video (5c756c41) topped out at PERMUTED 93.8 even with permuted/oracle assignment; cost-matrix prefers wrong assignment due to feature-space ceiling, not matcher logic. Re-opening requires a new appearance signal — not new diagnostic scripts.

See: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/class_a_NOGO_2026_05_06.md`.

| Script | Purpose |
|---|---|
| `diag_class_a_partition_signal.py` | Partition-signal probe across r10 candidates |
| `diag_r10_cluster_members.py` | Cluster-membership dump for rally r10 |
| `diag_r10_cost_matrix.py` | Cost-matrix dump (the headline diagnostic) |
| `diag_r10_court_sides.py` | Court-side breakdown for r10 |
| `diag_r10_partitions.py` | Partition enumeration for r10 |

## action_gt_legacy/

Status: SUPERSEDED 2026-05-12. These scripts read from or wrote to
`player_tracks.action_ground_truth_json` (the legacy JSON-array column). After the
`rally_action_ground_truth` table was introduced and all analysis scripts migrated to
`action_gt_query.load_for_rallies/load_for_videos`, these helpers became obsolete.

| Script | Purpose |
|---|---|
| `backfill_action_gt_trackid.py` | Backfilled `trackId` field on legacy `action_ground_truth_json` labels |
| `resave_ball_for_action_gt.py` | Re-saved ball positions for rallies with `action_ground_truth_json` |
| `repair_orphaned_gt.py` | Read-modify-write repairs on `action_ground_truth_json` column (tangled with the column — superseded by the new table) |
| `restore_canonical_drift_backups.py` | Restored `action_ground_truth_json` from backup JSON files (column-writer, no longer valid after table migration) |

## Follow-ups (not in this pass)

The `analysis/scripts/` directory still has ~370 tracked + ~100 untracked scripts, many of which are likely candidates for archival (chimera-stitching probes, within-rally repair probes, ReID/DINOv2 archaeology, individual phase-investigation scripts, etc.). Triaging them needs per-script verification that they aren't referenced by some long-running cron / agent / skill, so they're deferred to a follow-up sweep — not because they're necessarily live, but because the cost of incorrectly archiving a canonical script is high.
