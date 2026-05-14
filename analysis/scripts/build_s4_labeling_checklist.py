"""Stratify S4 flip candidates + emit a user-facing labeling checklist.

Reads `s4_fleet_candidates.json` from the fleet sweep. Picks 25-30 cases
stratified across these shape buckets:

  - ATTACK->ATTACK same-team (the failure-mode bucket). Take ALL because
    there are only 10. The volleyball heuristic "no two attacks by same
    player" might be wrong: 2 attempted attacks in a row can happen.
  - same-team SET / RECEIVE / DIG triples (the natural cascade pattern).
  - cross-team (opposing-team-prev) flips (anti-self-touch no-op; S4 = S3
    trajectory-integral disagreement only).
  - "random" — round out the sample with a stratified random pick from the
    remaining flips.
  - PLUS the 2 probe-B cascade cases (titi/a0881d82 f128 + f225) as
    anchor points already in the user's mental model.

Caps at <=3 cases per video so the labeling spans the fleet broadly.

Writes:
  analysis/reports/probe_b_sequence_aware/2026_05_14/s4_labeling_checklist.md

Usage:
    cd analysis
    uv run python scripts/build_s4_labeling_checklist.py
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent / "reports" / "probe_b_sequence_aware" / "2026_05_14"
DEFAULT_INPUT = REPORT_DIR / "s4_fleet_candidates.json"
DEFAULT_OUTPUT = REPORT_DIR / "s4_labeling_checklist.md"

TARGET_TOTAL = 28           # aim for 25-30
PER_VIDEO_CAP = 3           # at most N picks per source video
RANDOM_SEED = 42            # reproducible


@dataclass
class Pick:
    bucket: str
    flip: dict[str, Any]


def _ms_to_clock(ms: int) -> str:
    total_s = ms / 1000.0
    m = int(total_s // 60)
    s = total_s - m * 60
    return f"{m}:{s:06.3f}"


def _source_time(flip: dict[str, Any]) -> str:
    """Convert (rally_start_ms + frame/fps) to mm:ss.SSS source-video time."""
    rally_start_s = flip["rally_start_ms"] / 1000.0
    frame_offset_s = flip["pl_frame"] / max(flip["fps"], 1e-6)
    total_s = rally_start_s + frame_offset_s
    m = int(total_s // 60)
    s = total_s - m * 60
    return f"{m}:{s:06.3f}"


def _classify_bucket(f: dict[str, Any]) -> str:
    """Classify a flip into a stratification bucket.

    Failure-mode hierarchy (anti-self-touch fires ON):
      - attack_after_attack: ATTACK<-ATTACK same-team (rule may be wrong: 2
        attempts can legit happen)
      - attack_after_set:    ATTACK<-SET same-team (classic setter-also-attacker
        pipeline confusion — biggest "other" bucket, 39 cases)
      - same_team_chain:     SET/RECEIVE/DIG <- same-team triple (175 cases)
      - same_team_other:     any other same-team-chain action_type (recovery dig
        after own attack, etc.)
    Other:
      - cross_team_prev:     prev was opposite team — anti-self is no-op
      - same_team_no_self:   same-team chain but pipeline pick != prev_toucher
    """
    a_type = f["action_type"]
    prev = f["prev_action_type"]
    same_team_chain = f["pipeline_team"] == f["prev_toucher_team"]
    anti_self_fires = (
        f["pipeline_pid"] == f["prev_toucher_pid"]
        and prev != "BLOCK"
    )
    if not same_team_chain:
        return "cross_team_prev"
    if same_team_chain and not anti_self_fires:
        return "same_team_no_self"
    # Anti-self-touch fires + same-team chain — slice by action type.
    if a_type == "ATTACK" and prev == "ATTACK":
        return "attack_after_attack"
    if a_type == "ATTACK" and prev == "SET":
        return "attack_after_set"
    if a_type in ("SET", "RECEIVE", "DIG"):
        return "same_team_chain"
    return "same_team_other"


def _alt_ratio(f: dict[str, Any]) -> float:
    """How much trajectory-integral prefers s4 over pipeline. >1 means s4 is closer."""
    integrals = f["s3_integrals"]
    pl = float(integrals[str(f["pipeline_pid"])])
    s4 = float(integrals[str(f["s4_pid"])])
    if s4 <= 0:
        return float("inf")
    return pl / s4


def _stratified_pick(
    flips: list[dict[str, Any]],
    *,
    seed: int,
) -> list[Pick]:
    """Sample 25-30 cases stratified by bucket, capped at 3 per video.
    Always includes titi/a0881d82 f128 + f225 (probe-B cascade anchors)."""

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in flips:
        by_bucket[_classify_bucket(f)].append(f)

    rng = random.Random(seed)

    # Targets per bucket (sums to ~28 including 2 probe-B anchors):
    #   - attack_after_attack: 7 (slice the 10 — failure-mode bucket, take most)
    #   - attack_after_set:    7 (39-case bucket — biggest within-team flip type)
    #   - same_team_chain:     6 (SET/RECEIVE/DIG <- same-team triple)
    #   - same_team_other:     2 (e.g. ATTACK<-DIG or BLOCK<-SET on same team)
    #   - cross_team_prev:     4 (anti-self no-op; S3-only flip)
    targets = {
        "attack_after_attack": 7,
        "attack_after_set": 7,
        "same_team_chain": 6,
        "same_team_other": 2,
        "cross_team_prev": 4,
    }

    picks: list[Pick] = []
    per_video: dict[str, int] = defaultdict(int)
    seen_keys: set[tuple[str, int]] = set()

    def _try_add(f: dict[str, Any], bucket: str) -> bool:
        key = (f["rally_id"], f["pl_frame"])
        if key in seen_keys:
            return False
        if per_video[f["video_name"]] >= PER_VIDEO_CAP:
            return False
        picks.append(Pick(bucket=bucket, flip=f))
        per_video[f["video_name"]] += 1
        seen_keys.add(key)
        return True

    # Anchor: ALWAYS include titi/a0881d82 f128 + f225.
    for f in flips:
        if f["rally_short"] == "a0881d82" and f["pl_frame"] in (128, 225):
            _try_add(f, "probe_b_cascade_anchor")

    def _sample_bucket(bucket: str, target: int, *, rank_key=None) -> int:
        """Sample up to `target` items from a bucket. If rank_key given, sort
        by that key descending first (then take from top); else shuffle."""
        cands = by_bucket.get(bucket, [])
        # Filter out already-picked
        cands = [f for f in cands if (f["rally_id"], f["pl_frame"]) not in seen_keys]
        if rank_key:
            cands.sort(key=rank_key, reverse=True)
        else:
            rng.shuffle(cands)
        added = 0
        for f in cands:
            if added >= target or len(picks) >= TARGET_TOTAL:
                break
            if _try_add(f, bucket):
                added += 1
        return added

    # Stratified bucket sampling — prefer high alt_ratio (more decisive S4
    # disagreement) so user labels the strongest signal cases first.
    _sample_bucket("attack_after_attack",
                   targets["attack_after_attack"], rank_key=_alt_ratio)
    _sample_bucket("attack_after_set",
                   targets["attack_after_set"], rank_key=_alt_ratio)
    _sample_bucket("same_team_chain",
                   targets["same_team_chain"], rank_key=_alt_ratio)
    _sample_bucket("same_team_other",
                   targets["same_team_other"], rank_key=_alt_ratio)
    _sample_bucket("cross_team_prev",
                   targets["cross_team_prev"], rank_key=_alt_ratio)

    # Round-out with random remaining flips.
    remaining = [
        f for f in flips
        if (f["rally_id"], f["pl_frame"]) not in seen_keys
        and per_video[f["video_name"]] < PER_VIDEO_CAP
    ]
    rng.shuffle(remaining)
    for f in remaining:
        if len(picks) >= TARGET_TOTAL:
            break
        _try_add(f, "random_roundout")

    return picks


def _write_markdown(picks: list[Pick], all_flips_count: int, output: Path) -> None:
    bucket_counts: dict[str, int] = defaultdict(int)
    for p in picks:
        bucket_counts[p.bucket] += 1

    lines: list[str] = []
    lines.append("# S4 (anti-self-touch + trajectory-integral) — labeling checklist")
    lines.append("")
    lines.append("**Goal:** confirm whether each of S4's fleet-wide attribution flips matches")
    lines.append("the actual toucher at the contact frame. Open each rally in the rally editor,")
    lines.append("scrub to the source-video time, look at who is touching the ball, and:")
    lines.append("")
    lines.append("- If **S4's proposed pick** is the actual toucher: leave the suggested change to s4_pid.")
    lines.append("- If **pipeline's pick** (the current attribution) is correct: confirm it / leave unchanged.")
    lines.append("- If **neither** is correct: change to whoever actually touched the ball.")
    lines.append("- If the contact is a **false positive** (no real ball touch on that frame): delete the action.")
    lines.append("")
    lines.append("The editor writes to `rally_action_ground_truth`. When you finish, tell me **\"done with labeling\"**.")
    lines.append("")
    lines.append(f"**Total fleet S4 flip candidates:** {all_flips_count} across 71 videos.")
    lines.append(f"**This sample:** {len(picks)} cases stratified across buckets:")
    lines.append("")
    for b, c in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- `{b}`: {c}")
    lines.append("")
    lines.append("**Stratification notes (anti-self-touch fires unless noted):**")
    lines.append("")
    lines.append("- `probe_b_cascade_anchor` — the two titi/a0881d82 frames from Probe B (f128, f225). Already in your mental model.")
    lines.append("- `attack_after_attack` — same-team ATTACK<-ATTACK flips (10 fleet-wide, we sample 7). The highest-risk failure mode for S4: \"same player can't attack twice in a row\" is heuristic, not a hard volleyball rule (recovery touch + spike CAN happen). If S4 wins here it's defensible; if it loses on most cases, the rule itself is questionable.")
    lines.append("- `attack_after_set` — same-team ATTACK<-SET flips (39 fleet-wide, we sample 7). Classic setter-also-attacker pipeline confusion. This is the biggest within-team flip type and the most likely place S4 helps.")
    lines.append("- `same_team_chain` — same-team SET/RECEIVE/DIG <- prev-same-team contact (175 fleet-wide, we sample 6). Sampled by descending alt_ratio (most decisive S4 trajectory-integral disagreement).")
    lines.append("- `same_team_other` — same-team chain with other action_type pairs (31 fleet-wide, e.g. BLOCK<-SET, ATTACK<-DIG, ATTACK<-RECEIVE; we sample 2).")
    lines.append("- `cross_team_prev` — previous action was OPPOSITE team (140 fleet-wide, we sample 4). Anti-self-touch is a no-op here; S4 differs from pipeline purely because trajectory-integral disagrees with ball-position-at-contact. Example: DIG after an opposing ATTACK.")
    lines.append("")
    lines.append("**Time estimate:** ~30-45 minutes if you spend ~60s per case (browser navigation + frame inspection).")
    lines.append("")
    lines.append("| # | Bucket | Video | Rally order | Source time | Frame | Action | Prev | Pipeline pick | S4 proposed | Rally UUID | Verdict |")
    lines.append("|--:|:-------|:------|------------:|------------:|------:|:-------|:-----|:-------------|:------------|:-----------|:-------:|")
    for i, p in enumerate(picks, 1):
        f = p.flip
        time_str = _source_time(f)
        pl_str = f"p{f['pipeline_pid']} ({f['pipeline_team']})"
        s4_str = f"p{f['s4_pid']} ({f['s4_team']})"
        prev_str = f"f{f['prev_action_frame']} {f['prev_action_type']} p{f['prev_toucher_pid']}({f['prev_toucher_team']})"
        lines.append(
            f"| {i} | {p.bucket} | {f['video_name']} | #{f['rally_order']} | "
            f"**{time_str}** | {f['pl_frame']} | {f['action_type']} | {prev_str} | "
            f"{pl_str} | **{s4_str}** | `{f['rally_id']}` |  |"
        )
    lines.append("")
    lines.append("## How to label each row")
    lines.append("")
    lines.append("1. Open the rally in the editor (URL pattern: `/videos/<video>/rallies/<rally_uuid>`).")
    lines.append("2. Scrub to the **source time** column (= rally start + frame/fps, this is the global video time).")
    lines.append("3. Find the action at that frame in the editor's action list — its current player ID matches **Pipeline pick**.")
    lines.append("4. Decide whether that player actually touched the ball at that frame.")
    lines.append("    - If yes: leave it as Pipeline pick.")
    lines.append("    - If S4 proposed pick is right: change the player to S4's pick.")
    lines.append("    - If someone else: change to that player.")
    lines.append("    - If it's a non-contact: delete the action.")
    lines.append("")
    lines.append("## After you finish")
    lines.append("")
    lines.append("I'll run a measurement script that looks up the GT `resolved_track_id` for each (rally_id, frame) within a +/-5-frame tolerance and computes three precision metrics:")
    lines.append("")
    lines.append("- **S4 precision** = fraction of flips where GT matches S4's pick (the ship signal — high precision means S4's flip is correct).")
    lines.append("- **Pipeline precision (on the same set)** = fraction where GT matches the pipeline's original pick (= the harm rate — high means S4 would break more than it fixes).")
    lines.append("- **Neither precision** = fraction where GT is neither (= cases where S4 is wrong but in a different way than pipeline).")
    lines.append("")
    lines.append("**Ship gate (preliminary, will refine after seeing data):** S4 precision >= 70% AND pipeline precision <= 30% AND lift (S4 - pipeline) >= +30 pp on this sample of 25-30 → SHIP S4. Otherwise NO-SHIP.")
    lines.append("")

    output.write_text("\n".join(lines))
    print(f"Wrote checklist: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build S4 labeling checklist from fleet candidates",
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    flips = data["flips"]
    print(f"Loaded {len(flips)} fleet flip candidates from {args.input}")

    picks = _stratified_pick(flips, seed=args.seed)
    print(f"Selected {len(picks)} stratified picks (target {TARGET_TOTAL})")
    bucket_counts: dict[str, int] = defaultdict(int)
    for p in picks:
        bucket_counts[p.bucket] += 1
    for b, c in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        print(f"  {b}: {c}")

    # Video coverage
    per_video: dict[str, int] = defaultdict(int)
    for p in picks:
        per_video[p.flip["video_name"]] += 1
    print(f"Distinct videos: {len(per_video)} (cap {PER_VIDEO_CAP}/video)")

    _write_markdown(picks, all_flips_count=len(flips), output=Path(args.output))


if __name__ == "__main__":
    main()
