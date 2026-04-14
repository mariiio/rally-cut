"""W1b: Serve-chain HMM smoothing diagnostic.

Tests whether HMM/Viterbi smoothing over the rally sequence (using the volleyball
run-length prior) can lift noisy per-rally serve detection into near-perfect
score_accuracy.

Architecture being tested:
  - State per rally = currently serving team (A or B).
  - Observations = stored first-serve action's `team` or `courtSide → team`
    from actions_json, with a confidence score.
  - Transition prior: P(stay) vs P(flip). In volleyball a team keeps serving
    until they lose a rally, so stays are correlated in short runs.
  - Anchor: first rally of each video (highest-confidence observation pins
    the chain; later we'd use match metadata).

Three observation streams reported:
  1. raw_team         — stored serve `team` field, no smoothing
  2. raw_team + hmm   — same observations, Viterbi-smoothed
  3. courtSide + hmm  — bypasses the corrupted team-lookup layer (see W1 memo),
                        uses `courtSide -> team` as the observation, smoothed

Gate: any variant >= 95% on the 302 GT rallies -> chain architecture confirmed.
      all variants < 90% -> observation stream too noisy; team-assignment fix
                            becomes prerequisite.

Read-only. No DB writes. No production changes.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402

STATES = ("A", "B")


@dataclass
class RallyObs:
    rally_id: str
    video_id: str
    start_ms: int
    gt_serving_team: str  # 'A' or 'B'
    raw_team: str | None  # from first-serve action's `team` field
    court_side_team: str | None  # courtSide → team BEFORE applying side switches
    court_side_team_flipped: str | None  # courtSide → team AFTER side-switch correction
    confidence: float
    rally_index: int = 0
    side_flipped: bool = False  # True if this rally is on the flipped side per GT sideSwitches


def _load_all() -> dict[str, list[RallyObs]]:
    """Load every rally in each of the 11 score-GT videos, ordered by start_ms.

    Also loads per-video sideSwitches from videos.player_matching_gt_json and
    computes side_flipped per rally. The side-switch-corrected courtSide→team
    mapping (court_side_team_flipped) flips 'A' <-> 'B' on rallies where the
    running toggle is True.
    """
    # Load side switches per video
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json
            FROM videos
            WHERE id IN (
                SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL
            )
        """)
        video_switches: dict[str, list[int]] = {}
        for vid, gt in cur.fetchall():
            if isinstance(gt, dict):
                video_switches[vid] = list(
                    gt.get("sideSwitches", gt.get("side_switches", []))
                )
            else:
                video_switches[vid] = []

    query = """
        SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team, pt.actions_json
        FROM rallies r
        LEFT JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id IN (
            SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL
        )
          AND r.gt_serving_team IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """
    raw_rows: dict[str, list[tuple]] = defaultdict(list)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        for rid, vid, start_ms, gt_team, aj in cur.fetchall():
            raw_rows[vid].append((rid, start_ms or 0, gt_team, aj))

    out: dict[str, list[RallyObs]] = {}
    for vid, rows in raw_rows.items():
        rows.sort(key=lambda r: r[1])
        switch_set = set(video_switches.get(vid, []))
        flipped = False
        vid_out: list[RallyObs] = []
        for idx, (rid, start_ms, gt_team, aj) in enumerate(rows):
            # Apply toggle at this rally index
            if idx in switch_set:
                flipped = not flipped
            raw_team: str | None = None
            cs_team: str | None = None
            conf = 0.0
            if isinstance(aj, dict):
                acts = aj.get("actions", []) or []
                serves = [a for a in acts if a.get("action") == "serve"]
                if serves:
                    s = serves[0]
                    t = s.get("team")
                    if t in ("A", "B"):
                        raw_team = t
                    cs = s.get("courtSide")
                    cs_team = {"near": "A", "far": "B"}.get(cs)
                    conf = float(s.get("confidence") or 0.0)
            cs_flipped: str | None = None
            if cs_team is not None:
                if flipped:
                    cs_flipped = "B" if cs_team == "A" else "A"
                else:
                    cs_flipped = cs_team
            vid_out.append(
                RallyObs(
                    rally_id=rid,
                    video_id=vid,
                    start_ms=start_ms,
                    gt_serving_team=gt_team,
                    raw_team=raw_team,
                    court_side_team=cs_team,
                    court_side_team_flipped=cs_flipped,
                    confidence=conf,
                    rally_index=idx,
                    side_flipped=flipped,
                )
            )
        out[vid] = vid_out
    return out


# ---------- HMM ----------------------------------------------------------


def _log(x: float) -> float:
    return math.log(max(x, 1e-9))


def viterbi(
    observations: list[str | None],
    confidences: list[float],
    p_stay: float,
    obs_strength: float = 1.0,
) -> list[str]:
    """Decode the most likely state sequence.

    Emission model: if observation is 'A' with confidence c,
        P(obs='A' | state='A') = 0.5 + 0.5 * c * obs_strength
        P(obs='A' | state='B') = 0.5 - 0.5 * c * obs_strength
    If observation is None: uniform emission (no evidence).
    Initial prior: uniform.
    """
    if not observations:
        return []
    n = len(observations)
    # log probs
    log_trans = {
        ("A", "A"): _log(p_stay),
        ("A", "B"): _log(1 - p_stay),
        ("B", "A"): _log(1 - p_stay),
        ("B", "B"): _log(p_stay),
    }

    def emission(obs: str | None, conf: float, state: str) -> float:
        if obs is None or obs not in STATES:
            return _log(0.5)
        c = max(0.0, min(1.0, conf)) * obs_strength
        if obs == state:
            return _log(0.5 + 0.5 * c)
        return _log(0.5 - 0.5 * c)

    # dp[i][state] = (log_prob, backpointer)
    dp: list[dict[str, tuple[float, str | None]]] = [dict() for _ in range(n)]
    for s in STATES:
        dp[0][s] = (_log(0.5) + emission(observations[0], confidences[0], s), None)

    for i in range(1, n):
        for s in STATES:
            best_prev_score = -math.inf
            best_prev_state: str | None = None
            for prev in STATES:
                score = dp[i - 1][prev][0] + log_trans[(prev, s)]
                if score > best_prev_score:
                    best_prev_score = score
                    best_prev_state = prev
            em = emission(observations[i], confidences[i], s)
            dp[i][s] = (best_prev_score + em, best_prev_state)

    # Backtrack
    last_state = max(STATES, key=lambda s: dp[n - 1][s][0])
    path = [last_state]
    for i in range(n - 1, 0, -1):
        _, bp = dp[i][path[-1]]
        assert bp is not None
        path.append(bp)
    path.reverse()
    return path


# ---------- Evaluation --------------------------------------------------


def _accuracy(preds: list[str | None], gts: list[str]) -> tuple[int, int]:
    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    return correct, len(gts)


def _eval_raw(video_rallies: dict[str, list[RallyObs]], field: str) -> tuple[int, int, int]:
    correct = 0
    total = 0
    abstain = 0
    for rallies in video_rallies.values():
        for r in rallies:
            total += 1
            p = getattr(r, field)
            if p is None:
                abstain += 1
                continue
            if p == r.gt_serving_team:
                correct += 1
    return correct, total, abstain


def viterbi_count_constrained(
    observations: list[str | None],
    confidences: list[float],
    n_a_target: int,
    p_stay: float = 0.515,
    obs_strength: float = 1.0,
) -> list[str]:
    """Decode with exact-count constraint: output must contain exactly n_a_target 'A' states.

    State dp[i][s][k] = max log prob at rally i in state s having seen k A's.
    """
    n = len(observations)
    if n == 0:
        return []
    neg_inf = -1e18
    log_trans = {
        ("A", "A"): _log(p_stay),
        ("A", "B"): _log(1 - p_stay),
        ("B", "A"): _log(1 - p_stay),
        ("B", "B"): _log(p_stay),
    }

    def emission(obs: str | None, conf: float, state: str) -> float:
        if obs is None or obs not in STATES:
            return _log(0.5)
        c = max(0.0, min(1.0, conf)) * obs_strength
        if obs == state:
            return _log(0.5 + 0.5 * c)
        return _log(0.5 - 0.5 * c)

    # dp[i][s_idx][k] = (log_prob, prev_s_idx)
    # s_idx: 0=A, 1=B
    dp = [[[neg_inf] * (n_a_target + 2) for _ in range(2)] for _ in range(n)]
    bp = [[[-1] * (n_a_target + 2) for _ in range(2)] for _ in range(n)]

    # init
    k_a = 1
    k_b = 0
    if k_a <= n_a_target:
        dp[0][0][k_a] = _log(0.5) + emission(observations[0], confidences[0], "A")
    dp[0][1][k_b] = _log(0.5) + emission(observations[0], confidences[0], "B")

    for i in range(1, n):
        for s_idx, s in enumerate(STATES):
            # k range: at least enough prior A's, at most n_a_target
            k_min = 1 if s == "A" else 0
            for k in range(k_min, n_a_target + 1):
                em = emission(observations[i], confidences[i], s)
                best = neg_inf
                best_prev = -1
                for prev_idx, prev in enumerate(STATES):
                    prev_k = k - (1 if s == "A" else 0)
                    if prev_k < 0 or prev_k > n_a_target:
                        continue
                    pv = dp[i - 1][prev_idx][prev_k]
                    if pv <= neg_inf / 2:
                        continue
                    sc = pv + log_trans[(prev, s)] + em
                    if sc > best:
                        best = sc
                        best_prev = prev_idx
                if best > neg_inf / 2:
                    dp[i][s_idx][k] = best
                    bp[i][s_idx][k] = best_prev

    # Final: state at rally n-1 with k == n_a_target or n_a_target (exact)
    final_k = n_a_target
    best_final = neg_inf
    best_final_s = 0
    for s_idx in range(2):
        if dp[n - 1][s_idx][final_k] > best_final:
            best_final = dp[n - 1][s_idx][final_k]
            best_final_s = s_idx

    if best_final <= neg_inf / 2:
        # Infeasible (shouldn't happen if n_a_target in [0, n])
        return ["A"] * n_a_target + ["B"] * (n - n_a_target)

    # Backtrack
    path_idx = [best_final_s]
    k = final_k
    for i in range(n - 1, 0, -1):
        prev_s_idx = bp[i][path_idx[-1]][k]
        if path_idx[-1] == 0:  # current is A, so we added 1 to k
            k -= 1
        path_idx.append(prev_s_idx)
    path_idx.reverse()
    return [STATES[s] for s in path_idx]


def _eval_constrained(
    video_rallies: dict[str, list[RallyObs]],
    field: str,
    p_stay: float = 0.515,
) -> tuple[int, int]:
    """Per-video constrained Viterbi using GT-derived per-video A count."""
    correct = 0
    total = 0
    for rallies in video_rallies.values():
        obs = [getattr(r, field) for r in rallies]
        conf = [r.confidence for r in rallies]
        n_a_target = sum(1 for r in rallies if r.gt_serving_team == "A")
        preds = viterbi_count_constrained(obs, conf, n_a_target, p_stay=p_stay)
        for pred, r in zip(preds, rallies):
            total += 1
            if pred == r.gt_serving_team:
                correct += 1
    return correct, total


def _eval_hmm(
    video_rallies: dict[str, list[RallyObs]],
    field: str,
    p_stay: float,
    obs_strength: float = 1.0,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for rallies in video_rallies.values():
        obs = [getattr(r, field) for r in rallies]
        conf = [r.confidence for r in rallies]
        preds = viterbi(obs, conf, p_stay=p_stay, obs_strength=obs_strength)
        for pred, r in zip(preds, rallies):
            total += 1
            if pred == r.gt_serving_team:
                correct += 1
    return correct, total


def _run_length_stats(video_rallies: dict[str, list[RallyObs]]) -> dict:
    """Measure GT serving-run lengths to sanity-check p_stay."""
    runs: list[int] = []
    for rallies in video_rallies.values():
        cur_team = None
        cur_len = 0
        for r in rallies:
            if r.gt_serving_team == cur_team:
                cur_len += 1
            else:
                if cur_len > 0:
                    runs.append(cur_len)
                cur_team = r.gt_serving_team
                cur_len = 1
        if cur_len > 0:
            runs.append(cur_len)
    if not runs:
        return {"mean_run": 0, "p_stay_mle": 0.0, "n_runs": 0}
    total_rallies = sum(runs)
    # P(stay) MLE = (total - n_flips) / (total - 1)
    n_flips = len(runs) - 1  # flips between consecutive runs within same video... approx
    # More careful: count actual adjacent (stay, flip) pairs within each video
    stays = 0
    flips = 0
    for rallies in video_rallies.values():
        for i in range(1, len(rallies)):
            if rallies[i].gt_serving_team == rallies[i - 1].gt_serving_team:
                stays += 1
            else:
                flips += 1
    p_stay_mle = stays / max(1, stays + flips)
    return {
        "n_runs": len(runs),
        "mean_run": total_rallies / len(runs),
        "max_run": max(runs),
        "p_stay_mle": p_stay_mle,
        "adjacent_pairs": stays + flips,
    }


def main() -> int:
    print("Loading all rallies from 11 score-GT videos...")
    video_rallies = _load_all()
    total_rallies = sum(len(v) for v in video_rallies.values())
    print(f"  videos={len(video_rallies)} rallies={total_rallies}")
    for vid, rs in sorted(video_rallies.items()):
        print(f"    {vid[:8]}: {len(rs)} rallies")

    # Observation coverage
    raw_present = sum(1 for rs in video_rallies.values() for r in rs if r.raw_team is not None)
    cs_present = sum(1 for rs in video_rallies.values() for r in rs if r.court_side_team is not None)
    print(f"\nobservation coverage:")
    print(f"  raw_team present: {raw_present}/{total_rallies} ({raw_present/total_rallies*100:.1f}%)")
    print(f"  court_side present: {cs_present}/{total_rallies} ({cs_present/total_rallies*100:.1f}%)")

    # Side-switch coverage
    flipped_rallies = sum(1 for rs in video_rallies.values() for r in rs if r.side_flipped)
    print(f"  rallies on flipped side (per GT sideSwitches): {flipped_rallies}/{total_rallies} "
          f"({flipped_rallies/total_rallies*100:.1f}%)")

    # Run-length stats to set p_stay
    stats = _run_length_stats(video_rallies)
    print(f"\nGT run-length stats: {stats}")
    p_stay_mle = stats["p_stay_mle"]

    # GT class balance
    gts = [r.gt_serving_team for rs in video_rallies.values() for r in rs]
    cb = Counter(gts)
    print(f"GT class balance: A={cb['A']} B={cb['B']}  majority={max(cb.values())/total_rallies*100:.1f}%")

    print(f"\nBaseline (W1 production): 46.2% on 91-rally subset. "
          f"Gate on this 302-rally set: >= 95%.\n")

    # Raw baselines
    print("=== Raw (no HMM) ===")
    c, t, ab = _eval_raw(video_rallies, "raw_team")
    print(f"  raw_team                    {c}/{t} = {c/t*100:5.1f}%  (abstain={ab})")
    c, t, ab = _eval_raw(video_rallies, "court_side_team")
    print(f"  court_side_team (uncorr)    {c}/{t} = {c/t*100:5.1f}%  (abstain={ab})")
    c, t, ab = _eval_raw(video_rallies, "court_side_team_flipped")
    print(f"  court_side_team (GT-switch) {c}/{t} = {c/t*100:5.1f}%  (abstain={ab})")

    # HMM sweeps
    print("\n=== HMM (p_stay sweep) ===")
    sweep = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    if p_stay_mle > 0:
        sweep = sorted(set(sweep + [round(p_stay_mle, 3)]))
    print(f"  (MLE p_stay from GT = {p_stay_mle:.3f})")
    print(f"  {'p_stay':>7s}  {'raw+hmm':>10s}  {'cs+hmm':>10s}  {'cs_flip+hmm':>12s}")
    best = (0.0, "", 0.0)
    for p in sweep:
        c1, t1 = _eval_hmm(video_rallies, "raw_team", p_stay=p)
        c2, t2 = _eval_hmm(video_rallies, "court_side_team", p_stay=p)
        c3, t3 = _eval_hmm(video_rallies, "court_side_team_flipped", p_stay=p)
        r1 = c1 / t1
        r2 = c2 / t2
        r3 = c3 / t3
        print(f"  {p:7.3f}  {c1:4d}/{t1:3d}={r1*100:5.1f}%  "
              f"{c2:4d}/{t2:3d}={r2*100:5.1f}%  {c3:4d}/{t3:3d}={r3*100:5.1f}%")
        if r1 > best[0]:
            best = (r1, f"raw+hmm p_stay={p}", r1)
        if r2 > best[0]:
            best = (r2, f"courtSide+hmm p_stay={p}", r2)
        if r3 > best[0]:
            best = (r3, f"cs_flip+hmm p_stay={p}", r3)

    print(f"\nBest unconstrained: {best[1]} = {best[2]*100:.2f}%")

    # === Count-constrained Viterbi: user provides final score ===
    print("\n=== Count-constrained Viterbi (simulates user providing final score) ===")
    print(f"  {'p_stay':>7s}  {'raw+constr':>13s}  {'cs+constr':>13s}  {'cs_flip+constr':>15s}")
    best_c = (0.0, "", 0.0)
    for p in [0.50, 0.515, 0.60, 0.70, 0.80]:
        c1, t1 = _eval_constrained(video_rallies, "raw_team", p_stay=p)
        c2, t2 = _eval_constrained(video_rallies, "court_side_team", p_stay=p)
        c3, t3 = _eval_constrained(video_rallies, "court_side_team_flipped", p_stay=p)
        r1 = c1 / t1
        r2 = c2 / t2
        r3 = c3 / t3
        print(f"  {p:7.3f}  {c1:4d}/{t1:3d}={r1*100:5.1f}%  "
              f"{c2:4d}/{t2:3d}={r2*100:5.1f}%  {c3:4d}/{t3:3d}={r3*100:5.1f}%")
        if r1 > best_c[0]:
            best_c = (r1, f"raw+constr p_stay={p}", r1)
        if r2 > best_c[0]:
            best_c = (r2, f"courtSide+constr p_stay={p}", r2)
        if r3 > best_c[0]:
            best_c = (r3, f"cs_flip+constr p_stay={p}", r3)
    print(f"\nBest constrained: {best_c[1]} = {best_c[2]*100:.2f}%")

    # Update best tracker
    if best_c[2] > best[2]:
        best = best_c

    # Per-video breakdown at best config
    print("\n=== Per-video breakdown (best variant) ===")
    if "cs_flip" in best[1]:
        use_field = "court_side_team_flipped"
    elif "courtSide" in best[1] or "cs" in best[1]:
        use_field = "court_side_team"
    else:
        use_field = "raw_team"
    p_best = float(best[1].split("p_stay=")[1])
    for vid, rs in sorted(video_rallies.items()):
        obs = [getattr(r, use_field) for r in rs]
        conf = [r.confidence for r in rs]
        preds = viterbi(obs, conf, p_stay=p_best)
        c = sum(1 for p, r in zip(preds, rs) if p == r.gt_serving_team)
        print(f"  {vid[:8]}  {c:3d}/{len(rs):3d}  {c/len(rs)*100:5.1f}%")

    # Verdict
    print()
    if best[2] >= 0.95:
        print(f"VERDICT: GO — chain architecture hits {best[2]*100:.1f}% (>=95%). "
              "Team-assignment drift fix is OPTIONAL on score, required only "
              "for serve_attr / court_side.")
    elif best[2] >= 0.90:
        print(f"VERDICT: MAYBE — best {best[2]*100:.1f}% is in 90-95% band. "
              "Investigate whether observation stream cleanup (team-assignment "
              "drift fix) closes the gap, or whether anchor strategy matters.")
    else:
        print(f"VERDICT: NO-GO on raw chain — best {best[2]*100:.1f}% < 90%. "
              "Observation stream too noisy/biased. Team-assignment drift fix "
              "becomes prerequisite; re-run after fix to measure true HMM lift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
