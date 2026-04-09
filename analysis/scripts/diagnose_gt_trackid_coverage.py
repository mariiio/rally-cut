"""Session 11 pre-work diagnostic: GT ↔ positions_json trackId coverage.

Measures the gap between the current eval's literal-ID comparison
(`eval_action_detection.py:556`) and the ceiling achievable by normalizing
non-canonical trackIds via `match_analysis_json.rallies[i].trackToPlayer`.

GT labels carry no bbox/centroid, so direct spatial matching is not possible.
This script answers the fallback question: "how much of the identity-metric
loss is recoverable by mapping raw trackIds → canonical 1-4 at eval time?"

Usage:
    cd analysis
    uv run python scripts/diagnose_gt_trackid_coverage.py
    uv run python scripts/diagnose_gt_trackid_coverage.py --out outputs/session11/gt_trackid_coverage.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scripts.eval_action_detection import (
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
)

console = Console()


@dataclass
class RallyCoverage:
    rally_id: str
    video_id: str
    total_gt: int
    unlabeled: int  # playerTrackId == -1
    legacy_shifted: int  # playerTrackId >= 100
    literal_at_frame: int  # current eval path
    literal_anywhere: int  # weaker
    mapped_match: int  # mapping normalization path (gt id in 1..4, track at frame maps to it)
    has_match_analysis: bool = False
    positions_trackid_set: list[int] = field(default_factory=list)

    @property
    def evaluable(self) -> int:
        return max(self.total_gt - self.unlabeled, 0)

    @property
    def literal_ceiling(self) -> float:
        return self.literal_at_frame / self.evaluable if self.evaluable else 0.0

    @property
    def normalized_ceiling(self) -> float:
        """Union of literal_at_frame and mapped_match."""
        # Note: in the diagnostic we don't track per-label union, we track
        # per-label dominant-path assignment, so mapped_match already excludes
        # labels that literal_at_frame covered. Union = literal + mapped.
        unioned = self.literal_at_frame + self.mapped_match
        return unioned / self.evaluable if self.evaluable else 0.0

    @property
    def uplift(self) -> float:
        return self.normalized_ceiling - self.literal_ceiling


def _build_frame_to_trackids(positions_json: list[dict]) -> dict[int, set[int]]:
    """frame_number → set(trackId) lookup."""
    out: dict[int, set[int]] = defaultdict(set)
    for pp in positions_json:
        out[pp["frameNumber"]].add(pp["trackId"])
    return out


def analyze_rally(
    rally,
    t2p: dict[int, int],
) -> RallyCoverage:
    """Classify each GT label into coverage buckets."""
    cov = RallyCoverage(
        rally_id=rally.rally_id,
        video_id=rally.video_id,
        total_gt=len(rally.gt_labels),
        unlabeled=0,
        legacy_shifted=0,
        literal_at_frame=0,
        literal_anywhere=0,
        mapped_match=0,
        has_match_analysis=bool(t2p),
    )

    positions = rally.positions_json or []
    if not positions:
        cov.unlabeled = sum(1 for gt in rally.gt_labels if gt.player_track_id < 0)
        return cov

    frame_to_tids = _build_frame_to_trackids(positions)
    all_tids: set[int] = set()
    for tids in frame_to_tids.values():
        all_tids.update(tids)
    cov.positions_trackid_set = sorted(all_tids)

    for gt in rally.gt_labels:
        gid = gt.player_track_id
        if gid < 0:
            cov.unlabeled += 1
            continue
        if gid >= 100:
            cov.legacy_shifted += 1

        frame_tids = frame_to_tids.get(gt.frame, set())

        # 1. literal-at-frame (current eval path)
        if gid in frame_tids:
            cov.literal_at_frame += 1
            continue  # exclusive assignment to dominant path

        # 2. literal-anywhere (weaker signal)
        if gid in all_tids:
            cov.literal_anywhere += 1

        # 3. mapped match (the target): gid ∈ 1..4, and some track at this
        # frame maps to gid via trackToPlayer.
        if 1 <= gid <= 4 and t2p:
            for raw_tid in frame_tids:
                if t2p.get(raw_tid) == gid:
                    cov.mapped_match += 1
                    break

    return cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="outputs/session11/gt_trackid_coverage.json",
    )
    ap.add_argument("--rally", type=str, default=None, help="Single rally for debugging")
    args = ap.parse_args()

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    console.print(f"  Loaded {len(rallies)} rallies")

    video_ids = {r.video_id for r in rallies}
    console.print(f"[bold]Loading track_to_player maps for {len(video_ids)} videos...[/bold]")
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    console.print(f"  match_analysis present for {len(t2p_by_rally)}/{len(rallies)} rallies")

    console.print("\n[bold]Analyzing per-rally coverage...[/bold]")
    per_rally: list[RallyCoverage] = []
    for i, r in enumerate(rallies):
        t2p = t2p_by_rally.get(r.rally_id, {})
        cov = analyze_rally(r, t2p)
        per_rally.append(cov)
        print(
            f"[{i + 1}/{len(rallies)}] {r.rally_id[:8]} "
            f"gt={cov.total_gt} unlabeled={cov.unlabeled} "
            f"lit={cov.literal_at_frame} map={cov.mapped_match} "
            f"uplift={cov.uplift * 100:+.1f}pp",
            flush=True,
        )

    # Aggregate
    total_gt = sum(c.total_gt for c in per_rally)
    total_unlabeled = sum(c.unlabeled for c in per_rally)
    total_evaluable = sum(c.evaluable for c in per_rally)
    total_literal = sum(c.literal_at_frame for c in per_rally)
    total_mapped = sum(c.mapped_match for c in per_rally)
    total_literal_anywhere = sum(c.literal_anywhere for c in per_rally)
    total_legacy_shifted = sum(c.legacy_shifted for c in per_rally)

    literal_ceiling = total_literal / total_evaluable if total_evaluable else 0.0
    normalized_ceiling = (total_literal + total_mapped) / total_evaluable if total_evaluable else 0.0
    uplift = normalized_ceiling - literal_ceiling

    # Rallies dominated by mapping path
    mapping_dominant = [c for c in per_rally if c.mapped_match >= 1]
    mapping_dominant.sort(key=lambda c: -c.mapped_match)

    legacy_rallies = [c for c in per_rally if c.legacy_shifted > 0]

    # Print summary
    console.print()
    table = Table(title="GT Coverage — Aggregate", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Rallies", str(len(per_rally)))
    table.add_row("Total GT labels", str(total_gt))
    table.add_row("Unlabeled (playerTrackId=-1)", str(total_unlabeled))
    table.add_row("Evaluable (non-unlabeled)", str(total_evaluable))
    table.add_row("Legacy shifted (>=100)", str(total_legacy_shifted))
    table.add_row("[bold]literal_at_frame[/bold]", str(total_literal))
    table.add_row("literal_anywhere (weaker)", str(total_literal_anywhere))
    table.add_row("[bold]mapped_match (target)[/bold]", str(total_mapped))
    table.add_row(
        "[bold]Current ceiling (literal)[/bold]",
        f"{literal_ceiling * 100:.2f}%",
    )
    table.add_row(
        "[bold]Normalized ceiling (literal+mapped)[/bold]",
        f"{normalized_ceiling * 100:.2f}%",
    )
    table.add_row("[bold green]Uplift[/bold green]", f"[bold]{uplift * 100:+.2f}pp[/bold]")
    table.add_row("Rallies with mapped_match >= 1", str(len(mapping_dominant)))
    table.add_row("Rallies with legacy_shifted labels", str(len(legacy_rallies)))
    console.print(table)

    if mapping_dominant:
        console.print(
            f"\n[bold]Top 20 rallies by mapped_match count "
            f"(of {len(mapping_dominant)}):[/bold]"
        )
        top = Table()
        top.add_column("Rally")
        top.add_column("Video")
        top.add_column("GT", justify="right")
        top.add_column("Lit", justify="right")
        top.add_column("Map", justify="right")
        top.add_column("Uplift", justify="right")
        for c in mapping_dominant[:20]:
            top.add_row(
                c.rally_id[:8],
                c.video_id[:8],
                str(c.total_gt),
                str(c.literal_at_frame),
                str(c.mapped_match),
                f"{c.uplift * 100:+.1f}pp",
            )
        console.print(top)

    # Go/no-go gate
    console.print()
    if uplift >= 0.03:
        verdict = "GO — proceed to Phase B (uplift >= +3pp)"
        color = "bold green"
    elif uplift < 0.01:
        verdict = (
            "NO-GO — uplift < +1pp; stale-baseline damage is elsewhere. "
            "STOP and open a new diagnostic."
        )
        color = "bold red"
    else:
        verdict = (
            f"CONDITIONAL — uplift {uplift * 100:+.2f}pp between 1-3pp; "
            f"{len(mapping_dominant)} rallies dominated by mapping. "
            f"Proceed only if >= 20 rallies."
        )
        color = "bold yellow"
    console.print(f"[{color}]Verdict: {verdict}[/{color}]")

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": {
            "n_rallies": len(per_rally),
            "total_gt": total_gt,
            "total_unlabeled": total_unlabeled,
            "total_evaluable": total_evaluable,
            "total_literal_at_frame": total_literal,
            "total_literal_anywhere": total_literal_anywhere,
            "total_mapped_match": total_mapped,
            "total_legacy_shifted": total_legacy_shifted,
            "literal_ceiling": literal_ceiling,
            "normalized_ceiling": normalized_ceiling,
            "uplift": uplift,
            "n_rallies_with_mapping_path": len(mapping_dominant),
            "n_rallies_with_legacy_shifted": len(legacy_rallies),
            "verdict": verdict,
        },
        "per_rally": [asdict(c) for c in per_rally],
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    console.print(f"\n[dim]Wrote {out_path}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
