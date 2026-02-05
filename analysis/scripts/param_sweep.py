#!/usr/bin/env python3
"""Parameter sweep for optimizing ML rally detection.

Systematically tests parameter combinations on ground truth videos
to find optimal settings that maximize recall while minimizing
false positives and processing time.
"""

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.core.config import (
    GameStateConfig,
    RallyCutConfig,
    SegmentConfig,
    reset_config,
    set_config,
)
from rallycut.processing.cutter import VideoCutter

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParameterSet:
    """A single parameter configuration to test."""

    # Segment parameters
    min_play_duration: float
    min_gap_seconds: float
    padding_seconds: float
    # Speed parameters
    stride: int

    def to_config(self) -> RallyCutConfig:
        """Convert to RallyCutConfig object."""
        return RallyCutConfig(
            segment=SegmentConfig(
                min_play_duration=self.min_play_duration,
                padding_seconds=self.padding_seconds,
                min_gap_seconds=self.min_gap_seconds,
            ),
            game_state=GameStateConfig(
                stride=self.stride,
            ),
        )


@dataclass
class EvaluationResult:
    """Results from evaluating a parameter set on a video."""

    video_name: str
    expected_rally_count: int
    detected_segment_count: int
    # Recall metrics
    matched_rallies: int
    missed_rallies: int
    recall: float
    # Precision metrics
    false_positives: int
    precision: float
    # Boundary accuracy
    avg_start_error: float
    avg_end_error: float
    # Performance
    processing_time_seconds: float


@dataclass
class SweepResult:
    """Combined result for a parameter set across all videos."""

    params: ParameterSet
    video_results: list[EvaluationResult]
    # Aggregated metrics
    total_recall: float
    total_precision: float
    all_rallies_detected: bool
    total_false_positives: int
    avg_processing_time: float
    passes_tests: bool


# =============================================================================
# Parameter Grid
# =============================================================================

PARAMETER_GRID = {
    # Segment processing
    "min_play_duration": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "min_gap_seconds": [1.5, 2.0, 3.0, 4.0, 5.0],
    "padding_seconds": [0.5, 1.0, 1.5, 2.0],
    # Speed parameters
    "stride": [16, 24, 32, 48],
}

QUICK_GRID = {
    # Reduced grid for quick testing
    "min_play_duration": [2.0, 3.0],
    "min_gap_seconds": [2.0, 3.0],
    "padding_seconds": [1.0],
    "stride": [32],
}

SPEED_GRID = {
    # Grid focused on stride optimization with known-working detection params
    "min_play_duration": [2.0],
    "min_gap_seconds": [2.0],
    "padding_seconds": [1.0],
    "stride": [16, 24, 32, 48, 64],
}


def generate_parameter_sets(
    grid: dict,
    quick_mode: bool = False,
    speed_mode: bool = False,
) -> list[ParameterSet]:
    """Generate valid parameter combinations."""
    if speed_mode:
        grid = SPEED_GRID
    elif quick_mode:
        grid = QUICK_GRID

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    valid_sets = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        valid_sets.append(ParameterSet(**params))

    return valid_sets


# =============================================================================
# Detection Evaluator
# =============================================================================


class DetectionEvaluator:
    """Evaluates detection quality against ground truth."""

    def __init__(self, tolerance_seconds: float = 3.0):
        self.tolerance = tolerance_seconds

    def load_expected_rallies(self, json_path: Path) -> list[tuple[float, float]]:
        """Load expected rallies from fixture file."""
        with open(json_path) as f:
            raw_rallies = json.load(f)

        def parse_time_range(time_range: str) -> tuple[float, float]:
            start_str, end_str = time_range.split("-")

            def to_seconds(t: str) -> float:
                parts = t.split(":")
                return int(parts[0]) * 60 + int(parts[1])

            return to_seconds(start_str), to_seconds(end_str)

        return [parse_time_range(r) for r in raw_rallies]

    def evaluate(
        self,
        video_path: Path,
        expected_rallies: list[tuple[float, float]],
        params: ParameterSet,
    ) -> EvaluationResult:
        """Run detection and evaluate against expected rallies."""
        # Configure global config
        reset_config()
        set_config(params.to_config())

        # Create cutter with test parameters
        cutter = VideoCutter(
            padding_seconds=params.padding_seconds,
            min_play_duration=params.min_play_duration,
            stride=params.stride,
            min_gap_seconds=params.min_gap_seconds,
        )

        # Time the detection
        start_time = time.time()
        try:
            segments = cutter.analyze_only(video_path)
        except Exception as e:
            print(f"    ERROR: {e}")
            return EvaluationResult(
                video_name=video_path.name,
                expected_rally_count=len(expected_rallies),
                detected_segment_count=0,
                matched_rallies=0,
                missed_rallies=len(expected_rallies),
                recall=0.0,
                false_positives=0,
                precision=0.0,
                avg_start_error=float("inf"),
                avg_end_error=float("inf"),
                processing_time_seconds=time.time() - start_time,
            )
        processing_time = time.time() - start_time

        detected = [(s.start_time, s.end_time) for s in segments]

        # Match expected to detected (same logic as test)
        matched_indices = set()
        start_errors = []
        end_errors = []

        for exp_start, exp_end in expected_rallies:
            best_idx = None
            best_overlap = 0.0

            for idx, (det_start, det_end) in enumerate(detected):
                overlap_start = max(exp_start, det_start)
                overlap_end = min(exp_end, det_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx

            if best_idx is not None:
                det_start, det_end = detected[best_idx]
                start_ok = abs(exp_start - det_start) <= self.tolerance
                end_ok = abs(exp_end - det_end) <= self.tolerance

                if start_ok and end_ok:
                    matched_indices.add(best_idx)
                    start_errors.append(abs(exp_start - det_start))
                    end_errors.append(abs(exp_end - det_end))

        matched = len(matched_indices)
        missed = len(expected_rallies) - matched
        false_positives = len(detected) - len(matched_indices)

        recall = matched / len(expected_rallies) if expected_rallies else 1.0
        precision = matched / len(detected) if detected else 0.0

        return EvaluationResult(
            video_name=video_path.name,
            expected_rally_count=len(expected_rallies),
            detected_segment_count=len(detected),
            matched_rallies=matched,
            missed_rallies=missed,
            recall=recall,
            false_positives=false_positives,
            precision=precision,
            avg_start_error=sum(start_errors) / len(start_errors) if start_errors else float("inf"),
            avg_end_error=sum(end_errors) / len(end_errors) if end_errors else float("inf"),
            processing_time_seconds=processing_time,
        )


# =============================================================================
# Sweep Runner
# =============================================================================


class SweepRunner:
    """Orchestrates the parameter sweep."""

    def __init__(
        self,
        output_dir: Path,
        tolerance: float = 3.0,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = DetectionEvaluator(tolerance_seconds=tolerance)

        # Test videos and their expected rallies
        project_root = Path(__file__).parent.parent
        self.test_cases = [
            {
                "video": project_root / "tests/fixtures/match_first_2min.mp4",
                "expected": project_root / "tests/fixtures/expected_rallies.json",
                "name": "match1",
            },
            {
                "video": project_root / "tests/fixtures/match-2-first-2min.MOV",
                "expected": project_root / "tests/fixtures/expected_rallies_match2.json",
                "name": "match2",
            },
        ]

    def run_sweep(
        self,
        parameter_sets: list[ParameterSet],
        resume_from: Path | None = None,
    ) -> list[SweepResult]:
        """Run the full parameter sweep."""
        import hashlib

        results = []
        completed_hashes = set()

        # Load previous results if resuming
        if resume_from and resume_from.exists():
            with open(resume_from) as f:
                for line in f:
                    data = json.loads(line)
                    param_hash = hashlib.md5(
                        json.dumps(data["params"], sort_keys=True).encode()
                    ).hexdigest()
                    completed_hashes.add(param_hash)
            print(f"Resuming: {len(completed_hashes)} configurations already completed")

        # Prepare results file for incremental writing
        results_file = self.output_dir / "sweep_results.jsonl"

        total = len(parameter_sets)
        for i, params in enumerate(parameter_sets):
            param_hash = hashlib.md5(
                json.dumps(asdict(params), sort_keys=True).encode()
            ).hexdigest()

            if param_hash in completed_hashes:
                continue

            print(
                f"\n[{i+1}/{total}] min_play={params.min_play_duration}, "
                f"min_gap={params.min_gap_seconds}, "
                f"stride={params.stride}"
            )

            video_results = []
            for test_case in self.test_cases:
                if not test_case["video"].exists():
                    print(f"  Skipping {test_case['name']}: video not found")
                    continue

                expected = self.evaluator.load_expected_rallies(test_case["expected"])
                result = self.evaluator.evaluate(
                    test_case["video"],
                    expected,
                    params,
                )
                video_results.append(result)

                status = "PASS" if result.missed_rallies == 0 else "FAIL"
                print(
                    f"  {test_case['name']}: {status} "
                    f"(recall={result.recall:.0%}, FP={result.false_positives}, "
                    f"time={result.processing_time_seconds:.1f}s)"
                )

            # Aggregate results
            if video_results:
                total_expected = sum(r.expected_rally_count for r in video_results)
                total_matched = sum(r.matched_rallies for r in video_results)

                sweep_result = SweepResult(
                    params=params,
                    video_results=video_results,
                    total_recall=total_matched / total_expected if total_expected else 0,
                    total_precision=sum(r.precision for r in video_results) / len(video_results),
                    all_rallies_detected=all(r.missed_rallies == 0 for r in video_results),
                    total_false_positives=sum(r.false_positives for r in video_results),
                    avg_processing_time=sum(r.processing_time_seconds for r in video_results)
                    / len(video_results),
                    passes_tests=all(r.missed_rallies == 0 for r in video_results),
                )
                results.append(sweep_result)

                # Write incrementally
                self._append_result(results_file, sweep_result)

        return results

    def _append_result(self, path: Path, result: SweepResult):
        """Append result to JSONL file."""
        data = {
            "params": asdict(result.params),
            "total_recall": result.total_recall,
            "total_precision": result.total_precision,
            "all_rallies_detected": result.all_rallies_detected,
            "total_false_positives": result.total_false_positives,
            "avg_processing_time": result.avg_processing_time,
            "passes_tests": result.passes_tests,
            "video_results": [asdict(vr) for vr in result.video_results],
        }
        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")


# =============================================================================
# Result Analyzer
# =============================================================================


class ResultAnalyzer:
    """Analyzes sweep results to find optimal configurations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def load_results(self) -> list[SweepResult]:
        """Load results from JSONL file."""
        results_file = self.output_dir / "sweep_results.jsonl"
        results = []

        if not results_file.exists():
            return results

        with open(results_file) as f:
            for line in f:
                data = json.loads(line)
                params = ParameterSet(**data["params"])
                video_results = [EvaluationResult(**vr) for vr in data["video_results"]]
                results.append(
                    SweepResult(
                        params=params,
                        video_results=video_results,
                        total_recall=data["total_recall"],
                        total_precision=data["total_precision"],
                        all_rallies_detected=data["all_rallies_detected"],
                        total_false_positives=data["total_false_positives"],
                        avg_processing_time=data["avg_processing_time"],
                        passes_tests=data["passes_tests"],
                    )
                )

        return results

    def find_passing_configs(self, results: list[SweepResult]) -> list[SweepResult]:
        """Find all configurations that pass both tests."""
        return [r for r in results if r.passes_tests]

    def find_optimal(
        self,
        results: list[SweepResult],
        require_passing: bool = True,
    ) -> SweepResult | None:
        """Find optimal configuration balancing recall and speed.

        Priority:
        1. 100% recall (all expected rallies detected)
        2. Lowest false positives
        3. Fastest processing time
        """
        candidates = results
        if require_passing:
            candidates = self.find_passing_configs(results)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda r: (
                not r.passes_tests,
                r.total_false_positives,
                r.avg_processing_time,
            ),
        )

    def generate_report(self, results: list[SweepResult]):
        """Generate human-readable report and CSV."""
        passing = self.find_passing_configs(results)
        optimal = self.find_optimal(results)

        lines = [
            "=" * 60,
            "PARAMETER SWEEP RESULTS",
            "=" * 60,
            "",
            f"Total configurations tested: {len(results)}",
            f"Configurations passing all tests: {len(passing)}",
            "",
        ]

        if optimal:
            lines.extend(
                [
                    "OPTIMAL CONFIGURATION:",
                    "-" * 40,
                    f"  min_play_duration: {optimal.params.min_play_duration}",
                    f"  min_gap_seconds: {optimal.params.min_gap_seconds}",
                    f"  padding_seconds: {optimal.params.padding_seconds}",
                    f"  stride: {optimal.params.stride}",
                    "",
                    f"  Total Recall: {optimal.total_recall:.1%}",
                    f"  False Positives: {optimal.total_false_positives}",
                    f"  Avg Processing Time: {optimal.avg_processing_time:.1f}s",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "NO CONFIGURATION PASSES ALL TESTS",
                    "",
                ]
            )
            if results:
                best_recall = max(results, key=lambda r: r.total_recall)
                lines.append(f"Best by recall: {best_recall.total_recall:.1%}")
                lines.append(f"  min_play={best_recall.params.min_play_duration}")
                lines.append(f"  min_gap={best_recall.params.min_gap_seconds}")
            lines.append("")

        # Show top 5 passing configs sorted by speed
        if passing:
            lines.extend(
                [
                    "TOP 5 PASSING CONFIGS (by speed):",
                    "-" * 40,
                ]
            )
            sorted_passing = sorted(passing, key=lambda r: r.avg_processing_time)
            for i, r in enumerate(sorted_passing[:5]):
                lines.append(
                    f"  {i+1}. FP={r.total_false_positives}, "
                    f"time={r.avg_processing_time:.1f}s, "
                    f"stride={r.params.stride}"
                )
            lines.append("")

        # Write report
        report_path = self.output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        print("\n".join(lines))

        # Write CSV
        self._write_csv(results)

    def _write_csv(self, results: list[SweepResult]):
        """Write results to CSV for easy analysis."""
        csv_path = self.output_dir / "sweep_results.csv"

        if not results:
            return

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            param_fields = list(asdict(results[0].params).keys())
            writer.writerow(
                param_fields
                + [
                    "total_recall",
                    "total_precision",
                    "all_rallies_detected",
                    "total_false_positives",
                    "avg_processing_time",
                    "passes_tests",
                ]
            )

            # Data rows
            for r in results:
                param_values = list(asdict(r.params).values())
                writer.writerow(
                    param_values
                    + [
                        f"{r.total_recall:.4f}",
                        f"{r.total_precision:.4f}",
                        r.all_rallies_detected,
                        r.total_false_positives,
                        f"{r.avg_processing_time:.2f}",
                        r.passes_tests,
                    ]
                )

        print(f"\nCSV saved to: {csv_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for ML rally detection optimization"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("sweep_results"),
        help="Directory for output files (default: sweep_results/)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced parameter grid for quick testing",
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="Test stride values only (for speed optimization with known-working params)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from previous results file (JSONL)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="Tolerance in seconds for rally matching (default: 3.0)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results",
    )

    args = parser.parse_args()

    # Report only mode
    if args.report_only:
        analyzer = ResultAnalyzer(args.output_dir)
        results = analyzer.load_results()
        if not results:
            print(f"No results found in {args.output_dir}")
            return
        analyzer.generate_report(results)
        return

    # Generate parameter sets
    print("Generating parameter combinations...")
    param_sets = generate_parameter_sets(PARAMETER_GRID, quick_mode=args.quick, speed_mode=args.speed)
    print(f"Total valid combinations: {len(param_sets)}")

    if args.speed:
        print("(Speed mode: testing stride values with known-working detection params)")
    elif args.quick:
        print("(Quick mode: using reduced parameter grid)")

    # Estimate time
    est_time_per_combo = 20  # seconds per combo (2 videos x ~10s each)
    est_total_minutes = (len(param_sets) * est_time_per_combo) / 60
    print(f"Estimated total time: {est_total_minutes:.0f} minutes")

    # Run sweep
    runner = SweepRunner(args.output_dir, tolerance=args.tolerance)
    results = runner.run_sweep(
        param_sets,
        resume_from=args.resume or (args.output_dir / "sweep_results.jsonl"),
    )

    # Analyze and report
    analyzer = ResultAnalyzer(args.output_dir)
    all_results = analyzer.load_results()
    analyzer.generate_report(all_results)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
