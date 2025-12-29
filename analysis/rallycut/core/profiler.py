"""Performance profiling utilities for RallyCut."""

import json
import threading
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TimingEntry:
    """Single timing measurement."""

    component: str
    operation: str
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage with hierarchical support."""

    name: str
    start_time: float
    end_time: float = 0.0
    duration_seconds: float = 0.0
    items_processed: int = 0
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Calculate duration when stage ends."""
        self.duration_seconds = self.end_time - self.start_time


class PerformanceProfiler:
    """Thread-safe performance profiler for timing collection.

    Usage:
        profiler = PerformanceProfiler()

        # Simple timing
        with profiler.time("videomae", "inference"):
            # ... code to time ...

        # Stage tracking with hierarchy
        with profiler.stage("ml_analysis") as stage:
            stage.items_processed = 100
            with profiler.stage("preprocessing", parent="ml_analysis"):
                # ... preprocessing ...
            with profiler.stage("inference", parent="ml_analysis"):
                # ... inference ...

        report = profiler.report()
    """

    def __init__(self) -> None:
        self._entries: list[TimingEntry] = []
        self._stages: list[StageMetrics] = []
        self._active_stages: dict[str, StageMetrics] = {}
        self._lock = threading.Lock()
        self._enabled = True
        self._config_snapshot: dict[str, Any] = {}
        self._video_info: dict[str, Any] = {}

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling (context managers become no-ops)."""
        self._enabled = False

    def clear(self) -> None:
        """Clear all collected entries."""
        with self._lock:
            self._entries.clear()
            self._stages.clear()
            self._active_stages.clear()
            self._config_snapshot.clear()
            self._video_info.clear()

    def set_config(self, config: dict[str, Any]) -> None:
        """Store configuration snapshot for experiment tracking."""
        self._config_snapshot = config

    def set_video_info(
        self, path: str, duration_seconds: float, fps: float, frame_count: int
    ) -> None:
        """Store video information."""
        self._video_info = {
            "path": path,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "frame_count": frame_count,
        }

    @contextmanager
    def stage(
        self, name: str, parent: str | None = None, **metadata: Any
    ) -> Generator[StageMetrics, None, None]:
        """Context manager for tracking a pipeline stage.

        Args:
            name: Stage name (e.g., "proxy_generation", "ml_analysis")
            parent: Parent stage name for hierarchy
            **metadata: Additional metadata

        Yields:
            StageMetrics object that can be updated during execution
        """
        if not self._enabled:
            # Return a dummy stage that does nothing
            dummy = StageMetrics(name=name, start_time=0.0, parent=parent)
            yield dummy
            return

        stage = StageMetrics(
            name=name,
            start_time=time.perf_counter(),
            parent=parent,
            metadata=metadata,
        )

        with self._lock:
            self._active_stages[name] = stage

        try:
            yield stage
        finally:
            stage.end_time = time.perf_counter()
            stage.finalize()

            with self._lock:
                self._stages.append(stage)
                if name in self._active_stages:
                    del self._active_stages[name]

    @contextmanager
    def time(
        self, component: str, operation: str, **metadata: Any
    ) -> Generator[None, None, None]:
        """Context manager to time an operation.

        Args:
            component: High-level component name (e.g., "videomae", "motion", "decode")
            operation: Specific operation (e.g., "inference", "preprocess", "read_frame")
            **metadata: Additional metadata (e.g., batch_size=16, device="cuda")
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            entry = TimingEntry(
                component=component,
                operation=operation,
                duration_seconds=elapsed,
                metadata=metadata,
            )
            with self._lock:
                self._entries.append(entry)

    def get_entries(self) -> list[TimingEntry]:
        """Get all timing entries."""
        with self._lock:
            return list(self._entries)

    def report(self) -> dict[str, Any]:
        """Generate a summary report of all timings.

        Returns:
            Dict with structure:
            {
                "total_seconds": float,
                "by_component": {
                    "component_name": {
                        "total_seconds": float,
                        "percentage": float,
                        "operations": {
                            "operation_name": {
                                "total_seconds": float,
                                "count": int,
                                "avg_seconds": float,
                            }
                        }
                    }
                },
                "entries_count": int,
            }
        """
        entries = self.get_entries()
        if not entries:
            return {"total_seconds": 0.0, "by_component": {}, "entries_count": 0}

        # Calculate totals
        total_seconds = sum(e.duration_seconds for e in entries)

        # Group by component and operation
        by_component: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in entries:
            by_component[entry.component][entry.operation].append(entry.duration_seconds)

        # Build report
        component_report: dict[str, Any] = {}
        for component, operations in by_component.items():
            component_total = sum(sum(times) for times in operations.values())
            ops_report: dict[str, Any] = {}
            for op, times in operations.items():
                op_total = sum(times)
                ops_report[op] = {
                    "total_seconds": round(op_total, 4),
                    "count": len(times),
                    "avg_seconds": round(op_total / len(times), 4) if times else 0,
                }
            component_report[component] = {
                "total_seconds": round(component_total, 4),
                "percentage": round(100 * component_total / total_seconds, 1)
                if total_seconds > 0
                else 0,
                "operations": ops_report,
            }

        return {
            "total_seconds": round(total_seconds, 4),
            "by_component": component_report,
            "entries_count": len(entries),
        }

    def print_report(self) -> None:
        """Print a formatted report to stdout."""
        report = self.report()
        total = report["total_seconds"]

        print(f"\n{'=' * 60}")
        print(f"Performance Profile ({report['entries_count']} entries)")
        print(f"{'=' * 60}")
        print(f"Total time: {total:.2f}s\n")

        # Sort by total time descending
        sorted_components = sorted(
            report["by_component"].items(),
            key=lambda x: x[1]["total_seconds"],
            reverse=True,
        )

        for component, data in sorted_components:
            print(f"{component}: {data['total_seconds']:.2f}s ({data['percentage']:.1f}%)")
            sorted_ops = sorted(
                data["operations"].items(),
                key=lambda x: x[1]["total_seconds"],
                reverse=True,
            )
            for op, op_data in sorted_ops:
                print(
                    f"  {op}: {op_data['total_seconds']:.2f}s "
                    f"({op_data['count']}x, avg {op_data['avg_seconds']*1000:.1f}ms)"
                )
        print(f"{'=' * 60}\n")

    def get_stages(self) -> list[StageMetrics]:
        """Get all completed stage metrics."""
        with self._lock:
            return list(self._stages)

    def stages_report(self) -> dict[str, Any]:
        """Generate a report of stage timings with hierarchy.

        Returns:
            Dict with structure including stages, video info, and config.
        """
        stages = self.get_stages()

        # Build stage hierarchy
        stage_data = []
        for stage in stages:
            stage_data.append({
                "name": stage.name,
                "duration_seconds": round(stage.duration_seconds, 4),
                "items_processed": stage.items_processed,
                "parent": stage.parent,
                "metadata": stage.metadata,
            })

        # Calculate total from top-level stages only
        top_level_total = sum(
            s.duration_seconds for s in stages if s.parent is None
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_seconds": round(top_level_total, 4),
            "video": self._video_info,
            "config": self._config_snapshot,
            "stages": stage_data,
            "entries": self.report(),
        }

    def print_stages_report(self) -> None:
        """Print a formatted stages report to stdout."""
        stages = self.get_stages()
        if not stages:
            print("No stages recorded.")
            return

        # Get video info
        video_info = self._video_info
        video_name = Path(video_info.get("path", "unknown")).name
        video_duration = video_info.get("duration_seconds", 0)

        # Calculate total from top-level stages
        top_level = [s for s in stages if s.parent is None]
        total_time = sum(s.duration_seconds for s in top_level)

        print(f"\n{'=' * 65}")
        print("Performance Profile")
        if video_name != "unknown":
            print(f"Video: {video_name} ({video_duration:.1f}s)")
        print(f"{'=' * 65}")

        # Print stages table
        print(f"\n{'Stage':<30} | {'Time':>10} | {'%':>6} | {'Items':>8}")
        print("-" * 65)

        for stage in stages:
            indent = "  " if stage.parent else ""
            name = f"{indent}{stage.name}"
            pct = (stage.duration_seconds / total_time * 100) if total_time > 0 else 0
            items = stage.items_processed if stage.items_processed > 0 else ""
            print(f"{name:<30} | {stage.duration_seconds:>9.2f}s | {pct:>5.1f}% | {items:>8}")

        print("-" * 65)
        print(f"{'Total':<30} | {total_time:>9.2f}s | {'100.0':>5}% |")
        print(f"{'=' * 65}")

        # Print throughput stats if available
        if video_duration > 0 and total_time > 0:
            speed_ratio = video_duration / total_time
            print(f"\nProcessing speed: {speed_ratio:.2f}x realtime")

        print()

    def export_json(self, path: Path) -> None:
        """Export full report to JSON file."""
        report = self.stages_report()
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def export_csv(self, path: Path) -> None:
        """Export stages to CSV file."""
        stages = self.get_stages()

        with open(path, "w") as f:
            # Header
            f.write("stage,duration_seconds,items_processed,parent\n")

            # Data rows
            for stage in stages:
                parent = stage.parent or ""
                f.write(
                    f"{stage.name},{stage.duration_seconds:.4f},"
                    f"{stage.items_processed},{parent}\n"
                )


# Global profiler instance (disabled by default)
_global_profiler: PerformanceProfiler | None = None


def get_profiler() -> PerformanceProfiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
        _global_profiler.disable()  # Disabled by default for normal runs
    return _global_profiler


def enable_profiling() -> PerformanceProfiler:
    """Enable the global profiler and return it."""
    profiler = get_profiler()
    profiler.enable()
    profiler.clear()
    return profiler
