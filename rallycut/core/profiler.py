"""Performance profiling utilities for RallyCut."""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


@dataclass
class TimingEntry:
    """Single timing measurement."""

    component: str
    operation: str
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Thread-safe performance profiler for timing collection.

    Usage:
        profiler = PerformanceProfiler()

        with profiler.time("videomae", "inference"):
            # ... code to time ...

        report = profiler.report()
    """

    def __init__(self) -> None:
        self._entries: list[TimingEntry] = []
        self._lock = threading.Lock()
        self._enabled = True

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


# Global profiler instance (disabled by default)
_global_profiler: Optional[PerformanceProfiler] = None


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
