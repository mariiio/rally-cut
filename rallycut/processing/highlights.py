"""Highlight generation for RallyCut."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rallycut.core.models import TimeSegment
from rallycut.processing.exporter import FFmpegExporter


@dataclass
class ScoredSegment:
    """A segment with a highlight score."""

    segment: TimeSegment
    score: float
    rank: int = 0

    @property
    def duration(self) -> float:
        return self.segment.duration

    @property
    def start_time(self) -> float:
        return self.segment.start_time

    @property
    def end_time(self) -> float:
        return self.segment.end_time


class HighlightScorer:
    """
    Scores and ranks rallies for highlight generation.

    Scoring is based on rally duration - longer rallies are typically
    more exciting with more back-and-forth action.
    """

    def __init__(
        self,
        min_duration: float = 3.0,
        max_duration: float = 60.0,
    ):
        """
        Initialize highlight scorer.

        Args:
            min_duration: Minimum rally duration to consider (seconds)
            max_duration: Cap duration for scoring (prevents outliers)
        """
        self.min_duration = min_duration
        self.max_duration = max_duration

    def score_segments(
        self,
        segments: list[TimeSegment],
    ) -> list[ScoredSegment]:
        """
        Score and rank segments by highlight potential.

        Args:
            segments: List of play segments (rallies)

        Returns:
            List of ScoredSegments sorted by score (highest first)
        """
        scored = []

        for segment in segments:
            # Skip segments that are too short
            if segment.duration < self.min_duration:
                continue

            # Score based on duration (capped)
            duration = min(segment.duration, self.max_duration)
            score = duration / self.max_duration  # Normalize to 0-1

            scored.append(ScoredSegment(segment=segment, score=score))

        # Sort by score (highest first)
        scored.sort(key=lambda s: s.score, reverse=True)

        # Assign ranks
        for i, s in enumerate(scored):
            s.rank = i + 1

        return scored

    def get_top_highlights(
        self,
        segments: list[TimeSegment],
        count: int = 5,
    ) -> list[ScoredSegment]:
        """
        Get the top N highlights.

        Args:
            segments: List of play segments
            count: Number of highlights to return

        Returns:
            Top N scored segments
        """
        scored = self.score_segments(segments)
        return scored[:count]


class HighlightGenerator:
    """
    Generates highlight videos from scored segments.
    """

    def __init__(self):
        self.exporter = FFmpegExporter()
        self.scorer = HighlightScorer()

    def generate_highlights(
        self,
        input_path: Path,
        output_path: Path,
        segments: list[TimeSegment],
        count: int = 5,
        chronological: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> tuple[Path, list[ScoredSegment]]:
        """
        Generate a highlights video from the top rallies.

        Args:
            input_path: Source video path
            output_path: Output video path
            segments: All detected play segments
            count: Number of highlights to include
            chronological: If True, sort clips by time; if False, by score
            progress_callback: Progress callback

        Returns:
            Tuple of (output path, list of included highlights)
        """
        # Score and select top highlights
        top_highlights = self.scorer.get_top_highlights(segments, count)

        if not top_highlights:
            raise ValueError("No highlights found matching criteria")

        # Sort by time if chronological
        if chronological:
            export_order = sorted(top_highlights, key=lambda s: s.start_time)
        else:
            export_order = top_highlights

        # Extract segments for export
        export_segments = [h.segment for h in export_order]

        # Export
        self.exporter.export_segments(
            input_path=input_path,
            output_path=output_path,
            segments=export_segments,
            progress_callback=progress_callback,
        )

        return output_path, top_highlights

    def export_individual_clips(
        self,
        input_path: Path,
        output_dir: Path,
        segments: list[TimeSegment],
        count: int = 5,
        prefix: str = "highlight",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[tuple[Path, ScoredSegment]]:
        """
        Export top highlights as individual clips.

        Args:
            input_path: Source video path
            output_dir: Directory for output clips
            segments: All detected play segments
            count: Number of highlights to export
            prefix: Filename prefix for clips
            progress_callback: Progress callback

        Returns:
            List of (output_path, scored_segment) tuples
        """
        top_highlights = self.scorer.get_top_highlights(segments, count)

        if not top_highlights:
            raise ValueError("No highlights found matching criteria")

        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, highlight in enumerate(top_highlights):
            if progress_callback:
                progress_callback(i / len(top_highlights), f"Exporting clip {i + 1}")

            # Name clips by rank
            clip_path = output_dir / f"{prefix}_{highlight.rank:02d}.mp4"

            self.exporter.export_clip(
                input_path=input_path,
                output_path=clip_path,
                start_time=highlight.start_time,
                end_time=highlight.end_time,
            )

            results.append((clip_path, highlight))

        if progress_callback:
            progress_callback(1.0, "Export complete")

        return results
