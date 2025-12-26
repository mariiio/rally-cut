"""Statistics aggregation for RallyCut."""


from rallycut.core.models import (
    Action,
    ActionCount,
    ActionType,
    MatchStatistics,
    Rally,
    TimeSegment,
    VideoInfo,
)


class StatisticsAggregator:
    """Aggregates actions and segments into match statistics."""

    def __init__(self, video_info: VideoInfo):
        self.video_info = video_info

    def create_rallies(
        self,
        actions: list[Action],
        segments: list[TimeSegment] | None = None,
        rally_gap_seconds: float = 5.0,
    ) -> list[Rally]:
        """
        Create rally objects from actions.

        Args:
            actions: List of detected actions
            segments: Optional play segments to use as rally boundaries
            rally_gap_seconds: Time gap that separates rallies

        Returns:
            List of Rally objects
        """
        if not actions:
            return []

        # Sort by timestamp
        sorted_actions = sorted(actions, key=lambda a: a.timestamp)

        if segments:
            # Use segments as rally boundaries
            return self._rallies_from_segments(sorted_actions, segments)
        else:
            # Use time gaps to detect rallies
            return self._rallies_from_gaps(sorted_actions, rally_gap_seconds)

    def _rallies_from_segments(
        self,
        actions: list[Action],
        segments: list[TimeSegment],
    ) -> list[Rally]:
        """Create rallies based on play segments."""
        rallies = []

        for rally_id, segment in enumerate(segments, start=1):
            # Find actions within this segment
            rally_actions = [
                a for a in actions
                if segment.start_time <= a.timestamp <= segment.end_time
            ]

            rallies.append(
                Rally(
                    rally_id=rally_id,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    actions=rally_actions,
                )
            )

        return rallies

    def _rallies_from_gaps(
        self,
        actions: list[Action],
        rally_gap_seconds: float,
    ) -> list[Rally]:
        """Create rallies based on time gaps between actions."""
        if not actions:
            return []

        rallies = []
        current_actions = [actions[0]]
        rally_id = 1

        for action in actions[1:]:
            if action.timestamp - current_actions[-1].timestamp > rally_gap_seconds:
                # Save current rally
                rallies.append(
                    Rally(
                        rally_id=rally_id,
                        start_frame=current_actions[0].frame_idx,
                        end_frame=current_actions[-1].frame_idx,
                        start_time=current_actions[0].timestamp,
                        end_time=current_actions[-1].timestamp,
                        actions=current_actions,
                    )
                )
                rally_id += 1
                current_actions = [action]
            else:
                current_actions.append(action)

        # Don't forget last rally
        if current_actions:
            rallies.append(
                Rally(
                    rally_id=rally_id,
                    start_frame=current_actions[0].frame_idx,
                    end_frame=current_actions[-1].frame_idx,
                    start_time=current_actions[0].timestamp,
                    end_time=current_actions[-1].timestamp,
                    actions=current_actions,
                )
            )

        return rallies

    def count_actions(self, actions: list[Action]) -> dict[ActionType, ActionCount]:
        """
        Count actions by type.

        Args:
            actions: List of actions

        Returns:
            Dict mapping ActionType to ActionCount
        """
        counts: dict[ActionType, ActionCount] = {
            action_type: ActionCount(count=0)
            for action_type in ActionType
            if action_type != ActionType.BALL
        }

        for action in actions:
            if action.action_type in counts:
                counts[action.action_type] = ActionCount(
                    count=counts[action.action_type].count + 1
                )

        return counts

    def compute_statistics(
        self,
        actions: list[Action],
        segments: list[TimeSegment] | None = None,
    ) -> MatchStatistics:
        """
        Compute full match statistics.

        Args:
            actions: List of detected actions
            segments: Optional play segments

        Returns:
            MatchStatistics object
        """
        # Create rallies
        rallies = self.create_rallies(actions, segments)

        # Count actions
        action_counts = self.count_actions(actions)

        # Compute play/dead time
        if segments:
            play_duration = sum(s.duration for s in segments)
        else:
            play_duration = sum(r.duration for r in rallies)

        dead_time_duration = self.video_info.duration - play_duration

        # Rally statistics
        if rallies:
            rally_durations = [r.duration for r in rallies]
            avg_rally_duration = sum(rally_durations) / len(rally_durations)
            longest_rally_duration = max(rally_durations)
            shortest_rally_duration = min(rally_durations)
            total_touches = sum(len(r.actions) for r in rallies)
            touches_per_rally = total_touches / len(rallies)
        else:
            avg_rally_duration = 0.0
            longest_rally_duration = 0.0
            shortest_rally_duration = 0.0
            touches_per_rally = 0.0

        return MatchStatistics(
            video_info=self.video_info,
            total_rallies=len(rallies),
            total_duration=self.video_info.duration,
            play_duration=play_duration,
            dead_time_duration=dead_time_duration,
            serves=action_counts.get(ActionType.SERVE, ActionCount(0)),
            receptions=action_counts.get(ActionType.RECEPTION, ActionCount(0)),
            sets=action_counts.get(ActionType.SET, ActionCount(0)),
            attacks=action_counts.get(ActionType.ATTACK, ActionCount(0)),
            blocks=action_counts.get(ActionType.BLOCK, ActionCount(0)),
            avg_rally_duration=avg_rally_duration,
            longest_rally_duration=longest_rally_duration,
            shortest_rally_duration=shortest_rally_duration,
            touches_per_rally=touches_per_rally,
        )

    def to_dict(self, stats: MatchStatistics) -> dict:
        """Convert statistics to dictionary for JSON export."""
        return {
            "video": {
                "path": str(stats.video_info.path),
                "duration": stats.video_info.duration,
                "fps": stats.video_info.fps,
                "resolution": f"{stats.video_info.width}x{stats.video_info.height}",
            },
            "summary": {
                "total_rallies": stats.total_rallies,
                "total_duration": stats.total_duration,
                "play_duration": stats.play_duration,
                "dead_time_duration": stats.dead_time_duration,
                "dead_time_percentage": stats.dead_time_percentage,
            },
            "actions": {
                "serves": stats.serves.count,
                "receptions": stats.receptions.count,
                "sets": stats.sets.count,
                "attacks": stats.attacks.count,
                "blocks": stats.blocks.count,
            },
            "rallies": {
                "count": stats.total_rallies,
                "avg_duration": stats.avg_rally_duration,
                "longest_duration": stats.longest_rally_duration,
                "shortest_duration": stats.shortest_rally_duration,
                "avg_touches_per_rally": stats.touches_per_rally,
            },
        }
