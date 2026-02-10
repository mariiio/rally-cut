"""Label Studio API client for seamless integration."""

from __future__ import annotations

import os
import webbrowser
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

# Default Label Studio URL
DEFAULT_URL = "http://localhost:8080"

# Label config for video object tracking
LABEL_CONFIG = """
<View>
  <Video name="video" value="$video"/>
  <VideoRectangle name="box" toName="video"/>
  <Labels name="label" toName="video">
    <Label value="player" background="green"/>
    <Label value="ball" background="orange"/>
  </Labels>
</View>
"""


class LabelStudioClient:
    """Client for Label Studio API."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        api_key: str | None = None,
    ):
        """Initialize client.

        Args:
            url: Label Studio server URL.
            api_key: API key (from Account & Settings → Access Token).
                    If not provided, uses LABEL_STUDIO_API_KEY env var.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("LABEL_STUDIO_API_KEY")
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get HTTP client with auth headers."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Token {self.api_key}"
            self._client = httpx.Client(
                base_url=self.url,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    def is_running(self) -> bool:
        """Check if Label Studio server is running."""
        try:
            resp = self.client.get("/api/health")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def is_authenticated(self) -> bool:
        """Check if API key is valid."""
        if not self.api_key:
            return False
        try:
            resp = self.client.get("/api/current-user/whoami")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def get_or_create_project(self, name: str = "RallyCut Ground Truth") -> int:
        """Get existing project or create new one.

        Returns:
            Project ID.
        """
        # Check for existing project
        resp = self.client.get("/api/projects")
        resp.raise_for_status()
        projects = resp.json().get("results", [])

        for project in projects:
            if project.get("title") == name:
                return int(project["id"])

        # Create new project
        resp = self.client.post(
            "/api/projects",
            json={
                "title": name,
                "label_config": LABEL_CONFIG,
            },
        )
        resp.raise_for_status()
        return int(resp.json()["id"])

    def create_task(
        self,
        project_id: int,
        video_path: str,
        predictions: list[dict[str, Any]] | None = None,
    ) -> int:
        """Create a labeling task.

        Args:
            project_id: Project to add task to.
            video_path: Path to video file (will be served locally).
            predictions: Pre-filled predictions in Label Studio format.

        Returns:
            Task ID.
        """
        task_data: dict[str, Any] = {
            "data": {"video": video_path},
        }

        if predictions:
            task_data["predictions"] = [
                {
                    "model_version": "rallycut",
                    "result": predictions,
                }
            ]

        resp = self.client.post(
            f"/api/projects/{project_id}/tasks",
            json=task_data,
        )
        resp.raise_for_status()
        return int(resp.json()["id"])

    def get_task_annotations(self, task_id: int) -> list[dict[str, Any]]:
        """Get annotations for a task.

        Returns:
            List of annotation results.
        """
        resp = self.client.get(f"/api/tasks/{task_id}")
        resp.raise_for_status()
        task = resp.json()

        annotations = task.get("annotations", [])
        if not annotations:
            return []

        # Get latest annotation
        latest = max(annotations, key=lambda a: a.get("id", 0))
        result: list[dict[str, Any]] = latest.get("result", [])
        return result

    def get_project_tasks(self, project_id: int) -> list[dict[str, Any]]:
        """Get all tasks in a project."""
        resp = self.client.get(f"/api/projects/{project_id}/tasks")
        resp.raise_for_status()
        tasks: list[dict[str, Any]] = resp.json()
        return tasks

    def open_task(self, project_id: int, task_id: int) -> None:
        """Open task in browser for labeling."""
        url = f"{self.url}/projects/{project_id}/data?task={task_id}"
        webbrowser.open(url)

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


def build_predictions(
    player_positions: list[PlayerPosition],
    ball_positions: list[BallPosition] | None = None,
    frame_offset: int = 0,
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    """Convert tracking results to Label Studio prediction format.

    Args:
        player_positions: Player tracking predictions.
        ball_positions: Ball tracking predictions (optional).
        frame_offset: Frame offset for segment.
        fps: Video frame rate.

    Returns:
        List of Label Studio result objects.
    """
    # Group player positions by track ID
    player_by_track: dict[int, list[PlayerPosition]] = {}
    for pos in player_positions:
        if pos.track_id < 0:
            continue
        if pos.track_id not in player_by_track:
            player_by_track[pos.track_id] = []
        player_by_track[pos.track_id].append(pos)

    results: list[dict[str, Any]] = []

    # Create video rectangle annotations for each player track
    for track_id in sorted(player_by_track.keys()):
        positions = sorted(player_by_track[track_id], key=lambda p: p.frame_number)

        sequence: list[dict[str, Any]] = []
        for pos in positions:
            frame = pos.frame_number - frame_offset
            time = frame / fps

            # Convert normalized center to percentage top-left
            x_pct = (pos.x - pos.width / 2) * 100
            y_pct = (pos.y - pos.height / 2) * 100
            w_pct = pos.width * 100
            h_pct = pos.height * 100

            sequence.append({
                "frame": frame + 1,  # 1-indexed
                "enabled": True,
                "x": max(0, min(100, x_pct)),
                "y": max(0, min(100, y_pct)),
                "width": w_pct,
                "height": h_pct,
                "rotation": 0,
                "time": time,
            })

        if sequence:
            results.append({
                "id": f"player_{track_id}",
                "type": "videorectangle",
                "value": {
                    "sequence": sequence,
                    "labels": ["player"],
                },
                "origin": "prediction",
                "to_name": "video",
                "from_name": "box",
            })

    # Add ball track if available
    if ball_positions:
        ball_sequence: list[dict[str, Any]] = []
        for ball_pos in sorted(ball_positions, key=lambda p: p.frame_number):
            frame = ball_pos.frame_number - frame_offset
            time = frame / fps
            ball_size_pct = 2.0

            x_pct = ball_pos.x * 100 - ball_size_pct / 2
            y_pct = ball_pos.y * 100 - ball_size_pct / 2

            ball_sequence.append({
                "frame": frame + 1,
                "enabled": True,
                "x": max(0, min(100, x_pct)),
                "y": max(0, min(100, y_pct)),
                "width": ball_size_pct,
                "height": ball_size_pct,
                "rotation": 0,
                "time": time,
            })

        if ball_sequence:
            results.append({
                "id": "ball_0",
                "type": "videorectangle",
                "value": {
                    "sequence": ball_sequence,
                    "labels": ["ball"],
                },
                "origin": "prediction",
                "to_name": "video",
                "from_name": "box",
            })

    return results
