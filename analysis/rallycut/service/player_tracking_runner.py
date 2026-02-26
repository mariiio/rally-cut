"""Local player tracking runner for development.

This script runs player tracking locally and calls the webhook when complete,
just like Modal would in production. Used for local development where
Modal can't callback to localhost.

Usage:
    python -m rallycut.service.player_tracking_runner \
        --job-id <uuid> \
        --video-path <local_path_or_s3_key> \
        --rally-id <uuid> \
        --start-ms 1000 \
        --end-ms 15000 \
        --callback-url http://localhost:3001/v1/webhooks/player-tracking-complete \
        --webhook-secret <secret>
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import httpx

from rallycut.service.s3_utils import download_from_s3


def send_webhook(
    callback_url: str,
    webhook_secret: str | None,
    payload: dict,
) -> bool:
    """Send webhook with results."""
    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["X-Webhook-Secret"] = webhook_secret

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload, headers=headers)
            response.raise_for_status()
            print(f"[LOCAL] Webhook sent successfully to {callback_url}")
            return True
    except Exception as e:
        print(f"[LOCAL] Webhook failed: {e}")
        return False


def send_progress_webhook(
    base_callback_url: str,
    webhook_secret: str | None,
    job_id: str,
    progress: float,
    message: str,
) -> bool:
    """Send progress update webhook."""
    # Derive progress URL from base URL path
    base_url = base_callback_url.rsplit("/", 1)[0] if "/" in base_callback_url else base_callback_url
    progress_url = f"{base_url}/player-tracking-progress"

    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["X-Webhook-Secret"] = webhook_secret

    payload = {
        "job_id": job_id,
        "progress": progress,
        "message": message,
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(progress_url, json=payload, headers=headers)
            response.raise_for_status()
            return True
    except Exception as e:
        # Don't fail tracking on progress webhook error
        print(f"[LOCAL] Progress webhook failed: {e}")
        return False


def run_tracking(
    job_id: str,
    video_path: str,
    rally_id: str,
    start_ms: int,
    end_ms: int,
    callback_url: str,
    webhook_secret: str | None,
    s3_bucket: str | None = None,
    calibration_json: str | None = None,
    ball_positions_json: str | None = None,
) -> None:
    """Run player tracking and send results via webhook."""
    start_time = time.time()
    temp_dir = None
    local_video_path = None

    try:
        # Import tracking modules
        from rallycut.tracking.player_tracker import PlayerTracker

        # Parse calibration if provided
        calibrator = None
        if calibration_json:
            try:
                calibration_data = json.loads(calibration_json)
                # Handle both {"corners": [...]} and bare [...] formats
                if isinstance(calibration_data, list):
                    corners = calibration_data
                elif isinstance(calibration_data, dict):
                    corners = calibration_data.get("corners", [])
                    print(f"[LOCAL] Using calibration: courtType={calibration_data.get('courtType')}")
                else:
                    corners = []

                if isinstance(corners, list) and len(corners) == 4:
                    from rallycut.court.calibration import CourtCalibrator

                    calibrator = CourtCalibrator()
                    image_corners = [(c["x"], c["y"]) for c in corners]
                    calibrator.calibrate(image_corners)
                    print("[LOCAL] Court calibrator initialized from corners")
                else:
                    print(f"[LOCAL] Invalid calibration corners (expected 4, got {len(corners)})")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[LOCAL] Failed to parse calibration JSON: {e}")
        else:
            print("[LOCAL] No calibration data provided")

        # Parse ball positions if provided
        ball_positions = None
        if ball_positions_json:
            ball_positions = json.loads(ball_positions_json)
            print(f"[LOCAL] Using ball positions: {len(ball_positions)} detections")

        # Determine if we need to download from S3
        if video_path.startswith("s3://") or (s3_bucket and not Path(video_path).exists()):
            if video_path.startswith("s3://"):
                parts = video_path[5:].split("/", 1)
                s3_bucket = parts[0]
                video_s3_key = parts[1] if len(parts) > 1 else ""
            else:
                video_s3_key = video_path

            temp_dir = Path(tempfile.mkdtemp(prefix="rallycut_tracking_"))
            local_video_path = download_from_s3(video_s3_key, s3_bucket or "", temp_dir, label="[LOCAL]")
        else:
            local_video_path = Path(video_path)

        if not local_video_path.exists():
            raise FileNotFoundError(f"Video not found: {local_video_path}")

        print(f"[LOCAL] Starting player tracking for {local_video_path}")
        print(f"[LOCAL] Rally: {rally_id}")
        print(f"[LOCAL] Time range: {start_ms}ms - {end_ms}ms")

        # Auto-detect court if no manual calibration provided
        court_insights = None
        if calibrator is None:
            from rallycut.court.detector import CourtDetectionInsights
            from rallycut.tracking.player_tracker import auto_detect_court

            calibrator, auto_result = auto_detect_court(local_video_path)
            court_insights = CourtDetectionInsights.from_result(auto_result)
            if calibrator is not None:
                print(f"[LOCAL] Court auto-detected (confidence: {auto_result.confidence:.2f})")
            else:
                print("[LOCAL] Court auto-detection: no confident result")
                for tip in court_insights.recording_tips:
                    print(f"[LOCAL]   Tip: {tip}")

        # Compute calibration ROI if calibrator available
        court_roi = None
        if calibrator is not None:
            from rallycut.tracking.player_tracker import compute_court_roi_from_calibration

            cal_roi, cal_msg = compute_court_roi_from_calibration(calibrator)
            if cal_roi is not None:
                court_roi = cal_roi
                print("[LOCAL] Using calibration ROI")
            elif cal_msg:
                print(f"[LOCAL] Calibration ROI failed: {cal_msg}")
                # Invalidate calibrator — degenerate homography will cause
                # off-court filtering to silently reject all detections
                calibrator = None

        # Create tracker with tuned confidence threshold
        tracker = PlayerTracker(confidence=0.15, court_roi=court_roi)

        # Import filter config for beach volleyball
        from rallycut.tracking.player_filter import PlayerFilterConfig

        filter_config = PlayerFilterConfig()

        # Run tracking with progress webhook
        def progress_callback(progress: float) -> None:
            print(f"[LOCAL] Tracking progress: {progress * 100:.0f}%")
            send_progress_webhook(
                callback_url,
                webhook_secret,
                job_id,
                progress,
                "Tracking players",
            )

        result = tracker.track_video(
            video_path=local_video_path,
            start_ms=start_ms,
            end_ms=end_ms,
            progress_callback=progress_callback,
            ball_positions=ball_positions,
            filter_enabled=True,
            filter_config=filter_config,
            court_calibrator=calibrator,
            court_detection_insights=court_insights,
        )

        # Estimate court from player positions when line detection failed
        if result.positions:
            try:
                from rallycut.tracking.player_tracker import refine_court_with_players

                team_assigns = getattr(result, "team_assignments", {})
                if team_assigns:
                    refined_cal, refined_result = refine_court_with_players(
                        auto_result if court_insights is not None else None,
                        result.positions,
                        team_assigns,
                    )
                    method = getattr(refined_result, "fitting_method", "")
                    if refined_cal is not None and "player" in method:
                        calibrator = refined_cal
                        if court_insights is not None:
                            from rallycut.court.detector import CourtDetectionInsights
                            court_insights = CourtDetectionInsights.from_result(
                                refined_result,
                            )
                        print(f"[LOCAL] Court estimated from players ({method})")
            except Exception as e:
                print(f"[LOCAL] Court player estimation failed (non-fatal): {e}")

        processing_time_ms = (time.time() - start_time) * 1000

        # Build webhook payload
        webhook_payload: dict = {
            "job_id": job_id,
            "status": "completed",
            "tracks_json": result.to_dict(),
            "player_count": result.unique_track_count,
            "frame_count": result.frame_count,
            "processing_time_ms": round(processing_time_ms),
            "model_version": result.model_version,
        }
        if court_insights is not None:
            webhook_payload["courtDetection"] = court_insights.to_dict()

        print(f"[LOCAL] Tracking complete: {result.unique_track_count} players, {result.frame_count} frames")
        print(f"[LOCAL] Processing time: {processing_time_ms/1000:.1f}s")

        # Send webhook
        send_webhook(callback_url, webhook_secret, webhook_payload)

    except Exception as e:
        print(f"[LOCAL] Tracking failed: {e}")
        import traceback
        traceback.print_exc()

        # Send error webhook
        error_payload = {
            "job_id": job_id,
            "status": "failed",
            "error_message": str(e),
        }
        send_webhook(callback_url, webhook_secret, error_payload)

    finally:
        # Cleanup temp directory
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run player tracking locally")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--video-path", required=True, help="Local path or S3 key")
    parser.add_argument("--rally-id", required=True, help="Rally ID")
    parser.add_argument("--start-ms", type=int, required=True, help="Start time in milliseconds")
    parser.add_argument("--end-ms", type=int, required=True, help="End time in milliseconds")
    parser.add_argument("--callback-url", required=True, help="Webhook callback URL")
    parser.add_argument("--webhook-secret", help="Webhook secret for auth")
    parser.add_argument("--s3-bucket", help="S3 bucket name (for S3 downloads)")
    parser.add_argument("--calibration", help="Calibration data as JSON string")
    parser.add_argument("--ball-positions-json", help="Ball positions as JSON string")

    args = parser.parse_args()

    run_tracking(
        job_id=args.job_id,
        video_path=args.video_path,
        rally_id=args.rally_id,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        callback_url=args.callback_url,
        webhook_secret=args.webhook_secret,
        s3_bucket=args.s3_bucket,
        calibration_json=args.calibration,
        ball_positions_json=args.ball_positions_json,
    )


if __name__ == "__main__":
    main()
