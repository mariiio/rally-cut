"""
Video Export Lambda Handler

Processes video export jobs by:
1. Downloading video clips from S3
2. Extracting clip segments using FFmpeg
3. Applying watermark/downscaling for free tier
4. Concatenating clips into final output
5. Uploading result to S3
6. Calling webhook to notify completion
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import requests

s3_client = boto3.client("s3")

# Watermark image embedded as base64 or fetched from S3
WATERMARK_S3_KEY = "assets/watermark.png"


def generate_camera_filter(
    camera: dict,
    duration_sec: float,
    input_width: int = 1920,
    input_height: int = 1080,
) -> str:
    """
    Generate FFmpeg video filter for camera panning/zooming.
    Uses crop filter with expressions to animate position based on keyframes.
    """
    fps = 30
    total_frames = int(duration_sec * fps)

    aspect_ratio = camera.get("aspectRatio", "ORIGINAL")
    keyframes = camera.get("keyframes", [])

    # Calculate output dimensions
    if aspect_ratio == "VERTICAL":
        output_height = input_height
        output_width = round(output_height * 9 / 16)
    else:
        output_width = input_width
        output_height = input_height

    # No keyframes - static center crop
    if not keyframes:
        x = round((input_width - output_width) / 2)
        y = round((input_height - output_height) / 2)
        return f"crop={output_width}:{output_height}:{x}:{y}"

    # Sort keyframes by time
    sorted_kfs = sorted(keyframes, key=lambda k: k.get("timeOffset", 0))

    # Single keyframe - static position with zoom
    if len(sorted_kfs) == 1:
        kf = sorted_kfs[0]
        zoom = max(1, min(3, kf.get("zoom", 1)))
        cropped_w = round(output_width / zoom)
        cropped_h = round(output_height / zoom)
        max_x = input_width - cropped_w
        max_y = input_height - cropped_h
        x = round(kf.get("positionX", 0.5) * max_x)
        y = round(kf.get("positionY", 0.5) * max_y)
        return f"crop={cropped_w}:{cropped_h}:{x}:{y},scale={output_width}:{output_height}"

    # Helper for easing expressions
    def easing_expr(t: str, easing: str) -> str:
        if easing == "EASE_IN":
            return f"({t}*{t})"
        elif easing == "EASE_OUT":
            return f"(1-(1-{t})*(1-{t}))"
        elif easing == "EASE_IN_OUT":
            return f"({t}<0.5?2*{t}*{t}:1-(-2*{t}+2)*(-2*{t}+2)/2)"
        return t  # LINEAR

    # Build piecewise expression for a value
    def build_piecewise_expr(kfs: list, get_value, total_frames: int) -> str:
        if len(kfs) == 1:
            return str(get_value(kfs[0]))

        expr = ""
        for i in range(len(kfs) - 2, -1, -1):
            kf1, kf2 = kfs[i], kfs[i + 1]
            frame1 = round(kf1.get("timeOffset", 0) * total_frames)
            frame2 = round(kf2.get("timeOffset", 0) * total_frames)
            v1 = get_value(kf1)
            v2 = get_value(kf2)

            t_expr = f"(on-{frame1})/{max(1, frame2 - frame1)}"
            eased_t = easing_expr(t_expr, kf2.get("easing", "LINEAR"))
            interp_expr = f"{v1}+{eased_t}*({v2}-{v1})"

            if i == len(kfs) - 2:
                expr = f"if(lt(on,{frame1}),{v1},{interp_expr})"
            else:
                expr = f"if(lt(on,{frame1}),{v1},if(lt(on,{frame2}),{interp_expr},{expr}))"

        return expr or str(get_value(kfs[0]))

    zoom_expr = build_piecewise_expr(sorted_kfs, lambda k: max(1, min(3, k.get("zoom", 1))), total_frames)
    px_expr = build_piecewise_expr(sorted_kfs, lambda k: k.get("positionX", 0.5), total_frames)
    py_expr = build_piecewise_expr(sorted_kfs, lambda k: k.get("positionY", 0.5), total_frames)

    # Calculate filter expressions
    base_w = output_width
    base_h = output_height

    x_expr_final = f"floor(({px_expr})*({input_width}-{base_w}/({zoom_expr})))"
    y_expr_final = f"floor(({py_expr})*({input_height}-{base_h}/({zoom_expr})))"
    w_expr_final = f"floor({base_w}/({zoom_expr}))"
    h_expr_final = f"floor({base_h}/({zoom_expr}))"

    return f"crop=w='{w_expr_final}':h='{h_expr_final}':x='{x_expr_final}':y='{y_expr_final}',scale={output_width}:{output_height}"


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda entry point."""
    print(f"Received event: {json.dumps(event)}")

    # Validate required parameters
    required = ["jobId", "rallies", "callbackUrl", "webhookSecret", "s3Bucket"]
    missing = [key for key in required if key not in event]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    job_id = event["jobId"]
    tier = event.get("tier", "FREE")
    output_format = event.get("format", "mp4")
    rallies = event["rallies"]
    callback_url = event["callbackUrl"]
    webhook_secret = event["webhookSecret"]
    s3_bucket = event["s3Bucket"]

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_key = process_export(
                job_id=job_id,
                tier=tier,
                output_format=output_format,
                rallies=rallies,
                s3_bucket=s3_bucket,
                tmpdir=Path(tmpdir),
                callback_url=callback_url,
                webhook_secret=webhook_secret,
            )

        # Notify success (raise on error so Lambda fails if webhook fails)
        send_webhook(
            callback_url,
            webhook_secret,
            {
                "job_id": job_id,
                "status": "completed",
                "output_s3_key": output_key,
            },
            raise_on_error=True,
        )

        return {"statusCode": 200, "body": json.dumps({"success": True})}

    except Exception as e:
        print(f"Export failed: {e}")
        # Notify failure
        send_webhook(
            callback_url,
            webhook_secret,
            {
                "job_id": job_id,
                "status": "failed",
                "error_message": str(e),
            },
        )
        raise


def process_export(
    job_id: str,
    tier: str,
    output_format: str,
    rallies: list[dict],
    s3_bucket: str,
    tmpdir: Path,
    callback_url: str,
    webhook_secret: str,
) -> str:
    """Process the export and return the output S3 key."""
    clips_dir = tmpdir / "clips"
    clips_dir.mkdir()

    clip_files: list[Path] = []
    total_rallies = len(rallies)

    # Download and process each clip
    for i, rally in enumerate(rallies):
        progress = int((i / total_rallies) * 80)  # 0-80% for clip extraction
        send_progress(callback_url, webhook_secret, job_id, progress)

        video_s3_key = rally["videoS3Key"]
        start_ms = rally["startMs"]
        end_ms = rally["endMs"]

        # Download video from S3
        video_path = tmpdir / f"video_{i}.mp4"
        print(f"Downloading {video_s3_key} to {video_path}")
        s3_client.download_file(s3_bucket, video_s3_key, str(video_path))

        # Extract clip
        clip_path = clips_dir / f"clip_{i:04d}.mp4"
        camera = rally.get("camera")  # Optional camera edit data
        extract_clip(
            input_path=video_path,
            output_path=clip_path,
            start_ms=start_ms,
            end_ms=end_ms,
            tier=tier,
            s3_bucket=s3_bucket,
            tmpdir=tmpdir,
            camera=camera,
        )
        clip_files.append(clip_path)

        # Clean up source video to save space
        video_path.unlink()

    # Concatenate clips
    send_progress(callback_url, webhook_secret, job_id, 85)
    output_path = tmpdir / f"output.{output_format}"
    concatenate_clips(clip_files, output_path)

    # Upload to S3
    send_progress(callback_url, webhook_secret, job_id, 95)
    output_key = f"exports/{job_id}/output.{output_format}"
    print(f"Uploading to {output_key}")
    s3_client.upload_file(
        str(output_path),
        s3_bucket,
        output_key,
        ExtraArgs={"ContentType": f"video/{output_format}"},
    )

    return output_key


def extract_clip(
    input_path: Path,
    output_path: Path,
    start_ms: int,
    end_ms: int,
    tier: str,
    s3_bucket: str,
    tmpdir: Path,
    camera: dict | None = None,
) -> None:
    """Extract a clip from video, applying tier-specific processing and optional camera effects."""
    start_sec = start_ms / 1000
    duration_sec = (end_ms - start_ms) / 1000

    # Check if camera effects are enabled
    has_camera = camera is not None and camera.get("keyframes")
    if has_camera:
        print(f"Applying camera effects: {camera.get('aspectRatio', 'ORIGINAL')}, {len(camera.get('keyframes', []))} keyframes")

    if has_camera:
        # Camera edits require re-encoding
        camera_filter = generate_camera_filter(camera, duration_sec)
        print(f"Camera filter: {camera_filter}")
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-i",
            str(input_path),
            "-t",
            str(duration_sec),
            "-vf",
            camera_filter,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    elif tier == "PREMIUM":
        # Premium: fast copy, no re-encoding (when no camera effects)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-i",
            str(input_path),
            "-t",
            str(duration_sec),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    else:
        # Free: scale to 720p and add watermark
        watermark_path = tmpdir / "watermark.png"
        if not watermark_path.exists():
            try:
                s3_client.download_file(s3_bucket, WATERMARK_S3_KEY, str(watermark_path))
            except Exception as e:
                print(f"Could not download watermark: {e}, proceeding without")
                watermark_path = None

        if watermark_path and watermark_path.exists():
            # Scale to 720p height and overlay watermark in bottom-right corner
            filter_complex = (
                "[0:v]scale=-2:720[scaled];"
                "[scaled][1:v]overlay=W-w-20:H-h-20[out]"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_sec),
                "-i",
                str(input_path),
                "-i",
                str(watermark_path),
                "-t",
                str(duration_sec),
                "-filter_complex",
                filter_complex,
                "-map",
                "[out]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(output_path),
            ]
        else:
            # No watermark, just scale
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_sec),
                "-i",
                str(input_path),
                "-t",
                str(duration_sec),
                "-vf",
                "scale=-2:720",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(output_path),
            ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg stderr: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def concatenate_clips(clip_files: list[Path], output_path: Path) -> None:
    """Concatenate multiple clips into a single video."""
    if len(clip_files) == 1:
        # Just copy single clip
        clip_files[0].rename(output_path)
        return

    # Create concat file
    concat_file = output_path.parent / "concat.txt"
    with open(concat_file, "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")
    finally:
        # Clean up concat file
        concat_file.unlink(missing_ok=True)


def send_webhook(url: str, secret: str, payload: dict, raise_on_error: bool = False) -> None:
    """Send webhook notification.

    Args:
        url: Webhook URL
        secret: Webhook secret for authentication
        payload: JSON payload to send
        raise_on_error: If True, re-raise exceptions after logging (for critical webhooks)
    """
    print(f"Sending webhook to {url}: {payload}")
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"X-Webhook-Secret": secret, "Content-Type": "application/json"},
            timeout=30,
        )
        print(f"Webhook response: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Webhook failed: {e}")
        if raise_on_error:
            raise


def send_progress(url: str, secret: str, job_id: str, progress: int) -> None:
    """Send progress update."""
    # Extract base URL and construct progress endpoint
    parsed = urlparse(url)
    progress_url = f"{parsed.scheme}://{parsed.netloc}/v1/webhooks/export-progress"
    send_webhook(progress_url, secret, {"job_id": job_id, "progress": progress})
