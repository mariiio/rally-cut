"""Shared webhook payload construction for detection service runners."""

from __future__ import annotations

from rallycut.service.schemas import DetectionResponse


def build_detection_webhook_payload(
    response: DetectionResponse,
    job_id: str,
    *,
    result_s3_key: str | None = None,
) -> dict:
    """Build the webhook payload that the API expects from detection completion.

    API expects: {job_id, status, rallies: [{start_ms, end_ms}], suggested_rallies, ...}
    DetectionResponse has: {segments: [{start_time, end_time, segment_type}], ...}
    """
    rallies = []
    for segment in response.segments:
        if segment.segment_type.value == "rally":
            rallies.append({
                "start_ms": int(segment.start_time * 1000),
                "end_ms": int(segment.end_time * 1000),
            })

    suggested_rallies = []
    for sugg in response.suggested_segments:
        suggested_rallies.append({
            "start_ms": int(sugg.start_time * 1000),
            "end_ms": int(sugg.end_time * 1000),
            "confidence": sugg.avg_confidence,
            "rejection_reason": sugg.rejection_reason.value,
        })

    payload: dict = {
        "job_id": job_id,
        "status": "completed",
        "rallies": rallies,
        "suggested_rallies": suggested_rallies,
    }
    if result_s3_key is not None:
        payload["result_s3_key"] = result_s3_key

    return payload
