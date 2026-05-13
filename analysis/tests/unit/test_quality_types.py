from rallycut.quality.types import CheckResult, CourtDetection, Issue, QualityReport, Tier


def test_issue_serializes_to_user_facing_dict():
    issue = Issue(
        id="camera_too_far",
        tier=Tier.GATE,
        severity=0.8,
        message="Camera is very far — player tracking may be less accurate.",
        source="preflight",
        data={"medianBboxHeight": 0.08},
    )
    out = issue.to_dict()
    assert out["id"] == "camera_too_far"
    assert out["tier"] == "gate"
    assert out["severity"] == 0.8
    assert out["source"] == "preflight"
    assert out["data"]["medianBboxHeight"] == 0.08


def test_report_picks_top_3_by_tier_then_severity():
    results = [
        CheckResult(issues=[Issue("c", Tier.ADVISORY, 0.9, "c", "preflight")]),
        CheckResult(issues=[Issue("a", Tier.BLOCK, 0.4, "a", "preflight")]),
        CheckResult(issues=[Issue("b", Tier.GATE, 0.95, "b", "preflight")]),
        CheckResult(issues=[Issue("d", Tier.GATE, 0.2, "d", "preflight")]),
    ]
    report = QualityReport.from_checks(results, source="preflight")
    ids = [i["id"] for i in report.to_dict()["issues"]]
    # Top-3 by (tier, -severity): block "a" first, then the two gates by severity ("b" 0.95, "d" 0.2)
    assert ids == ["a", "b", "d"]


def test_report_preserves_from_source_metadata():
    report = QualityReport.from_checks([], source="preflight", sample_seconds=60, duration_ms=12345)
    d = report.to_dict()
    assert d["preflight"]["sampleSeconds"] == 60
    assert d["preflight"]["durationMs"] == 12345


def test_report_emits_court_detection_when_present():
    corners = [
        {"x": 0.2, "y": 0.8},  # near-left
        {"x": 0.8, "y": 0.8},  # near-right
        {"x": 0.7, "y": 0.4},  # far-right
        {"x": 0.3, "y": 0.4},  # far-left
    ]
    report = QualityReport.from_checks(
        [],
        source="preflight",
        court=CourtDetection(corners=corners, confidence=0.87),
    )
    d = report.to_dict()
    assert d["court"]["confidence"] == 0.87
    assert d["court"]["corners"] == corners


def test_report_omits_court_field_when_absent():
    report = QualityReport.from_checks([], source="preflight")
    d = report.to_dict()
    assert "court" not in d
