from rallycut.quality.types import Issue, Tier, QualityReport, CheckResult


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
