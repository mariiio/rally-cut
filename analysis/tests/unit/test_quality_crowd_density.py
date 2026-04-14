from rallycut.quality.crowd_density import check_crowd_density
from rallycut.quality.camera_distance import Detection
from rallycut.quality.types import Tier


def test_empty_court_passes():
    # 4 players, 0 spectators
    dets = [[Detection(0.5, 0.5, 0.1, 0.3) for _ in range(4)] for _ in range(10)]
    court_bbox = (0.2, 0.3, 0.8, 0.9)  # xmin, ymin, xmax, ymax
    result = check_crowd_density(dets, court_bbox)
    assert result.issues == []


def test_many_spectators_outside_court_gates():
    # 4 players on court + 12 people outside
    players = [Detection(0.5, 0.5, 0.1, 0.3) for _ in range(4)]
    spectators = [Detection(0.05 + i * 0.01, 0.1, 0.05, 0.15) for i in range(12)]
    dets = [players + spectators for _ in range(10)]
    court_bbox = (0.2, 0.3, 0.8, 0.9)
    result = check_crowd_density(dets, court_bbox)
    issue = next(i for i in result.issues if i.id == "crowded_scene")
    assert issue.tier == Tier.GATE


def test_people_inside_court_not_counted_as_crowd():
    # 6 people all inside court bbox — still a normal game
    dets = [[Detection(0.5, 0.5, 0.1, 0.3) for _ in range(6)] for _ in range(10)]
    court_bbox = (0.2, 0.3, 0.8, 0.9)
    result = check_crowd_density(dets, court_bbox)
    assert result.issues == []
