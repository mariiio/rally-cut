"""Coherence (game-rule) invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-coherence-invariants`) wires them in alongside the
existing PID-structural audit.

Rules:
  C-1: A team can have at most 3 consecutive contacts before the ball
       must cross to the other team.
  C-2: Possessions alternate teams.
  C-3: First action of a rally is `serve`.
  C-4: Consecutive actions must be by different players (exception: prev
       action is `block`).

Skip semantics: each rule has explicit skip conditions. Additionally, the
orchestrator (`run_all`) excludes rallies that fail any I-1 / I-3 / I-6
PID invariant — those are upstream issues that would produce noisy
downstream coherence violations.

Spec: docs/superpowers/specs/2026-05-10-coherence-invariants-v1-design.md
"""

from __future__ import annotations

from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.pid_invariants import Violation
from rallycut.tracking.pid_invariants import run_all as pid_run_all


def _team_for_action(
    action: dict[str, Any], team_assignments: dict[str, str]
) -> str | None:
    """Resolve an action to its team label ('A'/'B'). None if undeterminable."""
    tid = action.get("playerTrackId")
    if tid is None:
        return None
    label = team_assignments.get(str(tid))
    if label not in ("A", "B"):
        return None
    return label


def _actions_sorted_by_frame(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Defensive sort by frame — actions in DB may be unordered."""
    return sorted(actions, key=lambda a: int(a.get("frame", 0)))


def check_c1_three_contact_rule(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-1: a team can have at most 3 consecutive contacts before crossing."""
    if len(actions) < 2:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)

    violations: list[Violation] = []
    current_team: str | None = None
    current_frames: list[int] = []

    for action in sorted_actions:
        team = _team_for_action(action, team_assignments)
        if team is None:
            return []  # defensive — orchestrator should have skipped
        frame = int(action.get("frame", 0))
        if team == current_team:
            current_frames.append(frame)
        else:
            current_team = team
            current_frames = [frame]
        if len(current_frames) == 4:
            # First time we hit 4 consecutive — emit one violation per
            # offending sequence (don't keep emitting at 5, 6, etc.).
            violations.append(
                Violation(
                    invariant="C-1",
                    rally_id=rally_id,
                    detail=(
                        f"team {current_team} had 4 consecutive contacts "
                        f"at frames {current_frames}; max is 3"
                    ),
                )
            )
    return violations


# Action types that always end the current possession (ball crosses to the
# other side or starts a new turn).
_POSSESSION_END_ACTIONS = {"attack", "serve"}


def check_c2_alternating_possessions(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-2: possessions alternate teams.

    Possession ends when:
      - The action is `attack` (ball crosses to other team).
      - The action is `serve` (rally turn starts).
      - The current team has accumulated 3 contacts.

    After a possession ends, the next action must be by the OTHER team.
    """
    if len(actions) < 2:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)

    violations: list[Violation] = []
    contacts_in_possession: int = 0
    last_team: str | None = None
    last_action: dict[str, Any] | None = None
    last_index: int = -1

    for idx, action in enumerate(sorted_actions):
        team = _team_for_action(action, team_assignments)
        if team is None:
            return []  # defensive
        action_type = str(action.get("action", ""))

        if last_team is not None and last_action is not None:
            last_action_type = str(last_action.get("action", ""))
            possession_ended = (
                last_action_type in _POSSESSION_END_ACTIONS
                or contacts_in_possession >= 3
            )
            if possession_ended:
                # New action must be the OTHER team.
                if team == last_team:
                    other = "B" if last_team == "A" else "A"
                    violations.append(
                        Violation(
                            invariant="C-2",
                            rally_id=rally_id,
                            detail=(
                                f"team {last_team} action[{last_index}] "
                                f"(frame {last_action.get('frame')}, "
                                f"{last_action_type}) ended possession; "
                                f"next action[{idx}] (frame {action.get('frame')}, "
                                f"{action_type}) was also team {team} — "
                                f"expected team {other}"
                            ),
                        )
                    )
                # Reset possession either way (don't keep cascading violations
                # if the same wrong team has multiple actions).
                contacts_in_possession = 1
            else:
                if team != last_team:
                    # Possession transferred without an end-action — also a violation
                    # (mid-possession crossover). For v1, just track and move on.
                    contacts_in_possession = 1
                else:
                    contacts_in_possession += 1
        else:
            contacts_in_possession = 1

        last_team = team
        last_action = action
        last_index = idx

    return violations


def check_c3_first_action_is_serve(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
) -> list[Violation]:
    """C-3: rally's first action must be `serve`."""
    if not actions:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)
    first = sorted_actions[0]
    first_type = str(first.get("action", ""))
    if first_type == "serve":
        return []
    return [
        Violation(
            invariant="C-3",
            rally_id=rally_id,
            detail=(
                f"first action is {first_type!r} (frame {first.get('frame')}); "
                f"expected 'serve'"
            ),
        )
    ]


def check_c4_no_same_player_back_to_back(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-4: consecutive actions must be by different players.

    Exception: when ``action[i-1].action == 'block'`` the pair is exempt —
    block→cover by the same player is legal (and so is the rarer
    block→block by the same player, since the prev is still block).

    Skip pair semantics: if either action's ``playerTrackId`` is missing,
    -1, or not in ``team_assignments``, the comparison is meaningless and
    the pair is skipped. This is the *audit* side, which surfaces all
    same-player pairs regardless of action confidence; the Phase 2 repair
    pass mirrors Pass 2c's ``confidence < 0.3`` floor instead.
    """
    if len(actions) < 2:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)

    violations: list[Violation] = []
    for i in range(1, len(sorted_actions)):
        prev = sorted_actions[i - 1]
        curr = sorted_actions[i]
        prev_pid = prev.get("playerTrackId")
        curr_pid = curr.get("playerTrackId")

        if prev_pid is None or curr_pid is None:
            continue
        if prev_pid == -1 or curr_pid == -1:
            continue
        if str(prev_pid) not in team_assignments:
            continue
        if str(curr_pid) not in team_assignments:
            continue

        prev_action = str(prev.get("action", ""))
        if prev_action == "block":
            continue  # strict block exception

        if prev_pid != curr_pid:
            continue

        curr_action = str(curr.get("action", ""))
        violations.append(
            Violation(
                invariant="C-4",
                rally_id=rally_id,
                detail=(
                    f"action[{i - 1}] (frame {prev.get('frame')}, "
                    f"{prev_action}, player {prev_pid}) and "
                    f"action[{i}] (frame {curr.get('frame')}, "
                    f"{curr_action}, player {curr_pid}) "
                    f"attributed to same player; max is 1 unless prev is block"
                ),
                payload={
                    "prev_index": i - 1,
                    "curr_index": i,
                    "prev_frame": int(prev.get("frame", 0)),
                    "curr_frame": int(curr.get("frame", 0)),
                    "prev_action": prev_action,
                    "curr_action": curr_action,
                    "player_id": int(prev_pid),
                },
            )
        )
    return violations


# PID invariants whose failures should exclude a rally from coherence checks.
# These directly affect action attribution / team labeling, so coherence
# violations on these rallies would be downstream noise.
_UPSTREAM_BLOCKER_INVARIANTS = frozenset({"I-1", "I-3", "I-6"})


def run_all(*, video_id: str) -> list[Violation]:
    """Run all 4 coherence invariants against a video's persisted state.

    Skips rallies that fail upstream PID invariants (I-1 / I-3 / I-6) to
    avoid flagging downstream effects of structural problems.
    """
    upstream, _stale = pid_run_all(video_id=video_id)
    excluded_rallies: set[str] = {
        v.rally_id for v in upstream
        if v.invariant in _UPSTREAM_BLOCKER_INVARIANTS
    }

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND (r.status = 'CONFIRMED' OR r.status IS NULL)
        ORDER BY r.start_ms
    """

    violations: list[Violation] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rally_rows = cur.fetchall()

    for row in rally_rows:
        rally_id = cast(str, row[0])
        if rally_id in excluded_rallies:
            continue
        actions_json = row[1]
        if not isinstance(actions_json, dict):
            continue
        actions = actions_json.get("actions")
        team_assignments = actions_json.get("teamAssignments")
        if not isinstance(actions, list):
            continue
        if not isinstance(team_assignments, dict):
            team_assignments = {}

        violations.extend(
            check_c1_three_contact_rule(
                rally_id=rally_id, actions=actions,
                team_assignments=team_assignments,
            )
        )
        violations.extend(
            check_c2_alternating_possessions(
                rally_id=rally_id, actions=actions,
                team_assignments=team_assignments,
            )
        )
        violations.extend(
            check_c3_first_action_is_serve(
                rally_id=rally_id, actions=actions,
            )
        )
        violations.extend(
            check_c4_no_same_player_back_to_back(
                rally_id=rally_id, actions=actions,
                team_assignments=team_assignments,
            )
        )

    return violations
