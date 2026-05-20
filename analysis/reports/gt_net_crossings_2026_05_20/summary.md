# GT-derived Net-Crossing Probe — Summary (2026-05-20)

Substrate: 248 trusted-32 rallies, 1265 consecutive GT-contact pairs.
GT crossings: **546**, non-crossings: **719**, skipped: 0.

## Walker accuracy on GT crossings

- correct_flip (prev_action in {serve,attack}): 482/546 (88.3%)
- **missed_flip (walker didn't flip but should have): 64/546**

### Missed flips — by prev_action_type:

| prev_action | count |
|---|---:|
| set | 20 |
| dig | 17 |
| receive | 14 |
| block | 13 |

### Missed flips — by transition pattern:

| prev->curr | count |
|---|---:|
| set->attack | 18 |
| block->dig | 12 |
| dig->attack | 7 |
| receive->set | 6 |
| dig->set | 6 |
| receive->dig | 5 |
| dig->dig | 4 |
| receive->attack | 3 |
| set->dig | 2 |
| block->attack | 1 |

## Walker accuracy on GT non-crossings

- correct_stay (prev_action NOT in {serve,attack}): 699/719 (97.2%)
- **over_flip (walker would flip but shouldn't): 20/719**

### Over-flips — by prev_action_type:

| prev_action | count |
|---|---:|
| attack | 11 |
| serve | 9 |

### Over-flips — by transition pattern:

| prev->curr | count |
|---|---:|
| serve->receive | 9 |
| attack->dig | 8 |
| attack->attack | 2 |
| attack->block | 1 |

## Interpretation

- High `missed_flip` count for a non-{serve,attack} action → that action should be added to `_NET_CROSSING_ACTIONS`, OR the action is being mis-typed by the classifier when it actually IS the net-crossing event.
- High `over_flip` count for a transition pattern → walker incorrectly flips on those rallies; needs a guard.
- BLOCK is documented as non-net-crossing in the walker; if many crossings have prev=block, the documentation/code is wrong (block-cover does cross sometimes).
