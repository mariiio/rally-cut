# Session 2 Harvest Summary

Generated: 2026-04-16 12:27:28 UTC

## Per-tier counts

| Tier | Harvested | Rejected | Rate | Gate |
|---|---:|---:|---:|---|
| gold | 4 | 2 | 50.0% | FAIL (<5%) |
| mid | 2422 | 24 | 1.0% | PASS (<15%) |
| positive | 2646 | 32 | 1.2% | - () |
| easy_neg | 2500 | 12 | 0.5% | - () |

## Train/val split (video-level)

- Train: 6200 pairs across 54 videos
- Val: 1372 pairs across 14 videos
- Split seed: 42, train fraction: 0.8

## Session 3 readiness

- Gold tier rejection gate (<5%): FAIL
- Mid tier rejection gate (<15%): PASS
- Gold volume ≥ 150: FAIL (4 actual)
- Overall Session 3 green light: NO

### Action needed

Gold tier is below the 150-pair Session 3 prerequisite. Schedule a Session 2b labeller-in-the-loop round: 10 rallies with dense near-net convergence × 10–15 hard pairs each. Defer Session 3 head-training until gold volume ≥ 150.
