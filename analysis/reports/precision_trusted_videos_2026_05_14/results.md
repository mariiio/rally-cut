# Precision validation on 12 trusted-GT videos (2026-05-14)

GT coverage: **641** rows across **164** rallies in 12 videos.

Per-video rally counts:
- titi: 28
- toto: 27
- lulu: 21
- wawa: 10
- caco: 6
- cece: 5
- cici: 7
- cuco: 7
- gaga: 6
- kaka: 14
- juju: 14
- yeye: 19

## A. Overall pipeline attribution precision (S0 baseline)

| Metric | Value |
|---|---|
| GT rows checked | 641 |
| Pipeline contact present | 483 |
| Pipeline correct (pid matches GT) | 406/483 = 84.1% |
| Pipeline wrong same-team | 13 (2.7%) |
| Pipeline wrong cross-team | 64 (13.3%) |
| Pipeline missing (no contact at GT frame) | 158 (24.6%) |

## B. A1.v1 (always-flip-curr) simulation

Rule: for same-player consecutive pairs (block exception),
flip the *curr* action's pid to its closest same-team alternate.

| Metric | Value |
|---|---|
| Same-player pairs where rule changes pipeline | 27 |
| Of those, new pick matches GT | 4/27 = 14.8% |
| Of those, new pick doesn't match GT | 23 |
| Baseline-correct among fires (rule attacks correct picks) | 16 |
| Baseline-wrong among fires (rule has work to do) | 11 |
| Of those: rule fixed | 4 / 11 |
| Of those: rule broke a correct | 16 / 16 |
| **Net delta vs S0** | **-12** |

## C. A1.v2-confidence (flip lower-conf side) simulation — measuring curr-side only

| Metric | Value |
|---|---|
| Same-player pairs where curr changed | 10 |
| Of those, new curr pick matches GT | 3/10 = 30.0% |
| Baseline-correct among fires (rule attacks correct picks) | 5 |
| Baseline-wrong among fires (rule has work to do) | 5 |
| Of those: rule fixed | 3 / 5 |
| Of those: rule broke a correct | 5 / 5 |
| **Net delta vs S0 (curr-side only)** | **-2** |

Note: A1.v2-conf also re-attributes prev-side actions; that benefit is
captured implicitly when the prev becomes the next iteration's curr.

## D. S4 trajectory-integral + anti-self-touch simulation

| Metric | Value |
|---|---|
| Contacts S4 changed pipeline pick | 44 |
| Of those, new pick matches GT | 5/44 = 11.4% |
| Baseline-correct on S4-changes (rule attacks correct picks) | 30 |
| Baseline-wrong on S4-changes (rule has work to do) | 14 |
| Of those: rule fixed | 5 / 14 |
| Of those: rule broke a correct | 30 / 30 |
| **Net delta vs S0** | **-25** |
| S4 skipped (< 5 pre-ball frames) | 21 |

## E. Contact-detector at-net FP rate

| Metric | Value |
|---|---|
| Pipeline at-net attack contacts (12 videos) | 180 |
| Of which have NO ATTACK GT within ±5 frames | 72 = 40.0% (likely FP / non-event) |

Per video:

| Video | At-net attacks | Unmatched (no GT) | % unmatched |
|---|---|---|---|
| titi | 36 | 21 | 58.3% |
| toto | 29 | 19 | 65.5% |
| lulu | 23 | 19 | 82.6% |
| wawa | 6 | 1 | 16.7% |
| caco | 8 | 3 | 37.5% |
| cece | 5 | 0 | 0.0% |
| cici | 9 | 1 | 11.1% |
| cuco | 7 | 0 | 0.0% |
| gaga | 9 | 0 | 0.0% |
| kaka | 11 | 0 | 0.0% |
| juju | 16 | 0 | 0.0% |
| yeye | 21 | 8 | 38.1% |

## Per-video breakdown

| Video | GT rows | Pipeline-correct | A1.v1 net | A1.v2c net | S4 net |
|---|---|---|---|---|---|
| titi | 81 | 53/68 = 77.9% | -2 | +1 | -5 |
| toto | 57 | 37/43 = 86.0% | -1 | -1 | -1 |
| lulu | 36 | 21/24 = 87.5% | -1 | +0 | +0 |
| wawa | 45 | 23/25 = 92.0% | +0 | +0 | -3 |
| caco | 22 | 15/19 = 78.9% | +0 | +0 | -2 |
| cece | 29 | 15/18 = 83.3% | +0 | +0 | +0 |
| cici | 42 | 31/36 = 86.1% | +0 | +0 | +0 |
| cuco | 48 | 38/39 = 97.4% | -1 | -1 | -2 |
| gaga | 48 | 28/37 = 75.7% | -1 | +1 | -3 |
| kaka | 74 | 51/57 = 89.5% | -1 | +0 | -1 |
| juju | 80 | 44/57 = 77.2% | -2 | -1 | -2 |
| yeye | 79 | 50/60 = 83.3% | -3 | -1 | -6 |

## Per-action-type breakdown

| Type | GT n | Pipeline-correct | Wrong same-team | Wrong cross-team | Missing |
|---|---|---|---|---|---|
| SERVE | 97 | 67/78 = 85.9% | 2 | 9 | 19 |
| RECEIVE | 94 | 65/73 = 89.0% | 3 | 5 | 21 |
| SET | 145 | 95/118 = 80.5% | 5 | 18 | 27 |
| ATTACK | 187 | 122/142 = 85.9% | 2 | 18 | 45 |
| BLOCK | 12 | n/a | 0 | 0 | 12 |
| DIG | 106 | 57/72 = 79.2% | 1 | 14 | 34 |

## Verdict

- **Baseline pipeline precision** (on pipeline contacts that hit a GT row): 84.1% (406/483)
- **Baseline pipeline recall** (correct picks / all resolved GT rows): 63.3% (406/641)
  - 'Pipeline missing' = 158/641 = 24.6%. Includes both detection misses (no pipeline action within ±5) and action-type mismatches (e.g. BLOCK→attack accounts for 10/12 GT blocks).
- **A1.v1 fires**: 27 pairs; net delta -12 (fixes 4, breaks 16/16)
- **A1.v2-conf fires**: 10 pairs; net delta -2 (fixes 3, breaks 5/5)
- **S4 changes**: 44 contacts; net delta -25 (fixes 5, breaks 30/30)
- **S4 simulated precision** (S4-pick vs GT, where evaluable): 78.9%
- **At-net attack FP rate**: 40.0% (72/180)

### Interpretation

Pipeline attribution-precision on the 12 trusted videos is **decent**
(84% on the contacts the pipeline detected; 63% recall against all GT
rows because contact-detector recall + action-type confusion together
drop 25% of GT contacts).

All three rule families regress in net delta on this corpus:
- **A1.v1**: every single same-player-back-to-back pair where the rule fired
  on a baseline-correct curr broke that correct pick (16/16). The pair's
  C-4 violation usually means the *prev* action was the misattributed one,
  not the curr. Flipping curr is therefore biased to break correct picks.
- **A1.v2-conf**: same effect but milder (10 fires vs 27). Confidence isn't
  reliable enough to choose the right side to flip; baseline still wins.
- **S4 (traj-int + anti-self)**: 30/30 baseline-correct picks broken; the
  trajectory signal pulls strongly toward setter/digger positions that
  often aren't the actual toucher. Net -25 over 44 changes.

The **at-net attack FP rate is 40% overall** (72/180 contacts have NO
ATTACK GT within ±5 frames). Per-video this splits sharply: titi/toto/lulu/
yeye sit at 38-83% while juju/kaka/gaga/cuco/cici/cece sit at 0-11%. That
bimodality is either an upstream signal (some videos have many phantom
at-net attacks the user deleted) or a GT-coverage artifact (less-labeled
videos look like FPs by absence). The 6 clean-floor videos suggest at-net
attacks ARE largely real contacts when labeled; the 6 noisy ones merit
manual review before drawing inference.

### Recommendation

Continue labeling T2 only if the goal is to grow corpus for *learned*
attribution (PGM Phase B, embedding-based scoring). Rule-based families
(A1.v1, A1.v2-conf, S4) all regress on this corpus and should not ship.

Independent of attribution, the contact-detector has measurable issues to
address with the existing GT:
1. BLOCK class — pipeline produced only 3 blocks across 108 rallies; GT has 12.
   10/12 GT blocks are pipeline-labeled `attack`. Block re-classification is
   reachable today.
2. Pipeline-missing rate of 25% is a *recall* ceiling that no attribution
   rule can lift. Half (102/158) are no-contact-within-±5 frames — that's
   contact-detector recall debt, not attribution.
