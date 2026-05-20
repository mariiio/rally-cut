# Chain Asymmetry Diagnostic — Summary (2026-05-20)

Substrate: 99 chain-disagreement contacts (from 151 processed).
Direction: 65 pipe=A/gt=B vs 34 pipe=B/gt=A.

## H1 — Serving-team detection bias

- First-serve attribution WRONG: **0/99 (0.0%)**
- Of wrong first-serves, seeded to A: 0, to B: 0
- H1 explains pipe=A errors: 0/65
- H1 explains pipe=B errors: 0/34
- **H1 dominant (≥60%):** NO

## H2 — team_assignments labeling skew

- Disagreement-rally avg A: 2.00, avg B: 2.01
- Significant A-skew (avg_A > 1.5x avg_B): NO

## H3 — Chain-walker init/transition bug

- Synthetic-seed: **26/99 (26.3%)**
- Chain integrity False at contact: **11/99 (11.1%)**
- Has UNKNOWN actions before contact: 0/99 (0.0%)
- **H3 dominant (≥60%):** NO

## Primary-cause assignment

- unexplained: 62/99 (62.6%)
- H3: 37/99 (37.4%)

## Verdict

- **No single hypothesis dominates.** Escalate to full chain-quality rewrite (separate brainstorm cycle).
