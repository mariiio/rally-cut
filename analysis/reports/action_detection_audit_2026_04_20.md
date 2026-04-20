# Action Detection Error Audit — 2026-04-20

**Scope**: 1049 error records produced by `build_action_error_corpus.py` (rebuilt 2026-04-20 with newly-wired seq metadata per error). Read-only audit. Reframes the prior "emission is the ceiling" verdict in specific ways documented below.

**Canonical baseline** (LOO-per-video, unchanged): Contact F1 **88.0%**, Action Acc **91.2%**. This audit explains what remains and what would move it.

## 1. Corpus shape (fresh)

| Class | Count | Prior (Apr 19) |
|---|---:|---:|
| `wrong_player` | 660 | 647 |
| `FN_contact` | 252 | 262 |
| `wrong_action` | 137 | 149 |

FN subcategories:

| Subcategory | Count | Share |
|---|---:|---:|
| `rejected_by_classifier` | 154 | 61.1% |
| `no_player_nearby` | 43 | 17.1% |
| `deduplicated` | 26 | 10.3% |
| `ball_dropout` | 18 | 7.1% |
| `no_candidate` | 8 | 3.2% |
| `rejected_by_gates` | 3 | 1.2% |

## 2. What the seq-metadata enrichment revealed (three reframings)

### 2.1 `rejected_by_classifier`: GBM ≈ 0 but seq ≈ 1 on most

Crosstab of GBM classifier confidence vs MS-TCN++ seq peak (non-bg, ±5f window) over all 154 `rejected_by_classifier` FNs:

| GBM conf \ seq peak | <0.50 | [0.50, 0.80) | [0.80, 0.95) | ≥0.95 | total |
|---|---:|---:|---:|---:|---:|
| <0.05 | 0 | 11 | 15 | **66** | **92** |
| [0.05, 0.10) | 0 | 3 | 5 | 24 | 32 |
| [0.10, 0.20) | 0 | 3 | 3 | 14 | 20 |
| [0.20, 0.30) | 0 | 1 | 2 | 4 | 7 |
| [0.30, 0.40) | 0 | 0 | 0 | 1 | 1 |
| ≥0.40 | 0 | 0 | 0 | 2 | 2 |
| total | **0** | 18 | 25 | **111** | **154** |

Two striking facts:

1. **Zero FNs have seq < 0.50.** Every rejected-by-classifier FN has the sequence model firing above or near the `SEQ_RECOVERY_TAU=0.80` rescue gate. The GBM alone is systematically under-scoring real contacts the seq model sees.
2. **60% (92/154) have GBM conf < 0.05 AND seq ≥ 0.95.** Not a threshold problem — the two emissions *disagree by the maximum possible margin* on a sixth of all remaining FNs. A flat threshold cut cannot resolve this; a decoder that combines both signals can.

Floor-lowering simulation (do NOT deploy without FP measurement):

- Lower `SEQ_RECOVERY_CLF_FLOOR` 0.20 → 0.10: rescues **26/154**
- Lower to 0.05: rescues **55/154**
- No-seq-signal FNs that *no* gate tune can rescue: **0/154**

**Ship memo framing ("emission is the ceiling") is half right.** GBM emission is the ceiling for a *single-classifier* system. But we have MS-TCN++ emission available and the Viterbi decoder is the correct architecture for fusing them. This audit strengthens the case for decoder flip beyond what the ship memo captured.

### 2.2 `no_player_nearby`: 88% seq-endorsed → it's a distance-gate problem

| seq peak range | count | interpretation |
|---|---:|---|
| ≥ 0.80 | **38** | seq agrees there was a contact → distance/tracking gap |
| [0.50, 0.80) | 1 | borderline |
| < 0.50 | 4 | no seq endorsement → likely GT mislabel or real no-contact |

Per-action `seq ≥ 0.80`: dig 10, serve 9, receive 9, set 4, block 3, attack 3.

**38 FNs where the candidate fired, seq endorsed the contact, but no player was within the 0.15 radius gate.** A cheap win: relax `player_contact_radius` 0.15 → 0.20 ONLY when seq ≥ 0.80 at the candidate frame. FP risk is low because the seq gate is strict.

### 2.3 `wrong_action`: 46% decoder-rescuable, 54% seq-agrees-with-pred

| gt → pred | count | seq=gt ≥0.80 (decoder fix) | seq=pred |
|---|---:|---:|---:|
| attack → dig | 23 | 2 | **19** |
| dig → set | 21 | 4 | **17** |
| receive → serve | 17 | **17** | 0 |
| attack → set | 16 | 3 | **13** |
| set → receive | 11 | **10** | 1 |
| set → dig | 10 | 2 | 8 |
| attack → receive | 6 | 5 | 1 |
| dig → attack | 6 | 2 | 4 |
| serve → receive | 6 | **6** | 0 |
| set → attack | 5 | 3 | 1 |
| set → serve | 3 | 3 | 0 |
| attack → serve | 3 | 3 | 0 |
| attack → block | 3 | 3 | 0 |
| others (≤2) | 7 | 0 | 4 |

**63/137 (46%) have decoder-fix signature** (`seq_argmax == gt_action AND seq_prob ≥ 0.80`). Consistent with the ship memo's +4.3pp action accuracy — decoder captures roughly this share.

**74/137 (54%) have seq agreeing with the wrong prediction.** Decoder won't fix these. Dominant confusion pairs (attack↔dig, dig↔set, attack→set) fall here. Genuine emission ambiguity; fixing them needs better emission, not structural priors (→ ship memo's "per-candidate crop-level features" roadmap).

### 2.4 `wrong_player` (660): distinct problem; seq has no purchase

Wrong-player means contact + action are correct, attribution is wrong. Seq correlates with `gt_action` at ≥94% across all classes (seq sees action, not player). This is an **attribution/tracking** workstream, not an action-detection one.

| gt_action | total | seq=gt ≥0.80 | seq=pred |
|---|---:|---:|---:|
| attack | 169 | 164 | 164 |
| set | 155 | 151 | 152 |
| serve | 128 | 128 | 128 |
| receive | 116 | 114 | 115 |
| dig | 90 | 85 | 85 |
| block | 2 | 1 | 1 |

**This audit cannot resolve wrong_player.** It needs per-error analysis on pose-attribution source, team assignment, per-rally concentration — flagged as a separate **Attribution Audit Phase 1** workstream.

## 3. ROI-ranked decision matrix

Deltas vs 88.0% F1 / 91.2% Action Acc baseline. "Evidence" cites corpus data, not ship-memo projections.

| # | Fix | Evidence | ΔAction Acc | ΔF1 | Effort | Risk | Rec |
|---|---|---|---:|---:|---|---|---|
| a | **Flip Viterbi decoder default-on** | 63/137 wrong_action rescuable (§2.3); ship memo LOO | **+4.3pp** | 0pp | 1h + 2wk A/B | Low (block F1 → 0 accepted) | **SHIP** |
| b | **Seq-gated relaxation of `no_player_nearby`** (0.15→0.20 when seq ≥ 0.80) | 38/43 FNs seq-endorsed (§2.2) | +0 | **+1.0 to +1.5pp** (est. 30-35 TP, <10 FP) | 4-6h | Medium (needs orthogonality + LOO A/B) | **INVESTIGATE** |
| c | **Lower `SEQ_RECOVERY_CLF_FLOOR` 0.20 → 0.15** | +26 FNs rescuable at 0.10 (§2.1), FP impact unmeasured | 0 | **+0.5 to +1.0pp** (pending FP measure) | 30min sweep | Medium (floor was previously tuned) | **PROBE ONLY** via existing `sweep_sequence_recovery.py` |
| d | **Audit `wrong_player` 660** | Largest error class (63% of all errors), untaxonomized | n/a (own metric) | n/a | 1-2 days | Low | **SPIN UP** separate workstream |
| e | Per-candidate crop-level features | 74/137 wrong_action + 92 GBM-0 FNs need emission upgrade (§2.3, §2.1) | up to +3pp | up to +2-4pp | 2-3 weeks | High | **BACKLOG** (ship memo roadmap) |

### Dropped from consideration

| Rejected option | Evidence |
|---|---|
| More decoder tuning (skip_penalty, action emission variants) | Ship memo sweep: already converged at sp=1.0+action |
| VideoMAE / E2E-Spot / scene backbones | `videomae_contact_nogo_2026_04_19.md` — orthogonality + neg control |
| Serve candidate heuristics | `serve_candidate_generator_nogo_2026_04_20.md` — +0.15pp ceiling |
| Pattern A multi-arm rescue | `pattern_a_c_dormant_2026_04_17.md` — reverted, failed 364-rally sweep |

## 4. What this audit overturns vs confirms

### Confirms
- **Ship memo's +4.3pp action accuracy is real and worth flipping the flag** — §2.3 finds 63 decoder-rescuable pairs, consistent with the +4.3pp over 1781 matched contacts.
- **VideoMAE/E2E-Spot/serve-heuristics NO-GOs remain valid** — none of the remaining FNs have a no-signal profile that a scene-level model would rescue.
- **Block F1 is structurally hard** — 22 block FNs in `rejected_by_classifier` are all seq-endorsed, but decoder *decreases* block F1 (12→0) because MS-TCN++ block signal is weak at training (30 examples). Not fixable this cycle.

### Overturns
- **"Emission is the ceiling."** Only true for single-classifier framing. Both emissions (GBM + MS-TCN++) disagree strongly on 92 FNs — decoder-actionable. The ceiling moves once the decoder is deployed.
- **"`no_player_nearby` is a hard attribution/tracking bottleneck."** 88% of these FNs have seq endorsement. Seq-gated distance-radius relaxation is cheap and recoverable. Not previously scoped.
- **"`rejected_by_classifier` is one uniform bucket."** It's two distinct populations: (i) conf < 0.05 with seq ≈ 1 (92 FNs, only decoder fixes), (ii) conf ∈ [0.05, 0.20) with seq endorsement (~62 FNs, floor tweak or decoder both viable). Lumping them obscured the signal-disagreement diagnosis.
- **Biggest error class (`wrong_player` 660) has never been audited.** Prior investigations focused on FN/wrong_action. Wrong_player is 63% of all corpus errors. Needs its own workstream.

## 5. Concrete next actions (by effort)

1. **Flip `ENABLE_CANDIDATE_DECODER=1` default-on.** Requires integration point in `detect_contacts()` or `classify_rally_actions` (decoder module is shipped but the flag is referenced only in documentation — not yet wired in code). 1h code + 2wk A/B.
2. **Wire seq-gated `player_contact_radius` relaxation.** New branch in `_find_nearest_player()` fallback path. Measure via `eval_loo_video.py` A/B + `orthogonality_probe.py` negative control. 4-6h.
3. **Kick off Attribution Audit Phase 1.** Scope a separate memo + diagnostic script for the 660 wrong_player bucket. 1-2 days.
4. **Probe `SEQ_RECOVERY_CLF_FLOOR` 0.20 → 0.15.** Cheap sanity via `scripts/sweep_sequence_recovery.py`. 30 min.

## 6. Artifacts

- `outputs/action_errors/corpus.jsonl` — rebuilt 2026-04-20 with seq metadata per record (1049 errors)
- `outputs/action_errors/rally_quality.json` — per-rally quality signals (updated)
- `outputs/action_errors/corpus_annotated.jsonl` — errors joined with rally quality
- `scripts/audit_action_errors_2026_04_20.py` — this audit, rerunnable
- `scripts/diagnose_fn_contacts.py` — extended `FNDiagnostic` with seq fields; accepts `sequence_probs` parameter
- `scripts/build_action_error_corpus.py` — threads `sequence_probs` into FN diagnostic; also attaches seq metadata on `wrong_action` / `wrong_player` records

## 7. Honest caveats

- The 46% decoder-rescuable share (§2.3) is a *signature match*, not a decoder run. Actual decoder behaviour depends on transition priors; some matches may fail on infeasible preceding-action context. Ship memo's +4.3pp is the authoritative measure; §2.3 is the per-error lens.
- Floor-lowering counts (§2.1) assume symmetric application on non-FN candidates as well; FP impact unmeasured. Do not ship without running `sweep_sequence_recovery.py`.
- `no_player_nearby` seq-gated relaxation (§2.2) must pass `orthogonality_probe.py` + negative control before any production commit.
- Wrong_player analysis is out of scope here — numbers listed for sizing only.
