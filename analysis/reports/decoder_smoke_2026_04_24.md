# Parallel decoder smoke test — side-by-side vs legacy

- Rallies: 10
- Both paths share ball/player positions + MS-TCN++ probs
- Legacy: `detect_contacts` + `run_decoder_for_production` overlay + `classify_rally_actions`
- Decoder: `detect_contacts_via_decoder` + `build_rally_actions_from_decoder`

## Aggregate action counts

| Class | Legacy | Decoder | Δ |
|---|---:|---:|---:|
| serve | 9 | 6 | -3 |
| receive | 10 | 8 | -2 |
| set | 10 | 8 | -2 |
| attack | 12 | 8 | -4 |
| dig | 11 | 2 | -9 |
| block | 0 | 1 | +1 |
| unknown | 0 | 0 | +0 |

## Per-rally side-by-side

### `37e14e1e…` (073cb11b…) — 4 GT contacts

- **GT**: serve → receive → set → attack
- **Legacy** (5): serve → receive → set → attack → dig
- **Decoder** (3): serve → receive → set

- serve: legacy 1 vs decoder 1; dig: legacy 1 vs decoder 0; decoder synth-serves: 0

### `39139435…` (07fedbd4…) — 6 GT contacts

- **GT**: serve → receive → set → attack → dig → attack
- **Legacy** (6): serve → receive → set → attack → attack → dig
- **Decoder** (4): serve → receive → set → attack

- serve: legacy 1 vs decoder 1; dig: legacy 1 vs decoder 0; decoder synth-serves: 0

### `0ab56722…` (0a383519…) — 4 GT contacts

- **GT**: serve → receive → set → attack
- **Legacy** (5): serve → receive → set → attack → dig
- **Decoder** (2): serve → receive

- serve: legacy 1 vs decoder 1; dig: legacy 1 vs decoder 0; decoder synth-serves: 0

### `25edb83f…` (16458e78…) — 9 GT contacts

- **GT**: serve → receive → set → attack → dig → attack → dig → set → attack
- **Legacy** (7): receive → set → attack → dig → dig → set → attack
- **Decoder** (7): serve → receive → set → attack → dig → set → attack

- serve: legacy 0 vs decoder 1; dig: legacy 2 vs decoder 1; decoder synth-serves: 0

### `5447e090…` (1a5da176…) — 4 GT contacts

- **GT**: serve → receive → set → attack
- **Legacy** (5): serve → receive → set → attack → dig
- **Decoder** (1): attack

- serve: legacy 1 vs decoder 0; dig: legacy 1 vs decoder 0; decoder synth-serves: 0

### `73581b32…` (1efa35cf…) — 5 GT contacts

- **GT**: serve → receive → set → attack → dig
- **Legacy** (4): serve → receive → set → attack
- **Decoder** (3): receive → set → attack

- serve: legacy 1 vs decoder 0; dig: legacy 0 vs decoder 0; decoder synth-serves: 0

### `2391489e…` (20c11175…) — 5 GT contacts

- **GT**: serve → receive → set → attack → dig
- **Legacy** (3): serve → receive → dig
- **Decoder** (0): 

- serve: legacy 1 vs decoder 0; dig: legacy 1 vs decoder 0; decoder synth-serves: 0

### `0102cbba…` (211e2a4c…) — 4 GT contacts

- **GT**: serve → receive → set → attack
- **Legacy** (4): serve → receive → set → attack
- **Decoder** (4): serve → receive → set → attack

- serve: legacy 1 vs decoder 1; dig: legacy 0 vs decoder 0; decoder synth-serves: 0

### `032ec267…` (23a5f798…) — 8 GT contacts

- **GT**: serve → receive → set → attack → block → dig → set → attack
- **Legacy** (7): serve → receive → set → attack → dig → dig → attack
- **Decoder** (4): receive → set → attack → block

- serve: legacy 1 vs decoder 0; dig: legacy 2 vs decoder 0; decoder synth-serves: 0

### `d724bbf0…` (23b662ba…) — 5 GT contacts

- **GT**: serve → receive → set → attack → dig
- **Legacy** (6): serve → receive → set → attack → dig → dig
- **Decoder** (5): serve → receive → set → attack → dig

- serve: legacy 1 vs decoder 1; dig: legacy 2 vs decoder 1; decoder synth-serves: 0
