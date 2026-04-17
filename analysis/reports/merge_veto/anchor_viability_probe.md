# Anchor Viability Probe

Measures within-team cosine similarity of Session-3 learned-ReID embeddings
across 43 GT rallies. A low cosine between teammates means the head can distinguish
them — a prerequisite for anchor-based identity propagation.

## Distribution of within-team cosine similarities

- **N pairs**: 77 (from 43 rallies)
- **Mean**: 0.745
- **Median**: 0.784
- **P25**: 0.663
- **P75**: 0.856

## Separation counts

- **Clear separation** (≥1 pair with cosine < 0.5): **7 / 43 rallies** (16%)
- **Ambiguous** (all pairs with cosine > 0.7): **22 / 43 rallies** (51%)

## Per-video best anchor

For each of 34 videos, the best (lowest cosine) rally across all rallies in that video.

| Video (short) | Best cosine | Viable? |
|:--------------|:------------|:-------:|
| `601d4a69` | 0.214 | yes |
| `635dcba2` | 0.302 | yes |
| `ce4c67a1` | 0.381 | yes |
| `4cd680c7` | 0.416 | yes |
| `cb3b68f0` | 0.445 | yes |
| `dd042609` | 0.459 | yes |
| `d5a6932f` | 0.550 | no |
| `ff175026` | 0.560 | no |
| `313c6c95` | 0.592 | no |
| `2e984c43` | 0.600 | no |
| `627c1add` | 0.622 | no |
| `ae81fff5` | 0.631 | no |
| `84e66e74` | 0.652 | no |
| `44e89f6c` | 0.656 | no |
| `16458e78` | 0.663 | no |
| `23b662ba` | 0.666 | no |
| `70ab9d7f` | 0.667 | no |
| `07fedbd4` | 0.701 | no |
| `304df1bf` | 0.712 | no |
| `3ad271bd` | 0.722 | no |
| `7ba71fc4` | 0.737 | no |
| `edc81438` | 0.740 | no |
| `0a383519` | 0.747 | no |
| `7d77980f` | 0.777 | no |
| `c6e4c876` | 0.792 | no |
| `56f2739d` | 0.794 | no |
| `1a5da176` | 0.794 | no |
| `fb83f876` | 0.823 | no |
| `b097dd2a` | 0.834 | no |
| `a5866029` | 0.882 | no |
| `a7ee3d38` | 0.908 | no |
| `1efa35cf` | 0.932 | no |
| `5c641cfe` | N/A | — |
| `920ba69d` | N/A | — |

**Videos with viable anchor (best cosine < 0.5): 6 / 34 (18%)**

## Verdict

**NOT VIABLE** — <30% of videos have a usable anchor rally (cosine < 0.5).

## Per-rally detail

| Rally (short) | Video (short) | Team-0 cos | Team-1 cos |
|:--------------|:--------------|:----------:|:----------:|
| `1bfcbc4f` | `07fedbd4` | 0.701 | 0.737 |
| `740ffd88` | `0a383519` | 0.847 | 0.747 |
| `21a9b203` | `0a383519` | 0.862 | 0.774 |
| `a43fb033` | `16458e78` | N/A | 0.663 |
| `fb8fd612` | `1a5da176` | 0.828 | 0.794 |
| `73581b32` | `1efa35cf` | 0.932 | N/A |
| `d724bbf0` | `23b662ba` | 0.831 | 0.666 |
| `21029e9f` | `2e984c43` | 0.600 | N/A |
| `29cb4e29` | `304df1bf` | 0.784 | 0.922 |
| `97f95cda` | `304df1bf` | 0.783 | 0.933 |
| `f6fa0cbb` | `304df1bf` | 0.712 | 0.914 |
| `572bff7e` | `313c6c95` | 0.592 | N/A |
| `8ce5a9e2` | `3ad271bd` | 0.834 | 0.769 |
| `2e8b3ce2` | `3ad271bd` | 0.766 | 0.722 |
| `e84deef3` | `3ad271bd` | 0.851 | 0.818 |
| `8c2d30ce` | `44e89f6c` | 0.656 | 0.917 |
| `1f87460b` | `4cd680c7` | 0.416 | 0.536 |
| `2dff5eeb` | `56f2739d` | 0.794 | 0.863 |
| `c3b31af2` | `5c641cfe` | N/A | N/A |
| `53ca3586` | `601d4a69` | 0.474 | 0.877 |
| `de7136d1` | `601d4a69` | 0.214 | 0.941 |
| `f0fdfcdb` | `627c1add` | 0.622 | 0.825 |
| `e5c1a9b3` | `635dcba2` | 0.302 | 0.426 |
| `9dbe457a` | `70ab9d7f` | 0.667 | 0.889 |
| `209be896` | `7ba71fc4` | 0.737 | 0.740 |
| `8b0b9e13` | `7d77980f` | 0.777 | 0.807 |
| `793625cd` | `84e66e74` | 0.831 | 0.652 |
| `0af554b5` | `920ba69d` | N/A | N/A |
| `bd77efd1` | `a5866029` | 0.926 | 0.882 |
| `0d84f858` | `a7ee3d38` | 0.933 | 0.908 |
| `7ff96129` | `ae81fff5` | 0.631 | 0.778 |
| `d4938222` | `ae81fff5` | N/A | 0.832 |
| `72c8229b` | `b097dd2a` | 0.834 | 0.856 |
| `fad29c31` | `c6e4c876` | 0.937 | 0.792 |
| `87ce7bff` | `cb3b68f0` | 0.661 | 0.636 |
| `b7f92cdc` | `cb3b68f0` | 0.445 | 0.762 |
| `9db9cb6b` | `ce4c67a1` | 0.694 | 0.381 |
| `072fb8c5` | `d5a6932f` | 0.550 | 0.864 |
| `21266995` | `dd042609` | 0.872 | 0.459 |
| `5e2e58fb` | `edc81438` | 0.842 | 0.817 |
| `0a376585` | `edc81438` | 0.740 | 0.879 |
| `d474b2ad` | `fb83f876` | 0.823 | 0.882 |
| `c48eeb7d` | `ff175026` | 0.808 | 0.560 |
