# Anchor Viability Probe

Measures within-team cosine similarity of Session-3 learned-ReID embeddings
across 43 GT rallies. A low cosine between teammates means the head can distinguish
them — a prerequisite for anchor-based identity propagation.

## Distribution of within-team cosine similarities

- **N pairs**: 86 (from 43 rallies)
- **Mean**: 0.789
- **Median**: 0.818
- **P25**: 0.705
- **P75**: 0.902

## Separation counts

- **Clear separation** (≥1 pair with cosine < 0.5): **2 / 43 rallies** (5%)
- **Ambiguous** (all pairs with cosine > 0.7): **27 / 43 rallies** (63%)

## Per-video best anchor

For each of 34 videos, the best (lowest cosine) rally across all rallies in that video.

| Video (short) | Best cosine | Viable? |
|:--------------|:------------|:-------:|
| `ff175026` | 0.420 | yes |
| `4cd680c7` | 0.480 | yes |
| `2e984c43` | 0.518 | no |
| `ce4c67a1` | 0.520 | no |
| `ae81fff5` | 0.523 | no |
| `5c641cfe` | 0.544 | no |
| `635dcba2` | 0.568 | no |
| `84e66e74` | 0.583 | no |
| `7d77980f` | 0.603 | no |
| `70ab9d7f` | 0.619 | no |
| `cb3b68f0` | 0.631 | no |
| `d5a6932f` | 0.632 | no |
| `3ad271bd` | 0.671 | no |
| `313c6c95` | 0.678 | no |
| `627c1add` | 0.678 | no |
| `07fedbd4` | 0.711 | no |
| `44e89f6c` | 0.715 | no |
| `0a383519` | 0.731 | no |
| `16458e78` | 0.733 | no |
| `601d4a69` | 0.753 | no |
| `7ba71fc4` | 0.788 | no |
| `304df1bf` | 0.814 | no |
| `1a5da176` | 0.840 | no |
| `23b662ba` | 0.860 | no |
| `dd042609` | 0.863 | no |
| `fb83f876` | 0.863 | no |
| `b097dd2a` | 0.867 | no |
| `c6e4c876` | 0.875 | no |
| `edc81438` | 0.886 | no |
| `56f2739d` | 0.899 | no |
| `1efa35cf` | 0.909 | no |
| `a7ee3d38` | 0.943 | no |
| `a5866029` | 0.945 | no |
| `920ba69d` | 0.951 | no |

**Videos with viable anchor (best cosine < 0.5): 2 / 34 (6%)**

## Verdict

**NOT VIABLE** — <30% of videos have a usable anchor rally (cosine < 0.5).

## Per-rally detail

| Rally (short) | Video (short) | Team-0 cos | Team-1 cos |
|:--------------|:--------------|:----------:|:----------:|
| `1bfcbc4f` | `07fedbd4` | 0.735 | 0.711 |
| `740ffd88` | `0a383519` | 0.830 | 0.918 |
| `21a9b203` | `0a383519` | 0.801 | 0.731 |
| `a43fb033` | `16458e78` | 0.801 | 0.733 |
| `fb8fd612` | `1a5da176` | 0.840 | 0.891 |
| `73581b32` | `1efa35cf` | 0.931 | 0.909 |
| `d724bbf0` | `23b662ba` | 0.860 | 0.871 |
| `21029e9f` | `2e984c43` | 0.518 | 0.892 |
| `29cb4e29` | `304df1bf` | 0.849 | 0.964 |
| `97f95cda` | `304df1bf` | 0.814 | 0.956 |
| `f6fa0cbb` | `304df1bf` | 0.886 | 0.959 |
| `572bff7e` | `313c6c95` | 0.678 | 0.839 |
| `8ce5a9e2` | `3ad271bd` | 0.763 | 0.671 |
| `2e8b3ce2` | `3ad271bd` | 0.704 | 0.758 |
| `e84deef3` | `3ad271bd` | 0.802 | 0.784 |
| `8c2d30ce` | `44e89f6c` | 0.715 | 0.918 |
| `1f87460b` | `4cd680c7` | 0.654 | 0.480 |
| `2dff5eeb` | `56f2739d` | 0.899 | 0.934 |
| `c3b31af2` | `5c641cfe` | 0.581 | 0.544 |
| `53ca3586` | `601d4a69` | 0.804 | 0.985 |
| `de7136d1` | `601d4a69` | 0.753 | 0.981 |
| `f0fdfcdb` | `627c1add` | 0.678 | 0.909 |
| `e5c1a9b3` | `635dcba2` | 0.568 | 0.723 |
| `9dbe457a` | `70ab9d7f` | 0.619 | 0.857 |
| `209be896` | `7ba71fc4` | 0.788 | 0.903 |
| `8b0b9e13` | `7d77980f` | 0.603 | 0.720 |
| `793625cd` | `84e66e74` | 0.826 | 0.583 |
| `0af554b5` | `920ba69d` | 0.960 | 0.951 |
| `bd77efd1` | `a5866029` | 0.970 | 0.945 |
| `0d84f858` | `a7ee3d38` | 0.951 | 0.943 |
| `7ff96129` | `ae81fff5` | 0.543 | 0.599 |
| `d4938222` | `ae81fff5` | 0.523 | 0.625 |
| `72c8229b` | `b097dd2a` | 0.867 | 0.910 |
| `fad29c31` | `c6e4c876` | 0.974 | 0.875 |
| `87ce7bff` | `cb3b68f0` | 0.770 | 0.792 |
| `b7f92cdc` | `cb3b68f0` | 0.631 | 0.823 |
| `9db9cb6b` | `ce4c67a1` | 0.744 | 0.520 |
| `072fb8c5` | `d5a6932f` | 0.632 | 0.774 |
| `21266995` | `dd042609` | 0.875 | 0.863 |
| `5e2e58fb` | `edc81438` | 0.886 | 0.896 |
| `0a376585` | `edc81438` | 0.901 | 0.905 |
| `d474b2ad` | `fb83f876` | 0.863 | 0.933 |
| `c48eeb7d` | `ff175026` | 0.547 | 0.420 |
