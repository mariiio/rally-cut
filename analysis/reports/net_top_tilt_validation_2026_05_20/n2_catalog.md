# Probe N2 — Tilt prevalence + solvePnP visual catalog

Samples: 77 videos. Bucket thresholds by |solvePnP tilt|.

## Bucket counts

| bucket | |tilt| range | count |
|--------|--------------|-------|
| flat | 0.000–0.005 | 30 |
| mild | 0.005–0.015 | 29 |
| notable | 0.015–0.030 | 17 |
| pronounced | 0.030–1.000 | 1 |

## Decision rule

* `notable + pronounced` ≤ 5 → tilt is rare; close as low-EV.
* Orange line hugs the visible net on tilted cases → C2 / C3 viable.
* Orange line is wrong on tilted cases → solvePnP teacher unreliable; C1 only.


## flat (30)

| name | fps | gt | M4 | |Δ|M4 | NLE mid |Δ| | tilt | conf | warnings | frame |
|------|-----|----|----|------|---------------|------|------|----------|-------|
| juju | 30.0 | 0.499 | 0.491 | 0.0084 | 0.0266 | +0.0050 | 1.00 |  | ![](frames/flat/juju.jpg) |
| vuvu | 59.9 | 0.264 | 0.268 | 0.0045 | 0.0070 | -0.0048 | 1.00 |  | ![](frames/flat/vuvu.jpg) |
| roro | 30.0 | 0.389 | 0.386 | 0.0028 | 0.0166 | +0.0048 | 1.00 |  | ![](frames/flat/roro.jpg) |
| kuku | 59.9 | 0.413 | 0.430 | 0.0170 | 0.0001 | -0.0046 | 1.00 |  | ![](frames/flat/kuku.jpg) |
| vava | 30.0 | 0.385 | 0.379 | 0.0058 | 0.0046 | +0.0043 | 1.00 |  | ![](frames/flat/vava.jpg) |
| popo | 30.0 | 0.404 | 0.409 | 0.0053 | 0.0132 | +0.0042 | 1.00 |  | ![](frames/flat/popo.jpg) |
| lili | 30.0 | 0.399 | 0.408 | 0.0089 | 0.0101 | -0.0035 | 1.00 |  | ![](frames/flat/lili.jpg) |
| jojo | 30.0 | 0.439 | 0.417 | 0.0216 | 0.0221 | +0.0032 | 1.00 |  | ![](frames/flat/jojo.jpg) |
| pupu | 30.0 | 0.341 | 0.350 | 0.0081 | 0.0166 | -0.0030 | 1.00 |  | ![](frames/flat/pupu.jpg) |
| koko | 30.0 | 0.413 | 0.437 | 0.0235 | 0.0085 | -0.0030 | 1.00 |  | ![](frames/flat/koko.jpg) |
| cuco | 30.0 | 0.259 | 0.261 | 0.0015 | 0.0149 | +0.0029 | 1.00 |  | ![](frames/flat/cuco.jpg) |
| kaka | 59.9 | 0.463 | 0.466 | 0.0031 | 0.0097 | +0.0028 | 1.00 |  | ![](frames/flat/kaka.jpg) |
| rara | 30.0 | 0.368 | 0.366 | 0.0023 | 0.0061 | -0.0027 | 1.00 |  | ![](frames/flat/rara.jpg) |
| ruru | 60.0 | 0.361 | 0.363 | 0.0014 | 0.0043 | -0.0027 | 1.00 |  | ![](frames/flat/ruru.jpg) |
| papa | 30.0 | 0.393 | 0.395 | 0.0022 | 0.0024 | -0.0026 | 1.00 |  | ![](frames/flat/papa.jpg) |
| tutu | 29.9 | 0.413 | 0.402 | 0.0102 | 0.0028 | -0.0025 | 1.00 |  | ![](frames/flat/tutu.jpg) |
| pipi | 30.0 | 0.395 | 0.395 | 0.0003 | 0.0172 | +0.0024 | 1.00 |  | ![](frames/flat/pipi.jpg) |
| cucu | 30.0 | 0.354 | 0.344 | 0.0100 | 0.0058 | -0.0024 | 1.00 |  | ![](frames/flat/cucu.jpg) |
| vivi | 59.3 | 0.367 | 0.354 | 0.0137 | 0.0016 | -0.0023 | 1.00 |  | ![](frames/flat/vivi.jpg) |
| macho | 30.0 | 0.350 | 0.364 | 0.0143 | 0.0083 | +0.0020 | 1.00 |  | ![](frames/flat/macho.jpg) |
| yeye | 29.7 | 0.369 | 0.367 | 0.0027 | 0.0124 | +0.0019 | 1.00 |  | ![](frames/flat/yeye.jpg) |
| gigi | 30.0 | 0.331 | 0.329 | 0.0025 | 0.0077 | +0.0015 | 1.00 |  | ![](frames/flat/gigi.jpg) |
| matttch | 30.0 | 0.293 | 0.300 | 0.0071 | 0.0073 | +0.0013 | 1.00 | sanity_failed | ![](frames/flat/matttch.jpg) |
| rere | 30.0 | 0.383 | 0.393 | 0.0094 | 0.0057 | +0.0013 | 1.00 |  | ![](frames/flat/rere.jpg) |
| muchi | 30.0 | 0.410 | 0.407 | 0.0027 | 0.0067 | +0.0012 | 1.00 |  | ![](frames/flat/muchi.jpg) |
| gugu | 29.6 | 0.313 | 0.306 | 0.0072 | 0.0028 | +0.0010 | 1.00 |  | ![](frames/flat/gugu.jpg) |
| lele | 29.4 | 0.414 | 0.407 | 0.0074 | 0.0089 | -0.0004 | 1.00 |  | ![](frames/flat/lele.jpg) |
| matchop | 30.0 | 0.244 | 0.261 | 0.0176 | 0.0131 | +0.0003 | 1.00 |  | ![](frames/flat/matchop.jpg) |
| mech | 30.0 | 0.244 | 0.268 | 0.0240 | 0.0208 | +0.0002 | 1.00 |  | ![](frames/flat/mech.jpg) |
| wewe | 59.9 | 0.188 | 0.193 | 0.0052 | 0.0132 | -0.0002 | 1.00 |  | ![](frames/flat/wewe.jpg) |

## mild (29)

| name | fps | gt | M4 | |Δ|M4 | NLE mid |Δ| | tilt | conf | warnings | frame |
|------|-----|----|----|------|---------------|------|------|----------|-------|
| matchc | 59.9 | 0.423 | 0.413 | 0.0100 | 0.0138 | +0.0134 | 1.00 |  | ![](frames/mild/matchc.jpg) |
| wowo | 59.9 | 0.265 | 0.267 | 0.0017 | 0.0047 | -0.0134 | 1.00 |  | ![](frames/mild/wowo.jpg) |
| mochi | 30.0 | 0.355 | 0.363 | 0.0076 | 0.0117 | +0.0119 | 1.00 |  | ![](frames/mild/mochi.jpg) |
| vovo | 59.9 | 0.251 | 0.248 | 0.0025 | 0.0026 | -0.0117 | 1.00 |  | ![](frames/mild/vovo.jpg) |
| haha | 59.9 | 0.473 | 0.464 | 0.0090 | 0.0298 | +0.0116 | 1.00 |  | ![](frames/mild/haha.jpg) |
| match | 30.0 | 0.210 | 0.201 | 0.0091 | 0.0010 | +0.0112 | 1.00 | sanity_failed | ![](frames/mild/match.jpg) |
| toto | 29.9 | 0.396 | 0.382 | 0.0141 | 0.0066 | +0.0111 | 1.00 |  | ![](frames/mild/toto.jpg) |
| yeye | 59.9 | 0.365 | 0.367 | 0.0014 | 0.0046 | +0.0110 | 1.00 | sanity_failed | ![](frames/mild/yeye.jpg) |
| mama | 30.0 | 0.259 | 0.265 | 0.0051 | 0.0136 | -0.0104 | 1.00 |  | ![](frames/mild/mama.jpg) |
| yiyi | 59.9 | 0.267 | 0.290 | 0.0224 | 0.0168 | +0.0102 | 1.00 | sanity_failed | ![](frames/mild/yiyi.jpg) |
| hehe | 59.9 | 0.487 | 0.496 | 0.0089 | 0.0145 | +0.0096 | 1.00 |  | ![](frames/mild/hehe.jpg) |
| riri | 30.0 | 0.327 | 0.330 | 0.0026 | 0.0060 | +0.0095 | 1.00 |  | ![](frames/mild/riri.jpg) |
| caco | 30.0 | 0.368 | 0.369 | 0.0008 | 0.0073 | -0.0089 | 1.00 |  | ![](frames/mild/caco.jpg) |
| gaga | 29.8 | 0.374 | 0.365 | 0.0083 | 0.0018 | +0.0089 | 1.00 |  | ![](frames/mild/gaga.jpg) |
| pepe | 30.0 | 0.331 | 0.317 | 0.0139 | 0.0043 | -0.0089 | 1.00 |  | ![](frames/mild/pepe.jpg) |
| moma | 30.0 | 0.274 | 0.280 | 0.0054 | 0.0138 | +0.0087 | 1.00 |  | ![](frames/mild/moma.jpg) |
| wiwi | 60.0 | 0.244 | 0.248 | 0.0049 | 0.0035 | -0.0086 | 1.00 |  | ![](frames/mild/wiwi.jpg) |
| veve | 30.0 | 0.354 | 0.350 | 0.0044 | 0.0136 | -0.0084 | 1.00 |  | ![](frames/mild/veve.jpg) |
| keke | 30.0 | 0.457 | 0.462 | 0.0051 | 0.0019 | +0.0083 | 1.00 |  | ![](frames/mild/keke.jpg) |
| tata | 30.0 | 0.402 | 0.405 | 0.0028 | 0.0215 | +0.0081 | 1.00 |  | ![](frames/mild/tata.jpg) |
| titi | 30.0 | 0.435 | 0.430 | 0.0044 | 0.0123 | +0.0076 | 1.00 |  | ![](frames/mild/titi.jpg) |
| matahtach | 30.0 | 0.482 | 0.458 | 0.0237 | 0.0352 | +0.0072 | 1.00 |  | ![](frames/mild/matahtach.jpg) |
| mechi | 25.0 | 0.319 | 0.355 | 0.0361 | 0.0226 | +0.0071 | 0.93 |  | ![](frames/mild/mechi.jpg) |
| matchope | 30.0 | 0.272 | 0.277 | 0.0051 | 0.0058 | -0.0067 | 1.00 |  | ![](frames/mild/matchope.jpg) |
| lolo | 30.0 | 0.401 | 0.387 | 0.0140 | 0.0007 | -0.0066 | 1.00 |  | ![](frames/mild/lolo.jpg) |
| kiki | 59.9 | 0.441 | 0.447 | 0.0058 | 0.0033 | +0.0055 | 1.00 |  | ![](frames/mild/kiki.jpg) |
| mame | 30.0 | 0.436 | 0.437 | 0.0008 | 0.0013 | +0.0053 | 1.00 |  | ![](frames/mild/mame.jpg) |
| yaya | 30.0 | 0.449 | 0.227 | 0.2221 | 0.0181 | +0.0052 | 1.00 |  | ![](frames/mild/yaya.jpg) |
| caca | 60.0 | 0.352 | 0.361 | 0.0096 | 0.0224 | +0.0050 | 0.98 |  | ![](frames/mild/caca.jpg) |

## notable (17)

| name | fps | gt | M4 | |Δ|M4 | NLE mid |Δ| | tilt | conf | warnings | frame |
|------|-----|----|----|------|---------------|------|------|----------|-------|
| jiji | 30.0 | 0.331 | 0.301 | 0.0304 | 0.0052 | +0.0298 | 1.00 |  | ![](frames/notable/jiji.jpg) |
| dark | 30.0 | 0.383 | 0.386 | 0.0029 | 0.0088 | +0.0284 | 1.00 |  | ![](frames/notable/dark.jpg) |
| wawa | 59.9 | 0.307 | 0.311 | 0.0037 | 0.0127 | -0.0275 | 1.00 |  | ![](frames/notable/wawa.jpg) |
| wuwu | 59.9 | 0.295 | 0.286 | 0.0095 | 0.0034 | -0.0266 | 1.00 |  | ![](frames/notable/wuwu.jpg) |
| jeje | 30.0 | 0.351 | 0.347 | 0.0038 | 0.0089 | +0.0262 | 1.00 |  | ![](frames/notable/jeje.jpg) |
| machi | 25.0 | 0.264 | 0.286 | 0.0212 | 0.0419 | +0.0239 | 0.91 |  | ![](frames/notable/machi.jpg) |
| mimi | 30.0 | 0.382 | 0.382 | 0.0004 | 0.0082 | +0.0238 | 0.99 |  | ![](frames/notable/mimi.jpg) |
| yaya | 59.9 | 0.199 | 0.227 | 0.0280 | 0.0465 | +0.0226 | 1.00 |  | ![](frames/notable/yaya.jpg) |
| michu | 30.0 | 0.321 | 0.337 | 0.0166 | 0.0039 | +0.0225 | 1.00 |  | ![](frames/notable/michu.jpg) |
| meme | 30.0 | 0.367 | 0.382 | 0.0145 | 0.0061 | +0.0205 | 1.00 |  | ![](frames/notable/meme.jpg) |
| mumu | 30.0 | 0.410 | 0.395 | 0.0143 | 0.0049 | +0.0203 | 1.00 |  | ![](frames/notable/mumu.jpg) |
| lulu | 59.9 | 0.321 | 0.293 | 0.0275 | 0.0159 | +0.0195 | 1.00 |  | ![](frames/notable/lulu.jpg) |
| tete | 30.0 | 0.358 | 0.358 | 0.0003 | 0.0019 | -0.0190 | 1.00 |  | ![](frames/notable/tete.jpg) |
| lala | 29.8 | 0.321 | 0.318 | 0.0024 | 0.0254 | +0.0189 | 1.00 |  | ![](frames/notable/lala.jpg) |
| cici | 30.0 | 0.380 | 0.359 | 0.0210 | 0.0313 | +0.0178 | 0.98 |  | ![](frames/notable/cici.jpg) |
| cece | 30.0 | 0.236 | 0.219 | 0.0166 | 0.0278 | -0.0173 | 0.99 |  | ![](frames/notable/cece.jpg) |
| natch | 30.0 | 0.337 | 0.318 | 0.0192 | 0.0162 | +0.0151 | 1.00 |  | ![](frames/notable/natch.jpg) |

## pronounced (1)

| name | fps | gt | M4 | |Δ|M4 | NLE mid |Δ| | tilt | conf | warnings | frame |
|------|-----|----|----|------|---------------|------|------|----------|-------|
| yoyo | 60.0 | 0.230 | 0.213 | 0.0176 | 0.0286 | -0.0318 | 1.00 |  | ![](frames/pronounced/yoyo.jpg) |
