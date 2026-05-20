# Net-top endpoints labeling convention (court keypoints 6 + 7)

Spec for human labelers (you, and any future contributor) who add the
two net-top endpoint keypoints to videos in the court-keypoint training
corpus.

## What to click

In each video's representative frame, click two points:

- **Keypoint 6 — net_top_left**: the point where the net top **tape**
  (the wide opaque band at the top of the net mesh) meets the **left
  sideline net post**.
- **Keypoint 7 — net_top_right**: the same point on the right post.

The midpoint of these two clicks is the line consumers will use
(replacing today's `estimate_net_line` solvePnP path). The tilt
between them is what enables tilt-aware net_y when the camera is
not perfectly leveled.

## Where exactly is "the net top tape"

The volleyball net has three visible horizontal features near its top:

1. **Antenna apex** — the small flag at the top of each antenna pole.
   This is **NOT** the net top. Antennas extend ~80 cm above the net.
2. **Top tape (white band)** — a 5-7 cm wide opaque white/branded
   band sewn over the top rope. **This is what we label.**
3. **Top of the mesh below the tape** — where the dark mesh visually
   begins below the tape. **NOT** the net top.

Click the **center of the tape's vertical extent** at the post,
not the top edge and not the bottom. If the tape's center is ambiguous
(e.g. the band is camera-blurred to ~10 pixels), pick the brightest
horizontal line within the band — that is closest to the physical
top rope under the tape.

## Visual examples

* **Good case** (juju, kaka, tete): clear sky background, sharp white
  tape, single click target. Click center-of-tape at each post.
* **Tilt case** (jiji, yoyo, wuwu): left and right click y-coords
  will visibly disagree. Click each independently — do not
  unconsciously average them.
* **Occlusion** (machi): if the left post is partly hidden by a
  player, **extrapolate the tape's height to where the post would
  be** (look at the tape's slope from the visible portion). If
  extrapolation is unreliable (tape obscured by >50% of its length),
  mark the keypoint visibility flag = 0 (not visible, skip this
  endpoint).
* **Wide-angle / yaya-class** (yaya 30fps): the net looks small in
  frame and the posts are far apart. Zoom in if the tool allows.
  The click target is still center-of-tape-at-post.
* **Mesh-with-no-tape edge case** (lala): some nets have a dark
  rope as the "top" and no white tape. Click the **dark rope
  itself**, not the top of the mesh below it. (lala in the SOTA
  validation pass was mislabeled because of this — both v8 N1 GT
  and my click confused the top-of-mesh with the top-rope.)

## What we do NOT label

* The antennas, antenna apices, or the antenna flag tips.
* The court boundary lines (already covered by keypoints 0-3).
* The net base / floor — that's at z=0; the net base is implicit
  in keypoints 4 + 5 (already computed from corners).

## Visibility flag

Each labeled keypoint has a visibility flag:
- `2` — clearly visible, click confident
- `1` — partially occluded or estimated by extrapolation
- `0` — not visible at all in the frame, skip (the trainer will
  treat this as missing data, not as a wrong label)

Default to `2` if the post is in frame and the tape is the dominant
feature at the click target. Use `1` for extrapolated clicks. Use
`0` only when the post is fully off-frame or covered such that the
tape's height cannot reasonably be inferred.

## Per-video labeling protocol

1. Open the calibration UI on the chosen frame (a mid-rally frame
   from a fixed-camera setup is ideal).
2. The 4 corners + 2 center keypoints are already in place from
   prior calibration work.
3. Two new orange handles appear at x≈8% and x≈92% (same idiom as
   the current shipping UI, but now they capture **independent**
   y-coordinates). Drag each independently to align with the visible
   net top tape at the matching post.
4. (If the visibility flag is set to anything other than 2, the UI
   shows a yellow caution indicator next to that handle.)
5. Click Save Calibration.

## Audit / spot-check

Before retraining, the dataset export script will render a sanity
overlay per video that shows the labeled endpoints + the connecting
line. Visually spot-check the 10-15 most-tilted videos to confirm
the labels look right.

If the labels look wrong on a video, re-label that one before training.

## Open question (decide before labeling starts)

**Which frame represents each video?** Options:
- A mid-rally frame (current N2 catalog convention).
- An empty-court frame from before play starts (cleaner view; no
  player occlusion).
- 3-5 frames per video, with the trainer doing the temporal aggregation.

Recommended: empty-court frame from `t = 2-3 seconds` if available;
otherwise mid-rally with the labeler instructed to skip ambiguous
clicks via visibility=0.
