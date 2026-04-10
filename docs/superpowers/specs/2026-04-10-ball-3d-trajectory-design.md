# Ball 3D Trajectory Estimation — Design Spec

## Context

The W3 stats-pack session shipped attack_direction, set_zones, and action_zone using
player-feet ground-plane projection. Four high-value stats remain blocked because the
ball is 1-4m above the court and a planar homography cannot project above-ground
objects: **serve speed** (#1 requested volleyball stat), **serve placement**, **attack
angle**, and **reception quality**.

Three prior approaches failed (direct homography, per-video scale factor, local
Jacobian) — all limited by the fundamental 2D ground-plane assumption. This session
prototypes a true 3D solution: recover a full pinhole camera model from court geometry,
then fit physics-constrained parabolic trajectories to 2D ball detections.

**Scope**: Research go/no-go. Validate camera model on 66 videos, prototype trajectory
fitter, evaluate physics sanity on all fitted serves. Ship-gate: physics metrics pass
thresholds defined below.

## Architecture

```
CourtCalibrator (existing)
    │  4 image↔court corner correspondences
    ▼
CameraModel (new, camera_model.py rewrite)
    │  K, R, t, P from cv2.solvePnP(IPPE)
    ▼
TrajectoryFitter3D (new, trajectory_3d.py)
    │  + ContactSequence (ball positions + contact frames)
    │  → Parabolic fit per free-flight arc
    ▼
List[FittedArc]
    │  (3D position, velocity, speed, residuals)
    ▼
eval_ball_3d.py (new)
    │  Physics sanity checks → go/no-go report
```

## 1. Camera Calibration — `camera_model.py` (independent rewrite)

### Problem

We have 4 court corners in normalized image coordinates (0-1) and known 3D positions
on the ground plane (z=0). Need: intrinsics K + extrinsics [R|t].

### Approach

Use `cv2.solvePnP` with `cv2.SOLVEPNP_IPPE` — designed for coplanar point
configurations, returns up to 2 solutions.

**Steps:**

1. Convert normalized image corners to pixel coordinates using video dimensions
   (from `BallTrackingResult.video_width/height`)
2. Define 3D court corner positions: `[(0,0,0), (8,0,0), (8,16,0), (0,16,0)]`
3. Estimate focal length prior: assume FOV ~50° → `f_px ≈ width / (2·tan(25°))`
4. Build initial K with estimated focal length, principal point at image center
5. Call `cv2.solvePnP(objectPoints, imagePoints, K, distCoeffs=None, flags=IPPE)`
6. For each solution: verify camera is above ground (height > 1m), court center is
   in front of camera, and reprojection error < 2px
7. Optionally refine K (focal length) via `cv2.solvePnP` with iterative flag or
   bundle adjustment against the 4 corners

### Data Structures

```python
@dataclass
class CameraModel:
    intrinsic_matrix: NDArray[np.float64]     # 3×3 K
    rotation: NDArray[np.float64]             # 3×3 R (world → camera)
    translation: NDArray[np.float64]          # 3×1 t
    projection_matrix: NDArray[np.float64]    # 3×4 P = K·[R|t]
    camera_position: NDArray[np.float64]      # 3×1 world position (-R^T·t)
    focal_length_px: float                    # focal length in pixels
    reprojection_error: float                 # mean corner error in pixels
    image_size: tuple[int, int]              # (width, height)
    is_valid: bool

def calibrate_camera(
    image_corners: list[tuple[float, float]],  # normalized 0-1
    court_corners: list[tuple[float, float]],  # meters, on z=0
    image_width: int,
    image_height: int,
) -> CameraModel | None: ...

def project_3d_to_image(
    camera: CameraModel,
    world_point: NDArray[np.float64],  # [X, Y, Z]
) -> tuple[float, float]: ...          # normalized image coords (0-1)

def image_ray(
    camera: CameraModel,
    image_point: tuple[float, float],  # normalized 0-1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (origin, direction) of the camera ray through the image point."""
```

### Validation

- Reprojection error on 4 corners: < 2px mean
- Camera height: 2-8m (tripod behind baseline or side)
- Court center (4, 8, 0) has positive depth
- Run on all 66 calibrated videos, report success rate

## 2. Trajectory Fitting — `trajectory_3d.py`

### Physics Model

Between contacts, the ball is in free flight under gravity (Phase 1, no drag):

```
X(t) = x0 + vx0·t
Y(t) = y0 + vy0·t
Z(t) = z0 + vz0·t - 0.5·g·t²
```

where g = 9.81 m/s², t is time in seconds relative to arc start, and (X, Y, Z) are
world coordinates in meters (X = court width axis, Y = court length axis, Z = up).

### Optimization

**Method**: `scipy.optimize.least_squares` with Levenberg-Marquardt.

**Parameters**: 6 unknowns — (x0, y0, z0, vx0, vy0, vz0)

**Residuals**: For each 2D observation (u_i, v_i) at time t_i:
```
[u_hat, v_hat, w] = P @ [X(t_i), Y(t_i), Z(t_i), 1]^T
residual = [(u_hat/w - u_i), (v_hat/w - v_i)]
```
Total: 2N residuals for N observations.

**Weighting**: Each observation weighted by `BallPosition.confidence`.

**Robustness**: Use `loss='soft_l1'` to handle outlier detections.

**Bounds**:
| Parameter | Lower | Upper | Unit |
|-----------|-------|-------|------|
| x0        | -5    | 13    | m    |
| y0        | -5    | 21    | m    |
| z0        | 0     | 5     | m    |
| vx0       | -30   | 30    | m/s  |
| vy0       | -30   | 30    | m/s  |
| vz0       | -15   | 15    | m/s  |

### Initialization Strategy

1. Project first and last 2D ball positions onto ground plane via existing homography
   → rough (x, y) start/end positions
2. Estimate horizontal velocity from displacement / time
3. Set z0 based on action type at the preceding contact:
   - serve: 3.0m, set: 2.5m, attack: 3.0m, dig: 0.5m, receive: 1.0m
4. Set vz0 to produce a reasonable peak height: `vz0 = sqrt(2·g·h_peak)` where
   h_peak = 2m above z0

### Arc Segmentation

Already available from contact detection:
- `ContactSequence.contacts` provides frame numbers of all contacts
- `ContactSequence.ball_positions` provides full ball trajectory
- Ball positions between consecutive contact frames form one arc
- **Minimum 5 frames per arc** — reject shorter arcs as low-confidence

### Data Structures

```python
@dataclass
class FittedArc:
    arc_index: int                              # 0-based within rally
    start_frame: int
    end_frame: int
    num_observations: int                       # frames with valid ball detection
    num_inliers: int                            # observations with reproj error < threshold

    # Fitted 3D trajectory parameters
    initial_position: NDArray[np.float64]       # [x0, y0, z0] meters
    initial_velocity: NDArray[np.float64]       # [vx0, vy0, vz0] m/s
    speed_at_start: float                       # |v0| in m/s

    # Derived quantities
    peak_height: float                          # max Z in meters
    net_crossing_height: float | None           # Z at Y=8m crossing, if applicable
    landing_position: tuple[float, float] | None  # (X, Y) where Z=0, if applicable

    # Quality metrics
    reprojection_rmse: float                    # pixels
    gravity_residual: float | None              # fitted g vs 9.81 (from free-g fit)
    is_valid: bool                              # passes all sanity checks

@dataclass
class TrajectoryResult:
    rally_id: str
    camera: CameraModel
    arcs: list[FittedArc]
    serve_speed_mps: float | None               # speed at first contact (serve)
    video_id: str
```

### Failure Modes and Diagnostics

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Short arc (< 5 frames) | Frame count check | Skip, flag as low-confidence |
| Degenerate camera | Reprojection error > 2px | Skip video |
| Solver divergence | least_squares status ≠ 1-4 | Skip arc |
| Outlier-dominated | >20% observations are outliers | Refit with RANSAC (3-point sample) |
| Depth-speed ambiguity | Condition number of Jacobian | Report uncertainty estimate |
| Missing contact (merged arcs) | Systematic U-shaped residuals | Split at max-residual point, refit |
| Ball out of frame | >50% frames with no detection | Flag arc, still fit on available data |

## 3. Evaluation — `eval_ball_3d.py`

### Pipeline

1. Load all rallies with court calibration from DB (via existing production_eval patterns)
2. For each video: run `calibrate_camera()`, check validity
3. For each rally: load `ContactSequence` + `ClassifiedAction` labels
4. For each free-flight arc between contacts: run trajectory fitting
5. Aggregate metrics per video and overall

### Ship-Gate Metrics

| Metric | Gate | Rationale |
|--------|------|-----------|
| Camera calibration success | ≥90% of 66 videos | Some cameras may be near-degenerate |
| Serve speed in [10, 35] m/s | ≥80% of fitted serves | Known amateur beach range (gravity-only biases high) |
| Net crossing height in [2.24, 5.0] m | ≥70% of crossing arcs | Must clear net, unlikely >5m |
| Landing Z ∈ [-0.5, 0.5] m | ≥70% of landing arcs | Ball near ground at landing |
| Gravity residual (free g within ±30% of 9.81) | ≥70% of arcs | Sanity on calibration + segmentation |
| Contact height in [0, 4] m | ≥80% of contacts | Physically reachable |
| Reprojection RMSE | Median < 5 px (at 1920×1080) | Fit must explain observed 2D |

### Report Format

Per-video summary:
```
[1/66] video_id=abc123: cam_height=4.2m, f=1850px, reproj=0.8px
  Arcs: 45 fitted, 42 valid (93%)
  Serve speeds: [18.2, 22.1, 19.5, ...] m/s, mean=19.9
  Net crossings: 38/40 in [2.24, 5.0]m, mean=2.8m
```

Aggregate:
```
=== GO/NO-GO SUMMARY ===
Camera success:      64/66 (97%)  [PASS ≥90%]
Serve speed valid:   182/210 (87%) [PASS ≥80%]
Net crossing valid:  285/350 (81%) [PASS ≥70%]
Landing Z valid:     190/240 (79%) [PASS ≥70%]
Gravity residual:    310/400 (78%) [PASS ≥70%]
Contact height:      520/600 (87%) [PASS ≥80%]
Reproj RMSE median:  3.2 px       [PASS <5px]

VERDICT: GO ✓ (all gates passed)
```

## 4. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `analysis/rallycut/court/camera_model.py` | **Rewrite** | solvePnP camera calibration |
| `analysis/rallycut/court/trajectory_3d.py` | **Create** | Parabolic trajectory fitting |
| `analysis/scripts/eval_ball_3d.py` | **Create** | Evaluation harness |
| `analysis/rallycut/court/__init__.py` | **Edit** | Export new modules |
| `analysis/tests/unit/test_camera_model.py` | **Create** | Unit tests for camera model |
| `analysis/tests/unit/test_trajectory_3d.py` | **Create** | Unit tests for trajectory fitter |

## 5. Existing Code to Reuse

- `CourtCalibrator` (`court/calibration.py`) — provides corner correspondences
- `ContactSequence` / `Contact` (`tracking/contact_detector.py`) — arc segmentation
- `ClassifiedAction` (`tracking/action_classifier.py`) — action type for z0 initialization
- `BallPosition` (`tracking/ball_tracker.py`) — 2D observations with confidence
- `BallTrackingResult` (`tracking/ball_tracker.py`) — video dimensions, fps
- Production eval DB loading patterns from `scripts/production_eval.py`

## 6. Out of Scope (Future, Post Go/No-Go)

- Air drag model (Phase 2 — adds ~10-15% serve speed accuracy)
- Wiring serve_speed into `match_stats.py` / `play_annotations.py`
- DB schema changes for 3D trajectory storage
- API/web exposure of 3D stats
- Apparent ball size as supplementary depth signal
- Serve placement / quality classification
- Attack angle computation

## 7. Verification Plan

1. **Unit tests**: Camera model with synthetic corners (known ground truth). Trajectory
   fitter with synthetic parabolic trajectory + known camera → verify recovery.
2. **Integration**: Run `eval_ball_3d.py` on all 66 calibrated videos. Check all
   ship-gate metrics pass.
3. **Spot check**: Manually inspect 10 serves — does the fitted 3D trajectory look
   plausible when projected back onto the video frame?
4. **Cross-video consistency**: Same player's serve speed should have lower variance
   within a match than across matches.
