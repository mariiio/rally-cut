"""WASB (Widely Applicable Strong Baseline) HRNet model for ball detection.

Ported from https://github.com/nttcom/WASB-SBDT (BMVC 2023).
Architecture: HRNet with multi-resolution branches for high-spatial-precision
ball detection. Pretrained on volleyball (40 training matches, 16 test matches).

Input: 3 consecutive RGB frames (9 channels) at 288x512, ImageNet-normalized.
Output: 3 heatmaps (one per frame) at 288x512, decoded to ball positions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rallycut.tracking.ball_filter import BallFilterConfig
    from rallycut.tracking.ball_smoother import TrajectorySmoothingConfig
    from rallycut.tracking.ball_tracker import BallTrackingResult

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# WASB standard dimensions (same as TrackNet)
IMG_HEIGHT = 288
IMG_WIDTH = 512
NUM_INPUT_FRAMES = 3

# ImageNet normalization (WASB uses RGB + ImageNet norm, not grayscale like VballNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BN_MOMENTUM = 0.1

# Weights directory
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights" / "wasb"

# WASB volleyball model config (from wasb.yaml)
WASB_CONFIG = {
    "name": "hrnet",
    "frames_in": 3,
    "frames_out": 3,
    "inp_height": 288,
    "inp_width": 512,
    "out_height": 288,
    "out_width": 512,
    "rgb_diff": False,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1],
                "NUM_CHANNELS": [32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [16, 32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {
                "NUM_DECONVS": 0,
                "KERNEL_SIZE": [],
                "NUM_BASIC_BLOCKS": 2,
            },
        },
        "INIT_WEIGHTS": True,
    },
}


# ---------------------------------------------------------------------------
# HRNet building blocks (from Microsoft HRNet, MIT License)
# ---------------------------------------------------------------------------


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


BLOCKS_DICT: dict[str, type[BasicBlock] | type[Bottleneck]] = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        blocks: type[BasicBlock] | type[Bottleneck],
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[BasicBlock] | type[Bottleneck],
        num_blocks: list[int],
        num_channels: list[int],
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers: list[nn.Module] = []
        layers.append(
            block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[BasicBlock] | type[Bottleneck],
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer: list[nn.Module | None] = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s: list[nn.Module] = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j], num_inchannels[i], 3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_inchannels[i]),
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_inchannels[j]),
                                    nn.ReLU(True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))  # type: ignore[arg-type]
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        return self.num_inchannels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):  # type: ignore[arg-type]
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])  # type: ignore[index]
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])  # type: ignore[index]
            x_fuse.append(self.relu(y))
        return x_fuse


# ---------------------------------------------------------------------------
# HRNet main model
# ---------------------------------------------------------------------------


class HRNet(nn.Module):
    """HRNet for ball detection (WASB architecture).

    Multi-resolution branches maintain spatial precision throughout the network,
    producing high-quality heatmaps at the input resolution.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = WASB_CONFIG

        self._frames_in = cfg["frames_in"]
        self._frames_out = cfg["frames_out"]
        self._out_scales = cfg["out_scales"]
        self._stem_strides = cfg["MODEL"]["EXTRA"]["STEM"]["STRIDES"]
        self._stem_inplanes = cfg["MODEL"]["EXTRA"]["STEM"]["INPLANES"]

        # Stem: 2 conv layers, 9 input channels (3 RGB frames)
        self.conv1 = nn.Conv2d(
            3 * self._frames_in,
            self._stem_inplanes,
            kernel_size=3,
            stride=self._stem_strides[0],
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            self._stem_inplanes,
            self._stem_inplanes,
            kernel_size=3,
            stride=self._stem_strides[1],
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.stage1_cfg = cfg["MODEL"]["EXTRA"]["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = BLOCKS_DICT[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, self._stem_inplanes, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # Stage 2
        self.stage2_cfg = cfg["MODEL"]["EXTRA"]["STAGE2"]
        num_channels_2 = self.stage2_cfg["NUM_CHANNELS"]
        block2 = BLOCKS_DICT[self.stage2_cfg["BLOCK"]]
        num_channels_2 = [num_channels_2[i] * block2.expansion for i in range(len(num_channels_2))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels_2)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels_2)

        # Stage 3
        self.stage3_cfg = cfg["MODEL"]["EXTRA"]["STAGE3"]
        num_channels_3 = self.stage3_cfg["NUM_CHANNELS"]
        block3 = BLOCKS_DICT[self.stage3_cfg["BLOCK"]]
        num_channels_3 = [num_channels_3[i] * block3.expansion for i in range(len(num_channels_3))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels_3)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels_3)

        # Stage 4
        self.stage4_cfg = cfg["MODEL"]["EXTRA"]["STAGE4"]
        num_channels_4 = self.stage4_cfg["NUM_CHANNELS"]
        block4 = BLOCKS_DICT[self.stage4_cfg["BLOCK"]]
        num_channels_4 = [num_channels_4[i] * block4.expansion for i in range(len(num_channels_4))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels_4)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels_4, multi_scale_output=True
        )

        # Final layers (1x1 conv to produce heatmaps)
        kernel_size = cfg["MODEL"]["EXTRA"]["FINAL_CONV_KERNEL"]
        self.final_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=pre_stage_channels[scale],
                    out_channels=self._frames_out,
                    kernel_size=kernel_size,
                )
                for scale in self._out_scales
            ]
        )

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers: list[nn.Module] = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(
        self, num_channels_pre: list[int], num_channels_cur: list[int]
    ) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur)
        num_branches_pre = len(num_channels_pre)
        transition_layers: list[nn.Module | None] = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s: list[nn.Module] = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre[-1]
                    outchannels = num_channels_cur[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)  # type: ignore[arg-type]

    def _make_stage(
        self,
        layer_config: dict,
        num_inchannels: list[int],
        multi_scale_output: bool = True,
    ) -> tuple[nn.Sequential, list[int]]:
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = BLOCKS_DICT[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules: list[nn.Module] = []
        for i in range(num_modules):
            reset_multi_scale = multi_scale_output if i == num_modules - 1 else True
            modules.append(
                HighResolutionModule(
                    num_branches, block, num_blocks, num_inchannels, num_channels,
                    fuse_method, reset_multi_scale,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()  # type: ignore[union-attr]
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # Stage 2
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # Stage 4
        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Final conv (per scale)
        y_out: dict[int, torch.Tensor] = {}
        for idx, scale in enumerate(self._out_scales):
            y_out[scale] = self.final_layers[idx](y_list[scale])
        return y_out


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_wasb_model(weights_path: Path | str | None = None, device: str = "cpu") -> HRNet:
    """Load WASB HRNet model with pretrained volleyball weights.

    Args:
        weights_path: Path to wasb_volleyball_best.pth.tar checkpoint.
            If None, looks in weights/wasb/ directory.
        device: Device to load model on.

    Returns:
        Loaded HRNet model in eval mode.
    """
    if weights_path is None:
        weights_path = WEIGHTS_DIR / "wasb_volleyball_best.pth.tar"

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"WASB weights not found at {weights_path}. "
            f"Download from: https://drive.google.com/file/d/1M9y4wPJqLc0K-z-Bo5DP8Ft5XwJuLqIS "
            f"and place in {WEIGHTS_DIR}/"
        )

    model = HRNet(WASB_CONFIG)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Handle DataParallel 'module.' prefix in saved weights
    state_dict = checkpoint["model_state_dict"]
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded WASB HRNet from {weights_path.name}")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Preprocess a single BGR frame for WASB inference.

    Resizes to 512x288, converts to RGB, normalizes with ImageNet stats.

    Args:
        frame_bgr: BGR frame from OpenCV (any resolution).

    Returns:
        Preprocessed frame as float32 array (3, H, W), normalized.
    """
    # Resize to model input size
    resized = cv2.resize(frame_bgr, (IMG_WIDTH, IMG_HEIGHT))
    # BGR -> RGB, uint8 -> float32 [0, 1]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # ImageNet normalization
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    return rgb.transpose(2, 0, 1)


def decode_heatmap_wasb(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Decode a WASB heatmap to (x_norm, y_norm, confidence).

    Uses connected components with weighted centroid (matching WASB postprocessor).

    Args:
        heatmap: Raw logits from model output (H, W). Sigmoid is applied here.
        threshold: Score threshold for blob detection.

    Returns:
        Tuple of (x_norm, y_norm, confidence). Returns (0.5, 0.5, 0.0) if no detection.
    """
    # Apply sigmoid to convert logits to probabilities
    prob = 1.0 / (1.0 + np.exp(-heatmap.astype(np.float64)))
    prob = prob.astype(np.float32)

    max_val = float(prob.max())
    if max_val <= threshold:
        return 0.5, 0.5, 0.0

    # Binary threshold
    binary = (prob > threshold).astype(np.uint8)

    # Connected components
    n_labels, labels = cv2.connectedComponents(binary)

    best_score = 0.0
    best_x = 0.5
    best_y = 0.5

    for m in range(1, n_labels):
        ys, xs = np.where(labels == m)
        ws = prob[ys, xs]
        # Weighted centroid (use_hm_weight=True in WASB)
        score = float(ws.sum())
        x = float(np.sum(xs.astype(np.float64) * ws) / np.sum(ws))
        y = float(np.sum(ys.astype(np.float64) * ws) / np.sum(ws))

        if score > best_score:
            best_score = score
            best_x = x / heatmap.shape[1]  # Normalize to 0-1
            best_y = y / heatmap.shape[0]

    # Clamp to valid range
    best_x = max(0.0, min(1.0, best_x))
    best_y = max(0.0, min(1.0, best_y))

    # Convert score to 0-1 confidence (heuristic: clamp at reasonable range)
    confidence = min(1.0, best_score / 5.0) if best_score > 0 else 0.0

    return best_x, best_y, confidence


# ---------------------------------------------------------------------------
# WASBBallTracker: pipeline-compatible tracker
# ---------------------------------------------------------------------------


class WASBBallTracker:
    """Ball tracker using WASB HRNet (PyTorch).

    Drop-in alternative to BallTracker with the same track_video() interface.
    Uses 3-frame sliding window with stride 1 for maximum coverage.
    For each frame, keeps the highest-confidence detection across overlapping windows.

    WASB achieves 67.5% match rate on beach volleyball GT (vs 41.7% for VballNet)
    with higher positional accuracy (51.9px vs 92.7px mean error).
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        device: str | None = None,
        threshold: float = 0.3,
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.threshold = threshold
        self._model: HRNet | None = None

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def _ensure_model(self) -> HRNet:
        if self._model is None:
            self._model = load_wasb_model(self.weights_path, device=self.device)
        return self._model

    def track_video(
        self,
        video_path: Path | str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
        filter_config: BallFilterConfig | None = None,
        enable_filtering: bool = True,
        preserve_raw: bool = False,
        enable_smoothing: bool = False,
        smoothing_config: TrajectorySmoothingConfig | None = None,
    ) -> BallTrackingResult:
        """Track ball positions using WASB HRNet.

        Same interface as BallTracker.track_video().
        """
        from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter
        from rallycut.tracking.ball_smoother import (
            TrajectoryPostProcessor,
            TrajectorySmoothingConfig,
        )
        from rallycut.tracking.ball_tracker import BallPosition, BallTrackingResult

        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        model = self._ensure_model()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            start_frame = 0
            end_frame = total_frames
            if start_ms is not None:
                start_frame = int(start_ms / 1000 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if end_ms is not None:
                end_frame = min(int(end_ms / 1000 * fps), total_frames)

            frames_to_process = end_frame - start_frame
            logger.info(
                f"WASB processing frames {start_frame}-{end_frame} "
                f"({frames_to_process} frames, {fps:.1f} fps, device={self.device})"
            )

            # Read all frames into memory (WASB uses 3-frame sliding window)
            raw_frames: list[np.ndarray] = []
            for _ in range(frames_to_process):
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)

            if len(raw_frames) < NUM_INPUT_FRAMES:
                processing_time_ms = (time.time() - start_time) * 1000
                return BallTrackingResult(
                    positions=[],
                    frame_count=len(raw_frames),
                    video_fps=fps,
                    video_width=video_width,
                    video_height=video_height,
                    processing_time_ms=processing_time_ms,
                    model_version="wasb_hrnet",
                )

            # Run sliding window inference
            frame_detections: dict[int, BallPosition] = {}
            inference_count = 0

            with torch.no_grad():
                for i in range(len(raw_frames) - NUM_INPUT_FRAMES + 1):
                    preprocessed = [
                        preprocess_frame(raw_frames[i + j]) for j in range(NUM_INPUT_FRAMES)
                    ]
                    triplet = np.concatenate(preprocessed, axis=0)  # (9, H, W)
                    x = torch.from_numpy(triplet).float().unsqueeze(0).to(self.device)

                    output = model(x)
                    heatmaps = output[0][0].cpu().numpy()  # (3, H, W)
                    inference_count += 1

                    for j in range(NUM_INPUT_FRAMES):
                        frame_idx = i + j
                        if frame_idx >= len(raw_frames):
                            break

                        x_norm, y_norm, conf = decode_heatmap_wasb(
                            heatmaps[j], threshold=self.threshold
                        )

                        if conf > 0.0:
                            bp = BallPosition(
                                frame_number=frame_idx,
                                x=x_norm,
                                y=y_norm,
                                confidence=conf,
                                motion_energy=0.0,
                            )
                            existing = frame_detections.get(frame_idx)
                            if existing is None or conf > existing.confidence:
                                frame_detections[frame_idx] = bp

                    if progress_callback and inference_count % 30 == 0:
                        progress = i / max(1, len(raw_frames) - NUM_INPUT_FRAMES)
                        progress_callback(min(0.99, progress))

            # Build complete position list (zeros for undetected frames)
            positions: list[BallPosition] = []
            for frame_idx in range(len(raw_frames)):
                if frame_idx in frame_detections:
                    positions.append(frame_detections[frame_idx])
                else:
                    positions.append(BallPosition(
                        frame_number=frame_idx,
                        x=0.5, y=0.5, confidence=0.0,
                    ))

            if progress_callback:
                progress_callback(1.0)

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"WASB completed {inference_count} inferences in "
                f"{processing_time_ms / 1000:.1f}s ({self.device})"
            )

            # Apply temporal filtering
            raw_positions = None
            if enable_filtering:
                if preserve_raw:
                    raw_positions = positions.copy()
                config = filter_config or BallFilterConfig()
                temporal_filter = BallTemporalFilter(config)
                positions = temporal_filter.filter_batch(positions)

            # Apply post-processing smoothing
            if enable_smoothing:
                smooth_config = smoothing_config or TrajectorySmoothingConfig()
                smoother = TrajectoryPostProcessor(smooth_config)
                positions = smoother.process(positions)

            return BallTrackingResult(
                positions=positions,
                frame_count=len(raw_frames),
                video_fps=fps,
                video_width=video_width,
                video_height=video_height,
                processing_time_ms=processing_time_ms,
                model_version="wasb_hrnet",
                filtering_enabled=enable_filtering,
                raw_positions=raw_positions,
            )

        finally:
            cap.release()
