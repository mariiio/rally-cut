"""BiLSTM vs GBM: joint action classification + player attribution.

Trains a bidirectional LSTM over rally contact sequences for joint
action type + player attribution prediction, and races it against
the existing per-contact GBM baselines under leave-one-video-out CV.

Key insight: GBMs treat each contact independently; the BiLSTM sees
the full rally sequence bidirectionally, potentially leveraging
serve→receive→set→attack patterns and alternating team touches.

Usage:
    cd analysis
    uv run python scripts/eval_bilstm_joint.py                     # Full LOO-video CV
    uv run python scripts/eval_bilstm_joint.py --learning-curve    # Scaling comparison
    uv run python scripts/eval_bilstm_joint.py --max-rallies 50    # Limit training data
    uv run python scripts/eval_bilstm_joint.py --skip-gbm          # LSTM only
    uv run python scripts/eval_bilstm_joint.py --hidden-dim 64 --num-layers 1
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from rallycut.evaluation.sanity_checks import SanityViolation, check_illegal_sequences
from rallycut.evaluation.split import add_split_argument, video_split
from rallycut.tracking.action_type_classifier import (
    extract_action_features,
)
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.temporal_attribution.features import (
    extract_attribution_features,
)
from scripts.eval_action_detection import (
    RallyData,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

ACTION_CLASSES = ["serve", "receive", "set", "attack", "dig"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}
NUM_ACTIONS = len(ACTION_CLASSES)
NUM_SLOTS = 4
ACTION_FEAT_DIM = 20
ATTR_FEAT_DIM = 36
TEAM_FEAT_DIM = 4
FEATURE_DIM = ACTION_FEAT_DIM + ATTR_FEAT_DIM + TEAM_FEAT_DIM  # 60


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContactSample:
    features: np.ndarray  # (FEATURE_DIM,)
    action_label: int  # 0-4
    attribution_label: int  # 0-3 canonical slot, -1 if not evaluable
    rally_id: str
    video_id: str


@dataclass
class RallySequence:
    contacts: list[ContactSample]
    rally_id: str
    video_id: str


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _parse_ball(raw: list[dict]) -> list[BallPos]:
    return [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in raw
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]


def _parse_players(raw: list[dict]) -> list[PlayerPos]:
    return [
        PlayerPos(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
        )
        for pp in raw
    ]


def extract_joint_features_for_rally(
    rally: RallyData,
    team_assignments: dict[int, int] | None,
    tolerance: int = 5,
) -> list[ContactSample]:
    """Extract joint feature vectors for all matched contacts in a rally."""
    if not rally.ball_positions_json:
        return []

    ball_positions = _parse_ball(rally.ball_positions_json)
    if not ball_positions:
        return []

    player_positions: list[PlayerPos] = []
    if rally.positions_json:
        player_positions = _parse_players(rally.positions_json)

    # Re-run contact detection (same as training pipeline)
    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=ContactDetectionConfig(),
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
    )

    if not contact_seq.contacts:
        return []

    # Match detected contacts to GT
    pred_actions = [
        {"frame": c.frame, "action": "unknown", "playerTrackId": c.player_track_id}
        for c in contact_seq.contacts
    ]
    matches, _ = match_contacts(rally.gt_labels, pred_actions, tolerance=tolerance)

    frame_to_idx: dict[int, int] = {}
    for idx, c in enumerate(contact_seq.contacts):
        frame_to_idx[c.frame] = idx

    samples: list[ContactSample] = []

    for m in matches:
        if m.pred_frame is None:
            continue
        if m.gt_action == "block":
            continue
        if m.gt_action not in ACTION_CLASSES:
            continue

        contact_idx = frame_to_idx.get(m.pred_frame)
        if contact_idx is None:
            continue

        contact = contact_seq.contacts[contact_idx]

        # --- Action features (20 dims) ---
        action_feat = extract_action_features(
            contact=contact,
            index=contact_idx,
            all_contacts=contact_seq.contacts,
            ball_positions=contact_seq.ball_positions or None,
            net_y=contact_seq.net_y,
            rally_start_frame=contact_seq.rally_start_frame,
            player_positions=player_positions or None,
        )
        action_arr = action_feat.to_array()

        # --- Attribution features (36 dims) + canonical slot IDs ---
        # Compute contact context for attribution
        side_count = action_feat.contact_count_on_current_side

        attr_result = extract_attribution_features(
            contact_frame=contact.frame,
            ball_positions=ball_positions,
            player_positions=player_positions,
            contact_index=contact_idx,
            side_count=side_count,
        )

        if attr_result is None:
            # Fall back: zero attribution features, mark attribution as not evaluable
            attr_arr = np.zeros(ATTR_FEAT_DIM, dtype=np.float64)
            canonical_tids: list[int] = []
        else:
            attr_arr, canonical_tids = attr_result

        # --- Team features (4 dims) ---
        team_arr = np.full(TEAM_FEAT_DIM, 0.5, dtype=np.float64)
        if team_assignments and canonical_tids:
            for s, tid in enumerate(canonical_tids[:4]):
                if tid in team_assignments:
                    team_arr[s] = float(team_assignments[tid])

        # --- Combine ---
        combined = np.concatenate([action_arr, attr_arr, team_arr])

        # --- Labels ---
        action_label = ACTION_TO_IDX[m.gt_action]

        # Attribution label: map GT track_id to canonical slot
        gt_tid = -1
        for gt in rally.gt_labels:
            if gt.frame == m.gt_frame:
                gt_tid = gt.player_track_id
                break

        attribution_label = -1  # not evaluable
        if gt_tid >= 0 and canonical_tids and gt_tid in canonical_tids:
            attribution_label = canonical_tids.index(gt_tid)

        samples.append(ContactSample(
            features=combined,
            action_label=action_label,
            attribution_label=attribution_label,
            rally_id=rally.rally_id,
            video_id=rally.video_id,
        ))

    return samples


def load_all_sequences() -> list[RallySequence]:
    """Load all rallies with GT and extract feature sequences."""
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies loaded")

    # Load team assignments
    video_ids = {r.video_id for r in rallies}
    rally_team_map = _load_match_team_assignments(video_ids)

    console.print(f"  Team assignments for {len(rally_team_map)} rallies")

    # Extract features per rally
    sequences: list[RallySequence] = []
    total_contacts = 0
    total_attr_evaluable = 0

    for i, rally in enumerate(rallies):
        ta = rally_team_map.get(rally.rally_id)
        samples = extract_joint_features_for_rally(rally, ta)

        if samples:
            sequences.append(RallySequence(
                contacts=samples,
                rally_id=rally.rally_id,
                video_id=rally.video_id,
            ))
            total_contacts += len(samples)
            total_attr_evaluable += sum(1 for s in samples if s.attribution_label >= 0)

        if (i + 1) % 20 == 0 or i == len(rallies) - 1:
            console.print(
                f"  [{i + 1}/{len(rallies)}] {len(sequences)} sequences, "
                f"{total_contacts} contacts"
            )

    console.print(f"\n  Total: {len(sequences)} sequences, {total_contacts} contacts, "
                  f"{total_attr_evaluable} attribution-evaluable")

    # Class distribution
    action_counts: dict[str, int] = defaultdict(int)
    for seq in sequences:
        for c in seq.contacts:
            action_counts[ACTION_CLASSES[c.action_label]] += 1
    console.print(f"  Actions: {dict(sorted(action_counts.items()))}")

    return sequences


# ---------------------------------------------------------------------------
# PyTorch Dataset and Model
# ---------------------------------------------------------------------------

class RallyDataset(Dataset):
    def __init__(self, sequences: list[RallySequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        feats = torch.tensor(
            np.array([c.features for c in seq.contacts]),
            dtype=torch.float32,
        )
        actions = torch.tensor(
            [c.action_label for c in seq.contacts],
            dtype=torch.long,
        )
        attrs = torch.tensor(
            [c.attribution_label for c in seq.contacts],
            dtype=torch.long,
        )
        attr_mask = torch.tensor(
            [c.attribution_label >= 0 for c in seq.contacts],
            dtype=torch.bool,
        )
        return feats, actions, attrs, attr_mask


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feats_list, actions_list, attrs_list, masks_list = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in feats_list], dtype=torch.long)

    feats_padded = pad_sequence(feats_list, batch_first=True)
    actions_padded = pad_sequence(actions_list, batch_first=True, padding_value=-1)
    attrs_padded = pad_sequence(attrs_list, batch_first=True, padding_value=-1)
    masks_padded = pad_sequence(masks_list, batch_first=True, padding_value=False)

    return feats_padded, actions_padded, attrs_padded, masks_padded, lengths


class BiLSTMJoint(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_actions: int = NUM_ACTIONS,
        num_slots: int = NUM_SLOTS,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_dim * 2, num_actions)
        self.attribution_head = nn.Linear(hidden_dim * 2, num_slots)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output)
        action_logits = self.action_head(output)
        attr_logits = self.attribution_head(output)
        return action_logits, attr_logits


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _normalize_features(
    train_seqs: list[RallySequence],
    test_seqs: list[RallySequence],
) -> tuple[list[RallySequence], list[RallySequence]]:
    """Z-score normalize features using training set statistics."""
    all_feats = np.array([
        c.features for seq in train_seqs for c in seq.contacts
    ])
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std[std < 1e-8] = 1.0

    def _norm_seq(seqs: list[RallySequence]) -> list[RallySequence]:
        result = []
        for seq in seqs:
            new_contacts = []
            for c in seq.contacts:
                new_contacts.append(ContactSample(
                    features=(c.features - mean) / std,
                    action_label=c.action_label,
                    attribution_label=c.attribution_label,
                    rally_id=c.rally_id,
                    video_id=c.video_id,
                ))
            result.append(RallySequence(
                contacts=new_contacts,
                rally_id=seq.rally_id,
                video_id=seq.video_id,
            ))
        return result

    return _norm_seq(train_seqs), _norm_seq(test_seqs)


def train_bilstm(
    train_seqs: list[RallySequence],
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.4,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    max_epochs: int = 300,
    patience: int = 30,
    seed: int = 42,
    attr_loss_weight: float = 0.5,
) -> BiLSTMJoint:
    """Train BiLSTM with early stopping on a held-out validation split."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split 85/15 for early stopping
    rng = np.random.RandomState(seed)
    n = len(train_seqs)
    indices = rng.permutation(n)
    val_size = max(1, int(0.15 * n))
    val_idx = set(indices[:val_size].tolist())

    val_seqs = [train_seqs[i] for i in range(n) if i in val_idx]
    trn_seqs = [train_seqs[i] for i in range(n) if i not in val_idx]

    if not trn_seqs or not val_seqs:
        trn_seqs = train_seqs
        val_seqs = train_seqs[:1]

    train_dataset = RallyDataset(trn_seqs)
    val_dataset = RallyDataset(val_seqs)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
    )

    model = BiLSTMJoint(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15,
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        for feats, actions, attrs, attr_mask, lengths in train_loader:
            optimizer.zero_grad()
            action_logits, attr_logits = model(feats, lengths)

            # Flatten for loss
            B, T, _ = action_logits.shape
            action_logits_flat = action_logits.reshape(B * T, -1)
            actions_flat = actions.reshape(B * T)
            action_loss = F.cross_entropy(
                action_logits_flat, actions_flat, ignore_index=-1,
            )

            attr_logits_flat = attr_logits.reshape(B * T, -1)
            attrs_flat = attrs.reshape(B * T)
            attr_mask_flat = attr_mask.reshape(B * T)

            if attr_mask_flat.any():
                attr_loss = F.cross_entropy(
                    attr_logits_flat[attr_mask_flat],
                    attrs_flat[attr_mask_flat],
                )
            else:
                attr_loss = torch.tensor(0.0)

            loss = action_loss + attr_loss_weight * attr_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for feats, actions, attrs, attr_mask, lengths in val_loader:
                action_logits, attr_logits = model(feats, lengths)
                B, T, _ = action_logits.shape
                al = F.cross_entropy(
                    action_logits.reshape(B * T, -1),
                    actions.reshape(B * T),
                    ignore_index=-1,
                )
                af = attr_mask.reshape(B * T)
                if af.any():
                    atl = F.cross_entropy(
                        attr_logits.reshape(B * T, -1)[af],
                        attrs.reshape(B * T)[af],
                    )
                else:
                    atl = torch.tensor(0.0)
                val_losses.append((al + attr_loss_weight * atl).item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_bilstm(
    model: BiLSTMJoint,
    test_seqs: list[RallySequence],
) -> tuple[float, float, int, int]:
    """Evaluate BiLSTM on test sequences.

    Returns (action_accuracy, attribution_accuracy, n_contacts, n_attr_evaluable).
    """
    model.eval()
    action_correct = 0
    action_total = 0
    attr_correct = 0
    attr_total = 0

    with torch.no_grad():
        for seq in test_seqs:
            ds = RallyDataset([seq])
            feats, actions, attrs, attr_mask, lengths = collate_fn([ds[0]])
            action_logits, attr_logits = model(feats, lengths)

            action_preds = action_logits[0, :lengths[0]].argmax(dim=-1)
            action_labels = actions[0, :lengths[0]]
            action_correct += (action_preds == action_labels).sum().item()
            action_total += lengths[0].item()

            attr_preds = attr_logits[0, :lengths[0]].argmax(dim=-1)
            attr_labels = attrs[0, :lengths[0]]
            mask = attr_mask[0, :lengths[0]]
            if mask.any():
                attr_correct += (attr_preds[mask] == attr_labels[mask]).sum().item()
                attr_total += mask.sum().item()

    action_acc = action_correct / max(1, action_total)
    attr_acc = attr_correct / max(1, attr_total)
    return action_acc, attr_acc, action_total, attr_total


def predict_bilstm_actions(
    model: BiLSTMJoint,
    test_seqs: list[RallySequence],
) -> dict[str, list[str]]:
    """Return per-rally predicted action names for sanity checking."""
    model.eval()
    predictions: dict[str, list[str]] = {}
    with torch.no_grad():
        for seq in test_seqs:
            ds = RallyDataset([seq])
            feats, actions, attrs, attr_mask, lengths = collate_fn([ds[0]])
            action_logits, _ = model(feats, lengths)
            preds = action_logits[0, :lengths[0]].argmax(dim=-1).cpu().tolist()
            predictions[seq.rally_id] = [ACTION_CLASSES[p] for p in preds]
    return predictions


# ---------------------------------------------------------------------------
# GBM baseline
# ---------------------------------------------------------------------------

def run_gbm_fold(
    train_seqs: list[RallySequence],
    test_seqs: list[RallySequence],
) -> tuple[float, float, int, int]:
    """Train and evaluate GBM baselines for one fold.

    Returns (action_accuracy, attribution_accuracy, n_contacts, n_attr_evaluable).
    """
    # Collect features and labels
    train_action_X, train_action_y = [], []
    train_attr_X, train_attr_y = [], []
    for seq in train_seqs:
        for c in seq.contacts:
            train_action_X.append(c.features[:ACTION_FEAT_DIM])
            train_action_y.append(c.action_label)
            if c.attribution_label >= 0:
                train_attr_X.append(c.features[ACTION_FEAT_DIM:ACTION_FEAT_DIM + ATTR_FEAT_DIM])
                train_attr_y.append(c.attribution_label)

    test_action_X, test_action_y = [], []
    test_attr_X, test_attr_y = [], []
    for seq in test_seqs:
        for c in seq.contacts:
            test_action_X.append(c.features[:ACTION_FEAT_DIM])
            test_action_y.append(c.action_label)
            if c.attribution_label >= 0:
                test_attr_X.append(c.features[ACTION_FEAT_DIM:ACTION_FEAT_DIM + ATTR_FEAT_DIM])
                test_attr_y.append(c.attribution_label)

    # Action GBM
    action_acc = 0.0
    n_contacts = len(test_action_y)
    if train_action_X and test_action_X:
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=3, subsample=0.8, random_state=42,
        )
        clf.fit(np.array(train_action_X), np.array(train_action_y))
        preds = clf.predict(np.array(test_action_X))
        action_acc = float(np.mean(preds == np.array(test_action_y)))

    # Attribution HistGBM
    attr_acc = 0.0
    n_attr = len(test_attr_y)
    if train_attr_X and test_attr_X and len(set(train_attr_y)) > 1:
        clf2 = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, max_leaf_nodes=15, random_state=42,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
        )
        clf2.fit(np.array(train_attr_X), np.array(train_attr_y))
        preds2 = clf2.predict(np.array(test_attr_X))
        attr_acc = float(np.mean(preds2 == np.array(test_attr_y)))

    return action_acc, attr_acc, n_contacts, n_attr


# ---------------------------------------------------------------------------
# LOO-video CV
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    video_id: str
    n_contacts: int
    n_attr: int
    lstm_action_acc: float
    lstm_attr_acc: float
    gbm_action_acc: float
    gbm_attr_acc: float


def run_loocv(
    all_sequences: list[RallySequence],
    max_rallies: int | None = None,
    skip_gbm: bool = False,
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.4,
    lr: float = 1e-3,
    max_epochs: int = 300,
    patience: int = 30,
    seed: int = 42,
) -> list[FoldResult]:
    """Run leave-one-video-out CV for both LSTM and GBM."""
    unique_videos = sorted(set(seq.video_id for seq in all_sequences))
    console.print(f"\nLOO-video CV: {len(unique_videos)} folds, "
                  f"max_rallies={max_rallies or 'all'}")

    results: list[FoldResult] = []

    for fold_idx, held_out in enumerate(unique_videos):
        train_seqs = [s for s in all_sequences if s.video_id != held_out]
        test_seqs = [s for s in all_sequences if s.video_id == held_out]

        if not test_seqs or not train_seqs:
            continue

        # Subsample training if max_rallies specified
        if max_rallies and len(train_seqs) > max_rallies:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(train_seqs), max_rallies, replace=False)
            train_seqs = [train_seqs[i] for i in sorted(idx)]

        # Normalize
        train_norm, test_norm = _normalize_features(train_seqs, test_seqs)

        # LSTM
        model = train_bilstm(
            train_norm,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )
        lstm_action, lstm_attr, n_contacts, n_attr = evaluate_bilstm(model, test_norm)

        # GBM (uses un-normalized features)
        if skip_gbm:
            gbm_action, gbm_attr = 0.0, 0.0
            gbm_n, gbm_na = 0, 0
        else:
            gbm_action, gbm_attr, gbm_n, gbm_na = run_gbm_fold(
                train_seqs, test_seqs,
            )

        results.append(FoldResult(
            video_id=held_out,
            n_contacts=n_contacts,
            n_attr=n_attr,
            lstm_action_acc=lstm_action,
            lstm_attr_acc=lstm_attr,
            gbm_action_acc=gbm_action,
            gbm_attr_acc=gbm_attr,
        ))

        console.print(
            f"  [{fold_idx + 1}/{len(unique_videos)}] {held_out[:8]}: "
            f"n={n_contacts} | "
            f"LSTM act={lstm_action:.0%} attr={lstm_attr:.0%} | "
            f"GBM act={gbm_action:.0%} attr={gbm_attr:.0%}"
        )

    return results


def print_results(results: list[FoldResult], skip_gbm: bool = False) -> None:
    """Print aggregate results table."""
    table = Table(title="LOO-Video CV Results")
    table.add_column("Video", style="cyan", max_width=10)
    table.add_column("N", justify="right")
    table.add_column("LSTM Act", justify="right")
    table.add_column("LSTM Attr", justify="right")
    if not skip_gbm:
        table.add_column("GBM Act", justify="right")
        table.add_column("GBM Attr", justify="right")

    for r in results:
        row = [
            r.video_id[:8],
            str(r.n_contacts),
            f"{r.lstm_action_acc:.0%}",
            f"{r.lstm_attr_acc:.0%}" if r.n_attr > 0 else "-",
        ]
        if not skip_gbm:
            row.extend([
                f"{r.gbm_action_acc:.0%}",
                f"{r.gbm_attr_acc:.0%}" if r.n_attr > 0 else "-",
            ])
        table.add_row(*row)

    # Weighted aggregates
    total_n = sum(r.n_contacts for r in results)
    total_na = sum(r.n_attr for r in results)

    lstm_action_agg = sum(r.lstm_action_acc * r.n_contacts for r in results) / max(1, total_n)
    lstm_attr_agg = sum(
        r.lstm_attr_acc * r.n_attr for r in results if r.n_attr > 0
    ) / max(1, total_na)

    agg_row = ["TOTAL", str(total_n), f"{lstm_action_agg:.1%}", f"{lstm_attr_agg:.1%}"]
    if not skip_gbm:
        gbm_action_agg = sum(r.gbm_action_acc * r.n_contacts for r in results) / max(1, total_n)
        gbm_attr_agg = sum(
            r.gbm_attr_acc * r.n_attr for r in results if r.n_attr > 0
        ) / max(1, total_na)
        agg_row.extend([f"{gbm_action_agg:.1%}", f"{gbm_attr_agg:.1%}"])

    table.add_row(*agg_row, style="bold")
    console.print(table)


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

def run_learning_curve(
    all_sequences: list[RallySequence],
    sizes: list[int],
    output_dir: str = "outputs",
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.4,
    lr: float = 1e-3,
    max_epochs: int = 300,
    patience: int = 30,
    seed: int = 42,
) -> None:
    """Run LOO-video CV at multiple training set sizes and plot."""
    curve_data: dict[str, list[tuple[int, float]]] = {
        "LSTM Action": [],
        "LSTM Attr": [],
        "GBM Action": [],
        "GBM Attr": [],
    }

    for size in sizes:
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold]Learning curve: max_rallies = {size}[/bold]")
        console.print(f"[bold]{'='*60}[/bold]")

        results = run_loocv(
            all_sequences,
            max_rallies=size if size < len(all_sequences) else None,
            skip_gbm=False,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )

        print_results(results)

        total_n = sum(r.n_contacts for r in results)
        total_na = sum(r.n_attr for r in results)

        lstm_act = sum(r.lstm_action_acc * r.n_contacts for r in results) / max(1, total_n)
        lstm_attr = sum(
            r.lstm_attr_acc * r.n_attr for r in results if r.n_attr > 0
        ) / max(1, total_na)
        gbm_act = sum(r.gbm_action_acc * r.n_contacts for r in results) / max(1, total_n)
        gbm_attr = sum(
            r.gbm_attr_acc * r.n_attr for r in results if r.n_attr > 0
        ) / max(1, total_na)

        curve_data["LSTM Action"].append((size, lstm_act))
        curve_data["LSTM Attr"].append((size, lstm_attr))
        curve_data["GBM Action"].append((size, gbm_act))
        curve_data["GBM Attr"].append((size, gbm_attr))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Action accuracy
        for label, style in [("LSTM Action", "-o"), ("GBM Action", "--s")]:
            xs, ys = zip(*curve_data[label])
            ax1.plot(xs, [y * 100 for y in ys], style, label=label, markersize=8)
        ax1.set_xlabel("Training Rallies")
        ax1.set_ylabel("Action Accuracy (%)")
        ax1.set_title("Action Classification")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Attribution accuracy
        for label, style in [("LSTM Attr", "-o"), ("GBM Attr", "--s")]:
            xs, ys = zip(*curve_data[label])
            ax2.plot(xs, [y * 100 for y in ys], style, label=label, markersize=8)
        ax2.set_xlabel("Training Rallies")
        ax2.set_ylabel("Attribution Accuracy (%)")
        ax2.set_title("Player Attribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("BiLSTM vs GBM: Learning Curve (LOO-Video CV)", fontsize=14)
        fig.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "bilstm_learning_curve.png")
        fig.savefig(plot_path, dpi=150)
        console.print(f"\n[green]Saved learning curve to {plot_path}[/green]")
        plt.close(fig)
    except ImportError:
        console.print("[yellow]matplotlib not available, skipping plot[/yellow]")

    # Summary table
    summary = Table(title="Learning Curve Summary")
    summary.add_column("Rallies", justify="right")
    summary.add_column("LSTM Act", justify="right")
    summary.add_column("GBM Act", justify="right")
    summary.add_column("LSTM Attr", justify="right")
    summary.add_column("GBM Attr", justify="right")

    for i, size in enumerate(sizes):
        summary.add_row(
            str(size),
            f"{curve_data['LSTM Action'][i][1]:.1%}",
            f"{curve_data['GBM Action'][i][1]:.1%}",
            f"{curve_data['LSTM Attr'][i][1]:.1%}",
            f"{curve_data['GBM Attr'][i][1]:.1%}",
        )
    console.print(summary)


# ---------------------------------------------------------------------------
# Single train/held-out split
# ---------------------------------------------------------------------------

def run_single_split(
    all_sequences: list[RallySequence],
    skip_gbm: bool = False,
    hidden_dim: int = 96,
    num_layers: int = 2,
    dropout: float = 0.4,
    lr: float = 1e-3,
    max_epochs: int = 300,
    patience: int = 30,
    seed: int = 42,
) -> None:
    """Train on 'train' split, evaluate on 'held_out' split."""
    train_seqs = [s for s in all_sequences if video_split(s.video_id) == "train"]
    test_seqs = [s for s in all_sequences if video_split(s.video_id) == "held_out"]

    train_vids = sorted(set(s.video_id for s in train_seqs))
    test_vids = sorted(set(s.video_id for s in test_seqs))
    console.print(
        f"\nSingle split: {len(train_vids)} train videos ({len(train_seqs)} rallies), "
        f"{len(test_vids)} held-out videos ({len(test_seqs)} rallies)"
    )

    if not test_seqs:
        console.print("[red]No held-out sequences. Check split.[/red]")
        return
    if not train_seqs:
        console.print("[red]No training sequences. Check split.[/red]")
        return

    train_norm, test_norm = _normalize_features(train_seqs, test_seqs)

    # --- LSTM ---
    console.print("\n[bold]Training BiLSTM...[/bold]")
    model = train_bilstm(
        train_norm,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
    )
    lstm_action, lstm_attr, n_contacts, n_attr = evaluate_bilstm(model, test_norm)
    lstm_preds = predict_bilstm_actions(model, test_norm)

    # --- GBM ---
    if skip_gbm:
        gbm_action, gbm_attr = 0.0, 0.0
    else:
        console.print("[bold]Training GBM baselines...[/bold]")
        gbm_action, gbm_attr, _, _ = run_gbm_fold(train_seqs, test_seqs)

    # --- Results table ---
    table = Table(title="Train/Held-Out Split Results")
    table.add_column("Model", style="bold")
    table.add_column("Action Acc", justify="right")
    table.add_column("Attr Acc", justify="right")
    table.add_column("Contacts", justify="right")

    table.add_row("BiLSTM", f"{lstm_action:.1%}", f"{lstm_attr:.1%}", str(n_contacts))
    if not skip_gbm:
        table.add_row("GBM", f"{gbm_action:.1%}", f"{gbm_attr:.1%}", str(n_contacts))
    console.print(table)

    # --- Sanity checks on LSTM predictions ---
    all_violations: list[SanityViolation] = []
    for seq in test_seqs:
        preds = lstm_preds.get(seq.rally_id, [])
        if preds:
            all_violations.extend(
                check_illegal_sequences(preds, rally_id=seq.rally_id)
            )

    console.print(f"\n[bold]LSTM Sanity: {len(all_violations)} illegal sequences[/bold]")
    if all_violations and len(all_violations) <= 20:
        for v in all_violations:
            console.print(f"  [{v.rally_id[:8]}] {v.description}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BiLSTM vs GBM: joint action + attribution experiment"
    )
    parser.add_argument("--max-rallies", type=int, default=None,
                        help="Limit training rallies per fold")
    parser.add_argument("--learning-curve", action="store_true",
                        help="Run at 50/100/all and plot scaling comparison")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gbm", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    add_split_argument(parser)
    args = parser.parse_args()

    all_sequences = load_all_sequences()
    if not all_sequences:
        console.print("[red]No sequences extracted. Check DB connection.[/red]")
        return

    n_rallies = len(all_sequences)
    console.print(f"\n[bold]{n_rallies} rally sequences ready[/bold]")

    # Single train/held-out split mode
    if args.split != "all":
        run_single_split(
            all_sequences,
            skip_gbm=args.skip_gbm,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        return

    if args.learning_curve:
        sizes = [s for s in [50, 100, n_rallies] if s <= n_rallies]
        if sizes[-1] != n_rallies:
            sizes.append(n_rallies)
        run_learning_curve(
            all_sequences,
            sizes=sizes,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
    else:
        results = run_loocv(
            all_sequences,
            max_rallies=args.max_rallies,
            skip_gbm=args.skip_gbm,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        print_results(results, skip_gbm=args.skip_gbm)


if __name__ == "__main__":
    main()
