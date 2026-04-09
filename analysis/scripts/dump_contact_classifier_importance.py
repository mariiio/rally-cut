"""Dump feature importances of the production contact classifier GBM."""
from rallycut.tracking.contact_classifier import load_contact_classifier

clf = load_contact_classifier()
names = list(clf._feature_names)
assert len(names) == 32, f"expected 32 features, got {len(names)}"

importances = clf.model.feature_importances_
ranked = sorted(zip(names, importances), key=lambda x: -x[1])

SEQ = {f"seq_p_{c}" for c in ("background", "serve", "receive", "set", "attack", "dig", "block")}
POSE = {
    "nearest_active_wrist_velocity_max",
    "nearest_hand_ball_dist_min",
    "nearest_active_arm_extension_change",
    "nearest_pose_confidence_mean",
    "nearest_both_arms_raised",
}

print(f"{'rank':>4}  {'feature':45}  {'imp':>8}  {'cum':>8}  group")
cum = 0.0
for i, (n, imp) in enumerate(ranked, 1):
    cum += imp
    group = "SEQ" if n in SEQ else "POSE" if n in POSE else "base"
    flag = "  <-- seq" if group == "SEQ" else ""
    print(f"{i:>4}  {n:45}  {imp:>8.4f}  {cum:>8.4f}  {group}{flag}")

sum_seq = sum(i for n, i in zip(names, importances) if n in SEQ)
sum_pose = sum(i for n, i in zip(names, importances) if n in POSE)
sum_base = sum(importances) - sum_seq - sum_pose
print()
print(f"sum(base, 20):      {sum_base:.4f}")
print(f"sum(seq_p_*, 7):    {sum_seq:.4f}")
print(f"sum(pose, 5):       {sum_pose:.4f}")
print(f"total:              {sum_base + sum_seq + sum_pose:.4f}")
