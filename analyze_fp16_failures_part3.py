"""
Part 3: Final deep dive on the failure mechanism.
Key questions:
- Is margin/logit_diff_ratio the right discriminator?
- What exactly happens to the logit vector near the argmax for failures?
- Can we find a clean threshold that catches 100% with minimal false positives?
"""

import numpy as np
import cv2
import coremltools as ct
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

SIMCC_SPLIT = 2.0
DISAGREE_THRESHOLD_PX = 5.0

def preprocess(img, bbox):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    s = max(bw/288, bh/384) * 1.25
    x1c = max(0, int(cx - 288*s/2))
    y1c = max(0, int(cy - 384*s/2))
    x2c = min(w, int(cx + 288*s/2))
    y2c = min(h, int(cy + 384*s/2))
    crop = img[y1c:y2c, x1c:x2c]
    resized = cv2.resize(crop, (288, 384))
    blob = resized.astype(np.float32)
    blob[:,:,0] = (blob[:,:,0] - 123.675) / 58.395
    blob[:,:,1] = (blob[:,:,1] - 116.28) / 57.12
    blob[:,:,2] = (blob[:,:,2] - 103.53) / 57.375
    return np.transpose(blob, (2,0,1))[np.newaxis].astype(np.float32)


print("Loading models...")
fp32_model = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
fp16_model = ct.models.MLModel("rtmpose_fp16.mlpackage", compute_units=ct.ComputeUnit.ALL)
yolo = YOLO("yolo11s-seg.pt")

OUT_X = "var_2124"  # (1,133,576)
OUT_Y = "var_2125"  # (1,133,768)

image_paths = sorted(Path("test_images").glob("*.jpg"))

all_kpts = []

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    r = yolo.predict(img, classes=[0], verbose=False)[0]
    if len(r.boxes) == 0:
        continue
    idx = r.boxes.conf.argmax().item()
    bbox = r.boxes.xyxy[idx].cpu().numpy()
    blob = preprocess(img, bbox)

    fp32_outs = fp32_model.predict({"input_1": blob})
    fp16_outs = fp16_model.predict({"input_1": blob})

    fp32_x = fp32_outs[OUT_X][0]  # (133, 576)
    fp32_y = fp32_outs[OUT_Y][0]  # (133, 768)
    fp16_x = fp16_outs[OUT_X][0]
    fp16_y = fp16_outs[OUT_Y][0]

    for kpt_idx in range(133):
        fp32_ax = int(np.argmax(fp32_x[kpt_idx]))
        fp32_ay = int(np.argmax(fp32_y[kpt_idx]))
        fp16_ax = int(np.argmax(fp16_x[kpt_idx]))
        fp16_ay = int(np.argmax(fp16_y[kpt_idx]))

        dx = abs(fp32_ax - fp16_ax) / SIMCC_SPLIT
        dy = abs(fp32_ay - fp16_ay) / SIMCC_SPLIT
        px_dist = np.sqrt(dx**2 + dy**2)

        disagree = px_dist > DISAGREE_THRESHOLD_PX
        x_disagree = dx > DISAGREE_THRESHOLD_PX
        y_disagree = dy > DISAGREE_THRESHOLD_PX

        # Per-axis stats
        for axis_name, fp32_logits, fp16_logits, ax_fp32, ax_fp16, ax_disagree in [
            ("x", fp32_x[kpt_idx], fp16_x[kpt_idx], fp32_ax, fp16_ax, x_disagree),
            ("y", fp32_y[kpt_idx], fp16_y[kpt_idx], fp32_ay, fp16_ay, y_disagree),
        ]:
            sorted_fp32 = np.sort(fp32_logits)[::-1]
            margin = sorted_fp32[0] - sorted_fp32[1]
            max_logit = sorted_fp32[0]
            max_diff = float(np.max(np.abs(fp32_logits - fp16_logits)))

            # The key ratio: margin vs max_diff
            # If margin < max_diff, the fp16 perturbation could flip the argmax
            ratio = margin / max_diff if max_diff > 0 else float('inf')

            # Also compute: logit value at fp32 argmax in fp16
            fp16_val_at_fp32_argmax = float(fp16_logits[ax_fp32])
            fp16_max_val = float(np.max(fp16_logits))
            fp16_margin_at_fp32_pos = fp16_max_val - fp16_val_at_fp32_argmax

            all_kpts.append({
                "image": img_path.stem,
                "kpt_idx": kpt_idx,
                "axis": axis_name,
                "disagree": disagree,
                "axis_disagree": ax_disagree,
                "ax_fp32": ax_fp32,
                "ax_fp16": ax_fp16,
                "ax_diff": abs(ax_fp32 - ax_fp16),
                "margin": margin,
                "max_logit": max_logit,
                "max_diff": max_diff,
                "margin_to_diff_ratio": ratio,
                "fp16_val_at_fp32_argmax": fp16_val_at_fp32_argmax,
                "fp16_max_val": fp16_max_val,
                "fp16_margin_at_fp32_pos": fp16_margin_at_fp32_pos,
            })

# Separate per-axis records
agree_axis = [k for k in all_kpts if not k["axis_disagree"]]
disagree_axis = [k for k in all_kpts if k["axis_disagree"]]

print(f"\nPer-axis analysis: {len(all_kpts)} total axis-records")
print(f"  Axis-agree: {len(agree_axis)}, Axis-disagree: {len(disagree_axis)}")

# ============================================================
# 1. Margin / max_diff ratio as discriminator
# ============================================================
print("\n" + "=" * 80)
print("1. MARGIN-TO-DIFF RATIO")
print("   If margin < max_diff (ratio < 1.0), fp16 perturbation could flip argmax")
print("=" * 80)

agree_ratios = np.array([k["margin_to_diff_ratio"] for k in agree_axis])
disagree_ratios = np.array([k["margin_to_diff_ratio"] for k in disagree_axis])

# Cap ratios at 100 for display
agree_ratios_capped = np.clip(agree_ratios, 0, 100)
disagree_ratios_capped = np.clip(disagree_ratios, 0, 100)

print(f"\n  Margin/max_diff ratio:")
print(f"    AGREE:    mean={agree_ratios_capped.mean():.3f}  median={np.median(agree_ratios_capped):.3f}  min={agree_ratios.min():.6f}")
print(f"    DISAGREE: mean={disagree_ratios_capped.mean():.3f}  median={np.median(disagree_ratios_capped):.3f}  max={disagree_ratios.max():.6f}")

# What fraction of axis-disagree have ratio < 1.0?
below_1 = sum(1 for k in disagree_axis if k["margin_to_diff_ratio"] < 1.0)
print(f"\n  Disagree with ratio < 1.0: {below_1}/{len(disagree_axis)} ({100*below_1/len(disagree_axis):.1f}%)")
below_1_agree = sum(1 for k in agree_axis if k["margin_to_diff_ratio"] < 1.0)
print(f"  Agree with ratio < 1.0:    {below_1_agree}/{len(agree_axis)} ({100*below_1_agree/len(agree_axis):.1f}%)")

# Threshold sweep on ratio
print(f"\n  Filter: ratio < threshold (per axis)")
print(f"    {'Threshold':>10} {'Caught':>10} {'FP':>10} {'Precision':>10} {'Recall':>8}")
for thresh in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    caught = sum(1 for k in disagree_axis if k["margin_to_diff_ratio"] < thresh)
    fp = sum(1 for k in agree_axis if k["margin_to_diff_ratio"] < thresh)
    total = caught + fp
    prec = caught / total if total > 0 else 0
    recall = caught / len(disagree_axis) if disagree_axis else 0
    print(f"    {thresh:>10.1f} {caught:>5}/{len(disagree_axis):<5} {fp:>10} {prec:>10.1%} {recall:>8.1%}")


# ============================================================
# 2. Look at the outlier Y-axis failures with high margin
# ============================================================
print("\n" + "=" * 80)
print("2. OUTLIER FAILURES: Y-axis disagree with margin >= 0.05")
print("=" * 80)

outliers = [k for k in disagree_axis if k["axis"] == "y" and k["margin"] >= 0.05]
print(f"\n  Found {len(outliers)} outlier(s):")
for k in outliers:
    print(f"    {k['image']} kpt={k['kpt_idx']}: margin={k['margin']:.4f}, max_diff={k['max_diff']:.4f}, "
          f"ratio={k['margin_to_diff_ratio']:.3f}")
    print(f"      fp32_argmax={k['ax_fp32']}, fp16_argmax={k['ax_fp16']}, diff={k['ax_diff']}")
    print(f"      fp16_val_at_fp32_pos={k['fp16_val_at_fp32_argmax']:.4f}, fp16_max={k['fp16_max_val']:.4f}, "
          f"fp16_gap={k['fp16_margin_at_fp32_pos']:.4f}")


# ============================================================
# 3. Summary: what is the best single filter?
# ============================================================
print("\n" + "=" * 80)
print("3. BEST SINGLE-METRIC FILTER COMPARISON")
print("   For each metric, show the threshold that catches 100% of failures")
print("   and the resulting false positive rate")
print("=" * 80)

# margin_to_diff_ratio approach (per-axis)
# We need to convert back to per-keypoint: a keypoint fails if EITHER axis is flagged
# First, group by (image, kpt_idx)
kpt_data = defaultdict(dict)
for k in all_kpts:
    key = (k["image"], k["kpt_idx"])
    kpt_data[key][k["axis"]] = k

disagree_kpts = set()
for k in all_kpts:
    if k["disagree"]:
        disagree_kpts.add((k["image"], k["kpt_idx"]))

total_kpts = len(kpt_data)
n_disagree_kpts = len(disagree_kpts)
agree_kpts = set(kpt_data.keys()) - disagree_kpts

print(f"\n  Total keypoints: {total_kpts}, Disagree: {n_disagree_kpts}, Agree: {total_kpts - n_disagree_kpts}")

# Per-keypoint filter: flag if EITHER axis has ratio < threshold
print(f"\n  A) Margin/diff ratio (flag if either axis ratio < threshold):")
print(f"    {'Threshold':>10} {'Caught':>10} {'FP':>10} {'Precision':>10} {'Recall':>8} {'% total flagged':>16}")
for thresh in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
    caught = 0
    fp = 0
    for key in disagree_kpts:
        x_data = kpt_data[key].get("x", {})
        y_data = kpt_data[key].get("y", {})
        if (x_data and x_data.get("margin_to_diff_ratio", float('inf')) < thresh) or \
           (y_data and y_data.get("margin_to_diff_ratio", float('inf')) < thresh):
            caught += 1
    for key in agree_kpts:
        x_data = kpt_data[key].get("x", {})
        y_data = kpt_data[key].get("y", {})
        if (x_data and x_data.get("margin_to_diff_ratio", float('inf')) < thresh) or \
           (y_data and y_data.get("margin_to_diff_ratio", float('inf')) < thresh):
            fp += 1
    total = caught + fp
    prec = caught / total if total > 0 else 0
    recall = caught / n_disagree_kpts if n_disagree_kpts > 0 else 0
    pct_flagged = 100 * total / total_kpts
    print(f"    {thresh:>10.1f} {caught:>5}/{n_disagree_kpts:<5} {fp:>10} {prec:>10.1%} {recall:>8.1%} {pct_flagged:>15.1f}%")

# Per-keypoint filter: confidence < threshold
print(f"\n  B) Confidence < threshold:")
print(f"    {'Threshold':>10} {'Caught':>10} {'FP':>10} {'Precision':>10} {'Recall':>8} {'% total flagged':>16}")
for thresh in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    caught = 0
    fp = 0
    for key in disagree_kpts:
        x_data = kpt_data[key].get("x", {})
        if x_data:
            conf = min(x_data.get("max_logit", 999), kpt_data[key].get("y", {}).get("max_logit", 999))
            if conf < thresh:
                caught += 1
    for key in agree_kpts:
        x_data = kpt_data[key].get("x", {})
        if x_data:
            conf = min(x_data.get("max_logit", 999), kpt_data[key].get("y", {}).get("max_logit", 999))
            if conf < thresh:
                fp += 1
    total = caught + fp
    prec = caught / total if total > 0 else 0
    recall = caught / n_disagree_kpts if n_disagree_kpts > 0 else 0
    pct_flagged = 100 * total / total_kpts
    print(f"    {thresh:>10.1f} {caught:>5}/{n_disagree_kpts:<5} {fp:>10} {prec:>10.1%} {recall:>8.1%} {pct_flagged:>15.1f}%")


# Per-keypoint: min_margin < threshold
print(f"\n  C) Min margin < threshold:")
print(f"    {'Threshold':>10} {'Caught':>10} {'FP':>10} {'Precision':>10} {'Recall':>8} {'% total flagged':>16}")
for thresh in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    caught = 0
    fp = 0
    for key in disagree_kpts:
        x_data = kpt_data[key].get("x", {})
        y_data = kpt_data[key].get("y", {})
        if x_data and y_data:
            min_m = min(x_data.get("margin", 999), y_data.get("margin", 999))
            if min_m < thresh:
                caught += 1
    for key in agree_kpts:
        x_data = kpt_data[key].get("x", {})
        y_data = kpt_data[key].get("y", {})
        if x_data and y_data:
            min_m = min(x_data.get("margin", 999), y_data.get("margin", 999))
            if min_m < thresh:
                fp += 1
    total = caught + fp
    prec = caught / total if total > 0 else 0
    recall = caught / n_disagree_kpts if n_disagree_kpts > 0 else 0
    pct_flagged = 100 * total / total_kpts
    print(f"    {thresh:>10.3f} {caught:>5}/{n_disagree_kpts:<5} {fp:>10} {prec:>10.1%} {recall:>8.1%} {pct_flagged:>15.1f}%")


# ============================================================
# 4. The 3 high-margin failures: what is their ratio?
# ============================================================
print("\n" + "=" * 80)
print("4. MECHANISM CHECK: For failures with high margin on one axis,")
print("   is the OTHER axis the one that actually disagrees?")
print("=" * 80)

for key in sorted(disagree_kpts):
    x_data = kpt_data[key].get("x", {})
    y_data = kpt_data[key].get("y", {})
    if not x_data or not y_data:
        continue
    min_m = min(x_data.get("margin", 0), y_data.get("margin", 0))
    if min_m >= 0.05:
        print(f"\n  {key[0]} kpt={key[1]}:")
        print(f"    X: margin={x_data['margin']:.4f}, max_diff={x_data['max_diff']:.4f}, "
              f"ratio={x_data['margin_to_diff_ratio']:.3f}, disagree={x_data['axis_disagree']}, "
              f"fp32={x_data['ax_fp32']}, fp16={x_data['ax_fp16']}")
        print(f"    Y: margin={y_data['margin']:.4f}, max_diff={y_data['max_diff']:.4f}, "
              f"ratio={y_data['margin_to_diff_ratio']:.3f}, disagree={y_data['axis_disagree']}, "
              f"fp32={y_data['ax_fp32']}, fp16={y_data['ax_fp16']}")
