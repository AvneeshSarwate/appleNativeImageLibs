"""
Part 2: Deeper analysis of fp16 failure patterns.
Focus on:
- Edge argmax (0 or max) as failure signature
- X vs Y axis failures
- Combined margin+confidence threshold
- Logit vector correlation between fp32 and fp16
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

# Collect all keypoint data
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

        # fp32 logit stats
        fp32_x_max = float(np.max(fp32_x[kpt_idx]))
        fp32_y_max = float(np.max(fp32_y[kpt_idx]))
        fp32_x_margin = float(np.sort(fp32_x[kpt_idx])[-1] - np.sort(fp32_x[kpt_idx])[-2])
        fp32_y_margin = float(np.sort(fp32_y[kpt_idx])[-1] - np.sort(fp32_y[kpt_idx])[-2])

        # Correlation of logit vectors between fp32 and fp16
        corr_x = float(np.corrcoef(fp32_x[kpt_idx], fp16_x[kpt_idx])[0, 1])
        corr_y = float(np.corrcoef(fp32_y[kpt_idx], fp16_y[kpt_idx])[0, 1])

        # Max absolute difference in logit vectors
        max_logit_diff_x = float(np.max(np.abs(fp32_x[kpt_idx] - fp16_x[kpt_idx])))
        max_logit_diff_y = float(np.max(np.abs(fp32_y[kpt_idx] - fp16_y[kpt_idx])))

        # Is the fp16 argmax at an edge (0 or max-1)?
        fp16_x_edge = fp16_ax == 0 or fp16_ax == 575
        fp16_y_edge = fp16_ay == 0 or fp16_ay == 767
        fp32_x_edge = fp32_ax == 0 or fp32_ax == 575
        fp32_y_edge = fp32_ay == 0 or fp32_ay == 767

        disagree = px_dist > DISAGREE_THRESHOLD_PX
        x_disagree = dx > DISAGREE_THRESHOLD_PX
        y_disagree = dy > DISAGREE_THRESHOLD_PX

        all_kpts.append({
            "image": img_path.stem,
            "kpt_idx": kpt_idx,
            "disagree": disagree,
            "px_dist": px_dist,
            "dx": dx, "dy": dy,
            "x_disagree": x_disagree, "y_disagree": y_disagree,
            "fp32_ax": fp32_ax, "fp32_ay": fp32_ay,
            "fp16_ax": fp16_ax, "fp16_ay": fp16_ay,
            "fp32_x_max": fp32_x_max, "fp32_y_max": fp32_y_max,
            "fp32_x_margin": fp32_x_margin, "fp32_y_margin": fp32_y_margin,
            "confidence": min(fp32_x_max, fp32_y_max),
            "min_margin": min(fp32_x_margin, fp32_y_margin),
            "corr_x": corr_x, "corr_y": corr_y,
            "max_logit_diff_x": max_logit_diff_x,
            "max_logit_diff_y": max_logit_diff_y,
            "fp16_x_edge": fp16_x_edge, "fp16_y_edge": fp16_y_edge,
            "fp32_x_edge": fp32_x_edge, "fp32_y_edge": fp32_y_edge,
        })

agree = [k for k in all_kpts if not k["disagree"]]
disagree = [k for k in all_kpts if k["disagree"]]

print(f"\nTotal: {len(all_kpts)}, Agree: {len(agree)}, Disagree: {len(disagree)}")

# ============================================================
# A. Edge argmax analysis
# ============================================================
print("\n" + "=" * 80)
print("A. EDGE ARGMAX ANALYSIS")
print("   Do fp16 failures tend to have argmax at 0 or max (edge of logit range)?")
print("=" * 80)

# For disagreeing keypoints, check if the fp16 argmax jumped to edge
n_fp16_edge = sum(1 for k in disagree if k["fp16_x_edge"] or k["fp16_y_edge"])
n_fp16_edge_x = sum(1 for k in disagree if k["fp16_x_edge"])
n_fp16_edge_y = sum(1 for k in disagree if k["fp16_y_edge"])
n_fp32_edge = sum(1 for k in disagree if k["fp32_x_edge"] or k["fp32_y_edge"])

print(f"\n  DISAGREEING keypoints ({len(disagree)} total):")
print(f"    fp16 argmax at edge (either axis): {n_fp16_edge}/{len(disagree)} = {100*n_fp16_edge/len(disagree):.1f}%")
print(f"    fp16 X edge: {n_fp16_edge_x}, fp16 Y edge: {n_fp16_edge_y}")
print(f"    fp32 argmax at edge (either axis): {n_fp32_edge}/{len(disagree)} = {100*n_fp32_edge/len(disagree):.1f}%")

n_agree_fp16_edge = sum(1 for k in agree if k["fp16_x_edge"] or k["fp16_y_edge"])
n_agree_fp32_edge = sum(1 for k in agree if k["fp32_x_edge"] or k["fp32_y_edge"])
print(f"\n  AGREEING keypoints ({len(agree)} total):")
print(f"    fp16 argmax at edge (either axis): {n_agree_fp16_edge}/{len(agree)} = {100*n_agree_fp16_edge/len(agree):.1f}%")
print(f"    fp32 argmax at edge (either axis): {n_agree_fp32_edge}/{len(agree)} = {100*n_agree_fp32_edge/len(agree):.1f}%")

# For failures, what are the exact argmax values?
print("\n  Failure argmax positions (fp16 that jumped to edge):")
edge_failures = [k for k in disagree if k["fp16_x_edge"] or k["fp16_y_edge"]]
print(f"    Count: {len(edge_failures)}")
# Count how many are exactly 0 vs max
fp16_ax_0 = sum(1 for k in edge_failures if k["fp16_ax"] == 0)
fp16_ax_575 = sum(1 for k in edge_failures if k["fp16_ax"] == 575)
fp16_ay_0 = sum(1 for k in edge_failures if k["fp16_ay"] == 0)
fp16_ay_767 = sum(1 for k in edge_failures if k["fp16_ay"] == 767)
print(f"    fp16_ax == 0: {fp16_ax_0}, fp16_ax == 575: {fp16_ax_575}")
print(f"    fp16_ay == 0: {fp16_ay_0}, fp16_ay == 767: {fp16_ay_767}")


# ============================================================
# B. Which axis fails more?
# ============================================================
print("\n" + "=" * 80)
print("B. X vs Y AXIS FAILURE BREAKDOWN")
print("=" * 80)

only_x = sum(1 for k in disagree if k["x_disagree"] and not k["y_disagree"])
only_y = sum(1 for k in disagree if not k["x_disagree"] and k["y_disagree"])
both = sum(1 for k in disagree if k["x_disagree"] and k["y_disagree"])
neither_but_combined = sum(1 for k in disagree if not k["x_disagree"] and not k["y_disagree"])

print(f"\n  Only X axis >5px: {only_x}")
print(f"  Only Y axis >5px: {only_y}")
print(f"  Both axes >5px:   {both}")
print(f"  Neither alone >5px but combined >5px: {neither_but_combined}")


# ============================================================
# C. Logit correlation analysis
# ============================================================
print("\n" + "=" * 80)
print("C. LOGIT VECTOR CORRELATION: fp32 vs fp16")
print("=" * 80)

agree_corr_x = np.array([k["corr_x"] for k in agree])
agree_corr_y = np.array([k["corr_y"] for k in agree])
disagree_corr_x = np.array([k["corr_x"] for k in disagree])
disagree_corr_y = np.array([k["corr_y"] for k in disagree])

print(f"\n  X-axis correlation:")
print(f"    AGREE:    mean={agree_corr_x.mean():.6f}  min={agree_corr_x.min():.6f}")
print(f"    DISAGREE: mean={disagree_corr_x.mean():.6f}  min={disagree_corr_x.min():.6f}")
print(f"\n  Y-axis correlation:")
print(f"    AGREE:    mean={agree_corr_y.mean():.6f}  min={agree_corr_y.min():.6f}")
print(f"    DISAGREE: mean={disagree_corr_y.mean():.6f}  min={disagree_corr_y.min():.6f}")


# ============================================================
# D. Max logit diff analysis
# ============================================================
print("\n" + "=" * 80)
print("D. MAX ABSOLUTE LOGIT DIFFERENCE (fp32 vs fp16 raw logits)")
print("=" * 80)

agree_diff_x = np.array([k["max_logit_diff_x"] for k in agree])
agree_diff_y = np.array([k["max_logit_diff_y"] for k in agree])
disagree_diff_x = np.array([k["max_logit_diff_x"] for k in disagree])
disagree_diff_y = np.array([k["max_logit_diff_y"] for k in disagree])

print(f"\n  X-axis max|fp32-fp16| per keypoint:")
print(f"    AGREE:    mean={agree_diff_x.mean():.4f}  median={np.median(agree_diff_x):.4f}  max={agree_diff_x.max():.4f}")
print(f"    DISAGREE: mean={disagree_diff_x.mean():.4f}  median={np.median(disagree_diff_x):.4f}  max={disagree_diff_x.max():.4f}")
print(f"\n  Y-axis max|fp32-fp16| per keypoint:")
print(f"    AGREE:    mean={agree_diff_y.mean():.4f}  median={np.median(agree_diff_y):.4f}  max={agree_diff_y.max():.4f}")
print(f"    DISAGREE: mean={disagree_diff_y.mean():.4f}  median={np.median(disagree_diff_y):.4f}  max={disagree_diff_y.max():.4f}")


# ============================================================
# E. Combined threshold: margin + confidence
# ============================================================
print("\n" + "=" * 80)
print("E. COMBINED THRESHOLD ANALYSIS")
print("   Find best combined filter using margin and/or confidence")
print("=" * 80)

# Since all failures have min_margin < 0.1, let's check within that zone
# whether confidence further separates
low_margin_agree = [k for k in agree if k["min_margin"] < 0.1]
low_margin_disagree = [k for k in disagree if k["min_margin"] < 0.1]

print(f"\n  Keypoints with min_margin < 0.1:")
print(f"    Agree: {len(low_margin_agree)}, Disagree: {len(low_margin_disagree)}")

if low_margin_agree and low_margin_disagree:
    lma_conf = np.array([k["confidence"] for k in low_margin_agree])
    lmd_conf = np.array([k["confidence"] for k in low_margin_disagree])
    print(f"    Agree confidence:    mean={lma_conf.mean():.3f}  median={np.median(lma_conf):.3f}  min={lma_conf.min():.3f}")
    print(f"    Disagree confidence: mean={lmd_conf.mean():.3f}  median={np.median(lmd_conf):.3f}  min={lmd_conf.min():.3f}")

# Try margin < 0.1 AND confidence < various thresholds
print(f"\n  Filter: min_margin < 0.1 AND confidence < threshold:")
print(f"    {'Conf thresh':>12} {'Caught':>8} {'FP':>8} {'Precision':>10} {'Recall':>8}")
for conf_thresh in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    caught = sum(1 for k in disagree if k["min_margin"] < 0.1 and k["confidence"] < conf_thresh)
    fp = sum(1 for k in agree if k["min_margin"] < 0.1 and k["confidence"] < conf_thresh)
    total = caught + fp
    prec = caught / total if total > 0 else 0
    recall = caught / len(disagree) if disagree else 0
    print(f"    {conf_thresh:>12.1f} {caught:>5}/{len(disagree):<3} {fp:>8} {prec:>10.1%} {recall:>8.1%}")

# Also try: margin alone at different thresholds
print(f"\n  Filter: min_margin < threshold (margin alone):")
print(f"    {'Margin thresh':>14} {'Caught':>8} {'FP':>8} {'Precision':>10} {'Recall':>8}")
for m_thresh in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
    caught = sum(1 for k in disagree if k["min_margin"] < m_thresh)
    fp = sum(1 for k in agree if k["min_margin"] < m_thresh)
    total = caught + fp
    prec = caught / total if total > 0 else 0
    recall = caught / len(disagree) if disagree else 0
    print(f"    {m_thresh:>14.3f} {caught:>5}/{len(disagree):<3} {fp:>8} {prec:>10.1%} {recall:>8.1%}")

# Check: are there failures with margin >= 0.05?
high_margin_failures = [k for k in disagree if k["min_margin"] >= 0.05]
print(f"\n  Failures with min_margin >= 0.05: {len(high_margin_failures)}")
for k in high_margin_failures:
    print(f"    kpt={k['kpt_idx']} ({k['image']}) margin_x={k['fp32_x_margin']:.4f} margin_y={k['fp32_y_margin']:.4f} "
          f"conf={k['confidence']:.3f} dist={k['px_dist']:.1f}px "
          f"fp32=({k['fp32_ax']},{k['fp32_ay']}) fp16=({k['fp16_ax']},{k['fp16_ay']})")


# ============================================================
# F. The "axis-specific margin" approach
# ============================================================
print("\n" + "=" * 80)
print("F. AXIS-SPECIFIC MARGIN ANALYSIS")
print("   Check if the FAILING axis always has low margin on THAT axis")
print("=" * 80)

# For each failure, identify which axis(es) failed and check margin on that axis
x_fail_with_margin = []
y_fail_with_margin = []
for k in disagree:
    if k["x_disagree"]:
        x_fail_with_margin.append(k["fp32_x_margin"])
    if k["y_disagree"]:
        y_fail_with_margin.append(k["fp32_y_margin"])

if x_fail_with_margin:
    arr = np.array(x_fail_with_margin)
    print(f"\n  X-axis failures ({len(arr)} total):")
    print(f"    X-margin: mean={arr.mean():.4f}  max={arr.max():.4f}  min={arr.min():.4f}")
    print(f"    All < 0.05: {np.all(arr < 0.05)}")
    print(f"    All < 0.1:  {np.all(arr < 0.1)}")

if y_fail_with_margin:
    arr = np.array(y_fail_with_margin)
    print(f"\n  Y-axis failures ({len(arr)} total):")
    print(f"    Y-margin: mean={arr.mean():.4f}  max={arr.max():.4f}  min={arr.min():.4f}")
    print(f"    All < 0.05: {np.all(arr < 0.05)}")
    print(f"    All < 0.1:  {np.all(arr < 0.1)}")

# Now the key question: if we use axis-specific margin < threshold,
# how well does it predict axis-specific failure?
print(f"\n  AXIS-SPECIFIC FILTER: flag keypoint on axis A if margin_A < threshold")
print(f"    (catches axis-level failures; a keypoint is 'wrong' if either axis is flagged)")

# Compute per-axis failure flags
for thresh in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    # A keypoint is predicted-fail if EITHER x_margin < thresh OR y_margin < thresh
    caught = sum(1 for k in disagree
                 if (k["x_disagree"] and k["fp32_x_margin"] < thresh) or
                    (k["y_disagree"] and k["fp32_y_margin"] < thresh))
    # False positive: agree keypoint where either axis margin < thresh
    fp = sum(1 for k in agree
             if k["fp32_x_margin"] < thresh or k["fp32_y_margin"] < thresh)
    total = caught + fp
    prec = caught / total if total > 0 else 0
    recall = caught / len(disagree) if disagree else 0
    print(f"    margin < {thresh:.2f}: caught {caught}/{len(disagree)} ({recall:.1%}), FP={fp}, precision={prec:.1%}")
