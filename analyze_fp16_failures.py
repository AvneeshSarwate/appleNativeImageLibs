"""
Analyze fp16 vs fp32 RTMPose argmax disagreements.

For each keypoint, extract logit statistics (confidence, margin, entropy)
and determine whether these predict catastrophic fp16 failures.
"""

import numpy as np
import cv2
import coremltools as ct
from ultralytics import YOLO
from pathlib import Path
from scipy.special import softmax, log_softmax
from collections import defaultdict

SIMCC_SPLIT = 2.0
DISAGREE_THRESHOLD_PX = 5.0  # keypoints differing by more than this in pixels

# COCO-WholeBody 133 keypoint names (17 body + 6 feet + 68 face + 42 hands)
BODY_PART_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
    17: "left_big_toe", 18: "left_small_toe", 19: "left_heel",
    20: "right_big_toe", 21: "right_small_toe", 22: "right_heel",
}
# 23-90: face keypoints, 91-111: left hand, 112-132: right hand
for i in range(23, 91):
    BODY_PART_NAMES[i] = f"face_{i-23}"
for i in range(91, 112):
    BODY_PART_NAMES[i] = f"left_hand_{i-91}"
for i in range(112, 133):
    BODY_PART_NAMES[i] = f"right_hand_{i-112}"

# Group keypoints into categories
def get_kpt_group(idx):
    if idx <= 16:
        return "body"
    elif idx <= 22:
        return "feet"
    elif idx <= 90:
        return "face"
    elif idx <= 111:
        return "left_hand"
    else:
        return "right_hand"


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


def compute_logit_stats(logits_1d):
    """Compute statistics for a single keypoint's 1D logit distribution."""
    max_val = np.max(logits_1d)
    argmax_pos = np.argmax(logits_1d)

    # Sort descending to find top-2
    sorted_vals = np.sort(logits_1d)[::-1]
    second_max = sorted_vals[1]
    margin = max_val - second_max

    # Entropy of softmax distribution
    probs = softmax(logits_1d.astype(np.float64))
    # Use log_softmax for numerical stability
    log_probs = log_softmax(logits_1d.astype(np.float64))
    entropy = -np.sum(probs * log_probs)

    return {
        "max_logit": float(max_val),
        "second_max": float(second_max),
        "margin": float(margin),
        "entropy": float(entropy),
        "argmax": int(argmax_pos),
    }


# Load models
print("Loading models...")
fp32_model = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
fp16_model = ct.models.MLModel("rtmpose_fp16.mlpackage", compute_units=ct.ComputeUnit.ALL)
yolo = YOLO("yolo11s-seg.pt")
print("Models loaded.\n")

# Output names (sorted alphabetically)
OUT_X = "var_2124"  # shape (1,133,576) = simcc_x
OUT_Y = "var_2125"  # shape (1,133,768) = simcc_y

image_paths = sorted(Path("test_images").glob("*.jpg"))
print(f"Analyzing {len(image_paths)} test images...\n")

agree_stats = []     # stats for keypoints that agree
disagree_stats = []  # stats for keypoints that disagree

kpt_fail_counts = defaultdict(int)  # how many times each keypoint index fails
kpt_total_counts = defaultdict(int) # total appearances

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    r = yolo.predict(img, classes=[0], verbose=False)[0]
    if len(r.boxes) == 0:
        print(f"  {img_path.stem}: no person detected, skipping")
        continue

    idx = r.boxes.conf.argmax().item()
    bbox = r.boxes.xyxy[idx].cpu().numpy()
    blob = preprocess(img, bbox)

    # Run fp32
    fp32_outs = fp32_model.predict({"input_1": blob})
    fp32_x = fp32_outs[OUT_X]  # (1, 133, 576)
    fp32_y = fp32_outs[OUT_Y]  # (1, 133, 768)

    # Run fp16
    fp16_outs = fp16_model.predict({"input_1": blob})
    fp16_x = fp16_outs[OUT_X]  # (1, 133, 576)
    fp16_y = fp16_outs[OUT_Y]  # (1, 133, 768)

    # For each keypoint
    n_disagree = 0
    for kpt_idx in range(133):
        kpt_total_counts[kpt_idx] += 1

        # Get argmax positions
        fp32_ax = np.argmax(fp32_x[0, kpt_idx])
        fp32_ay = np.argmax(fp32_y[0, kpt_idx])
        fp16_ax = np.argmax(fp16_x[0, kpt_idx])
        fp16_ay = np.argmax(fp16_y[0, kpt_idx])

        # Pixel distance between fp32 and fp16 keypoints
        dx = abs(int(fp32_ax) - int(fp16_ax)) / SIMCC_SPLIT
        dy = abs(int(fp32_ay) - int(fp16_ay)) / SIMCC_SPLIT
        px_dist = np.sqrt(dx**2 + dy**2)

        # Compute logit stats from fp32 model (ground truth quality)
        stats_x_fp32 = compute_logit_stats(fp32_x[0, kpt_idx])
        stats_y_fp32 = compute_logit_stats(fp32_y[0, kpt_idx])
        stats_x_fp16 = compute_logit_stats(fp16_x[0, kpt_idx])
        stats_y_fp16 = compute_logit_stats(fp16_y[0, kpt_idx])

        # Confidence = min of max logits (as used in original pipeline)
        confidence = min(stats_x_fp32["max_logit"], stats_y_fp32["max_logit"])

        record = {
            "image": img_path.stem,
            "kpt_idx": kpt_idx,
            "kpt_name": BODY_PART_NAMES.get(kpt_idx, f"kpt_{kpt_idx}"),
            "kpt_group": get_kpt_group(kpt_idx),
            "px_dist": px_dist,
            "dx": dx,
            "dy": dy,
            "confidence": confidence,
            # fp32 stats
            "fp32_x_max": stats_x_fp32["max_logit"],
            "fp32_x_margin": stats_x_fp32["margin"],
            "fp32_x_entropy": stats_x_fp32["entropy"],
            "fp32_x_argmax": stats_x_fp32["argmax"],
            "fp32_y_max": stats_y_fp32["max_logit"],
            "fp32_y_margin": stats_y_fp32["margin"],
            "fp32_y_entropy": stats_y_fp32["entropy"],
            "fp32_y_argmax": stats_y_fp32["argmax"],
            # fp16 stats
            "fp16_x_max": stats_x_fp16["max_logit"],
            "fp16_x_margin": stats_x_fp16["margin"],
            "fp16_x_entropy": stats_x_fp16["entropy"],
            "fp16_x_argmax": stats_x_fp16["argmax"],
            "fp16_y_max": stats_y_fp16["max_logit"],
            "fp16_y_margin": stats_y_fp16["margin"],
            "fp16_y_entropy": stats_y_fp16["entropy"],
            "fp16_y_argmax": stats_y_fp16["argmax"],
        }

        if px_dist > DISAGREE_THRESHOLD_PX:
            disagree_stats.append(record)
            kpt_fail_counts[kpt_idx] += 1
            n_disagree += 1
        else:
            agree_stats.append(record)

    print(f"  {img_path.stem}: {n_disagree}/133 keypoints disagree (>{DISAGREE_THRESHOLD_PX}px)")

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("ANALYSIS: fp32 vs fp16 RTMPose Keypoint Agreement")
print("=" * 80)

n_agree = len(agree_stats)
n_disagree = len(disagree_stats)
n_total = n_agree + n_disagree
print(f"\nTotal keypoints analyzed: {n_total}")
print(f"  Agree  (<={DISAGREE_THRESHOLD_PX}px): {n_agree} ({100*n_agree/n_total:.1f}%)")
print(f"  Disagree (>{DISAGREE_THRESHOLD_PX}px): {n_disagree} ({100*n_disagree/n_total:.1f}%)")

# --- 1. Distribution comparison ---
print("\n" + "-" * 80)
print("1. LOGIT STATISTICS: AGREE vs DISAGREE keypoints")
print("-" * 80)

def summarize(records, field):
    vals = [r[field] for r in records]
    if not vals:
        return "N/A"
    arr = np.array(vals)
    return (f"mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
            f"std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")

for metric_name, field in [
    ("Confidence (min of max_x, max_y)", "confidence"),
    ("fp32 X margin (max - 2nd max)", "fp32_x_margin"),
    ("fp32 Y margin (max - 2nd max)", "fp32_y_margin"),
    ("fp32 X entropy", "fp32_x_entropy"),
    ("fp32 Y entropy", "fp32_y_entropy"),
    ("fp32 X max logit", "fp32_x_max"),
    ("fp32 Y max logit", "fp32_y_max"),
]:
    print(f"\n  {metric_name}:")
    print(f"    AGREE:    {summarize(agree_stats, field)}")
    print(f"    DISAGREE: {summarize(disagree_stats, field)}")


# --- 2. Which keypoints fail most ---
print("\n" + "-" * 80)
print("2. KEYPOINT FAILURE FREQUENCY (sorted by fail rate)")
print("-" * 80)

fail_rates = []
for kpt_idx in range(133):
    total = kpt_total_counts[kpt_idx]
    fails = kpt_fail_counts[kpt_idx]
    if total > 0:
        fail_rates.append((kpt_idx, fails, total, fails/total))

fail_rates.sort(key=lambda x: x[3], reverse=True)

# Show top 30 failing keypoints
print(f"\n  {'Idx':>4} {'Name':<20} {'Group':<12} {'Fails':>5} {'Total':>5} {'Rate':>6}")
for kpt_idx, fails, total, rate in fail_rates[:30]:
    if fails > 0:
        name = BODY_PART_NAMES.get(kpt_idx, f"kpt_{kpt_idx}")
        group = get_kpt_group(kpt_idx)
        print(f"  {kpt_idx:>4} {name:<20} {group:<12} {fails:>5} {total:>5} {rate:>6.1%}")

# Group-level summary
print(f"\n  By group:")
group_fails = defaultdict(int)
group_totals = defaultdict(int)
for kpt_idx in range(133):
    g = get_kpt_group(kpt_idx)
    group_fails[g] += kpt_fail_counts[kpt_idx]
    group_totals[g] += kpt_total_counts[kpt_idx]

for g in ["body", "feet", "face", "left_hand", "right_hand"]:
    f = group_fails[g]
    t = group_totals[g]
    print(f"    {g:<12}: {f:>4}/{t:>4} = {100*f/t:.1f}%")


# --- 3. Threshold analysis ---
print("\n" + "-" * 80)
print("3. THRESHOLD ANALYSIS: Can we predict failures?")
print("-" * 80)

# For each candidate threshold metric, check how well it separates agree/disagree
for metric_name, field, direction in [
    ("Confidence", "confidence", "below"),
    ("Min margin (min of x,y margin)", None, "below"),  # computed below
    ("Max entropy (max of x,y entropy)", None, "above"),  # computed below
]:
    if field is not None:
        agree_vals = np.array([r[field] for r in agree_stats])
        disagree_vals = np.array([r[field] for r in disagree_stats])
    elif "margin" in metric_name:
        agree_vals = np.array([min(r["fp32_x_margin"], r["fp32_y_margin"]) for r in agree_stats])
        disagree_vals = np.array([min(r["fp32_x_margin"], r["fp32_y_margin"]) for r in disagree_stats])
    elif "entropy" in metric_name:
        agree_vals = np.array([max(r["fp32_x_entropy"], r["fp32_y_entropy"]) for r in agree_stats])
        disagree_vals = np.array([max(r["fp32_x_entropy"], r["fp32_y_entropy"]) for r in disagree_stats])

    print(f"\n  {metric_name} (filter {direction}):")

    if direction == "below":
        # Test various thresholds
        all_vals = np.concatenate([agree_vals, disagree_vals])
        thresholds = np.percentile(all_vals, [1, 5, 10, 15, 20, 25, 30, 40, 50])
        thresholds = np.unique(np.round(thresholds, 2))
        # Also try some specific values
        extra = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
        thresholds = np.unique(np.concatenate([thresholds, extra]))
        thresholds.sort()

        print(f"    {'Threshold':>10} {'Failures caught':>16} {'False pos':>10} {'Precision':>10} {'Recall':>8}")
        for thresh in thresholds:
            caught = np.sum(disagree_vals < thresh)
            false_pos = np.sum(agree_vals < thresh)
            total_flagged = caught + false_pos
            precision = caught / total_flagged if total_flagged > 0 else 0
            recall = caught / len(disagree_vals) if len(disagree_vals) > 0 else 0
            print(f"    {thresh:>10.2f} {caught:>7}/{len(disagree_vals):<7} {false_pos:>10} {precision:>10.1%} {recall:>8.1%}")

    else:  # above
        all_vals = np.concatenate([agree_vals, disagree_vals])
        thresholds = np.percentile(all_vals, [50, 60, 70, 75, 80, 85, 90, 95, 99])
        thresholds = np.unique(np.round(thresholds, 2))
        extra = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        thresholds = np.unique(np.concatenate([thresholds, extra]))
        thresholds.sort()

        print(f"    {'Threshold':>10} {'Failures caught':>16} {'False pos':>10} {'Precision':>10} {'Recall':>8}")
        for thresh in thresholds:
            caught = np.sum(disagree_vals > thresh)
            false_pos = np.sum(agree_vals > thresh)
            total_flagged = caught + false_pos
            precision = caught / total_flagged if total_flagged > 0 else 0
            recall = caught / len(disagree_vals) if len(disagree_vals) > 0 else 0
            print(f"    {thresh:>10.2f} {caught:>7}/{len(disagree_vals):<7} {false_pos:>10} {precision:>10.1%} {recall:>8.1%}")


# --- 4. Detailed failure examples ---
print("\n" + "-" * 80)
print("4. DETAILED FAILURE EXAMPLES (worst 20 by pixel distance)")
print("-" * 80)

if disagree_stats:
    sorted_failures = sorted(disagree_stats, key=lambda r: r["px_dist"], reverse=True)
    print(f"\n  {'Image':<14} {'Kpt':<20} {'Group':<10} {'Dist':>6} "
          f"{'Conf':>6} {'Margin_X':>9} {'Margin_Y':>9} {'Entr_X':>7} {'Entr_Y':>7} "
          f"{'fp32_x':>6} {'fp16_x':>6} {'fp32_y':>6} {'fp16_y':>6}")
    for r in sorted_failures[:20]:
        print(f"  {r['image']:<14} {r['kpt_name']:<20} {r['kpt_group']:<10} {r['px_dist']:>6.1f} "
              f"{r['confidence']:>6.2f} {r['fp32_x_margin']:>9.3f} {r['fp32_y_margin']:>9.3f} "
              f"{r['fp32_x_entropy']:>7.2f} {r['fp32_y_entropy']:>7.2f} "
              f"{r['fp32_x_argmax']:>6} {r['fp16_x_argmax']:>6} "
              f"{r['fp32_y_argmax']:>6} {r['fp16_y_argmax']:>6}")


# --- 5. Histogram of confidence for agree vs disagree ---
print("\n" + "-" * 80)
print("5. CONFIDENCE DISTRIBUTION (histogram buckets)")
print("-" * 80)

agree_conf = np.array([r["confidence"] for r in agree_stats])
disagree_conf = np.array([r["confidence"] for r in disagree_stats])

buckets = [(-999, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 10), (10, 15), (15, 20), (20, 999)]
print(f"\n  {'Bucket':<12} {'Agree':>8} {'Disagree':>10} {'Fail rate':>10}")
for lo, hi in buckets:
    a = np.sum((agree_conf >= lo) & (agree_conf < hi))
    d = np.sum((disagree_conf >= lo) & (disagree_conf < hi))
    total = a + d
    rate = d / total if total > 0 else 0
    label = f"[{lo},{hi})" if hi < 999 else f"[{lo},+inf)"
    if lo == -999:
        label = f"(-inf,{hi})"
    print(f"  {label:<12} {a:>8} {d:>10} {rate:>10.1%}")


# --- 6. Min-margin analysis ---
print("\n" + "-" * 80)
print("6. MIN-MARGIN DISTRIBUTION (histogram)")
print("-" * 80)

agree_margin = np.array([min(r["fp32_x_margin"], r["fp32_y_margin"]) for r in agree_stats])
disagree_margin = np.array([min(r["fp32_x_margin"], r["fp32_y_margin"]) for r in disagree_stats])

margin_buckets = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.0),
                  (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 999)]
print(f"\n  {'Bucket':<14} {'Agree':>8} {'Disagree':>10} {'Fail rate':>10}")
for lo, hi in margin_buckets:
    a = np.sum((agree_margin >= lo) & (agree_margin < hi))
    d = np.sum((disagree_margin >= lo) & (disagree_margin < hi))
    total = a + d
    rate = d / total if total > 0 else 0
    label = f"[{lo},{hi})" if hi < 999 else f"[{lo},+inf)"
    print(f"  {label:<14} {a:>8} {d:>10} {rate:>10.1%}")


# --- 7. Key finding summary ---
print("\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)

if disagree_stats:
    # What is the max confidence among failures?
    max_fail_conf = max(r["confidence"] for r in disagree_stats)
    min_fail_conf = min(r["confidence"] for r in disagree_stats)
    # What is the max min-margin among failures?
    max_fail_margin = max(min(r["fp32_x_margin"], r["fp32_y_margin"]) for r in disagree_stats)
    min_fail_margin = min(min(r["fp32_x_margin"], r["fp32_y_margin"]) for r in disagree_stats)

    print(f"\n  Failure confidence range: [{min_fail_conf:.3f}, {max_fail_conf:.3f}]")
    print(f"  Failure min-margin range: [{min_fail_margin:.4f}, {max_fail_margin:.4f}]")

    # If we filter confidence < max_fail_conf, how many agree kpts do we also remove?
    false_pos_conf = np.sum(agree_conf < max_fail_conf)
    print(f"\n  To catch ALL {n_disagree} failures with confidence threshold < {max_fail_conf:.3f}:")
    print(f"    Would also filter {false_pos_conf} agreeing keypoints ({100*false_pos_conf/n_agree:.1f}% of agree)")
    print(f"    Total filtered: {n_disagree + false_pos_conf} of {n_total} ({100*(n_disagree+false_pos_conf)/n_total:.1f}%)")

    false_pos_margin = np.sum(agree_margin < max_fail_margin)
    print(f"\n  To catch ALL {n_disagree} failures with min-margin threshold < {max_fail_margin:.4f}:")
    print(f"    Would also filter {false_pos_margin} agreeing keypoints ({100*false_pos_margin/n_agree:.1f}% of agree)")
    print(f"    Total filtered: {n_disagree + false_pos_margin} of {n_total} ({100*(n_disagree+false_pos_margin)/n_total:.1f}%)")

    # Check if there's a clean separation
    # What percentage of failures are caught by confidence < median of disagree?
    median_conf = np.median(disagree_conf)
    caught_median = np.sum(disagree_conf < median_conf)
    fp_median = np.sum(agree_conf < median_conf)
    print(f"\n  Using disagree median confidence ({median_conf:.3f}) as threshold:")
    print(f"    Catches {caught_median}/{n_disagree} failures ({100*caught_median/n_disagree:.1f}%)")
    print(f"    False positives: {fp_median} ({100*fp_median/n_agree:.1f}% of agree)")
else:
    print("\n  No disagreements found! fp32 and fp16 agree on all keypoints.")
