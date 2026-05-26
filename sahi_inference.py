import os
import json
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm
import time
from ultralytics import YOLO
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH  = "best_yolo9c_960.pt"
TEST_IMAGES = "dataset/images/test"
TEST_LABELS = "dataset/labels/test"
OUTPUT_DIR  = "results/sahi_yolo9c"
IOU_THRESH  = 0.4      # IoU threshold to count a detection as TP
CONFIDENCE_THRESH = 0.15  # Minimum confidence for SAHI predictions
OVERLAP = 0.3       # Overlap ratio for slicing 
SLICE_SIZE = 640   # Slice size for SAHI

os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.time()

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    device="cuda:0",
    confidence_threshold=CONFIDENCE_THRESH,
)



# ─────────────────────────────────────────────
# STANDARD INFERENCE (no SAHI)
# ─────────────────────────────────────────────
images = sorted([f for f in os.listdir(TEST_IMAGES) if f.endswith(".PNG")])
print(f"Running inference on {len(images)} images...")

std_model = YOLO(MODEL_PATH)
std_all_tp, std_all_fp, std_all_fn = 0, 0, 0
std_per_image = []

print("Running standard inference...")
for img_name in tqdm(images, desc="Standard inference"):
    img_path   = os.path.join(TEST_IMAGES, img_name)
    label_path = os.path.join(TEST_LABELS, img_name.replace(".PNG", ".txt"))

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # Load ground truth
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    gt_boxes.append([
                        (cx - w/2)*W, (cy - h/2)*H,
                        (cx + w/2)*W, (cy + h/2)*H,
                    ])

    # Standard inference
    result = std_model(img_path, conf=CONFIDENCE_THRESH, verbose=False)[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()

    # Match predictions to ground truth
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_j = 0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            xi1, yi1 = max(pred[0], gt[0]), max(pred[1], gt[1])
            xi2, yi2 = min(pred[2], gt[2]), min(pred[3], gt[3])
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            area_pred = (pred[2]-pred[0]) * (pred[3]-pred[1])
            area_gt   = (gt[2]-gt[0])    * (gt[3]-gt[1])
            iou = inter / (area_pred + area_gt - inter + 1e-8)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= IOU_THRESH:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    std_all_tp += tp
    std_all_fp += fp
    std_all_fn += fn

    missed_boxes = [gt_boxes[j] for j in range(len(gt_boxes)) if j not in matched_gt]
    std_per_image.append({
        "image": img_name,
        "gt_count": len(gt_boxes), "pred_count": len(pred_boxes),
        "tp": tp, "fp": fp, "fn": fn,
        "missed_boxes": missed_boxes,
    })

std_precision = std_all_tp / (std_all_tp + std_all_fp + 1e-8)
std_recall    = std_all_tp / (std_all_tp + std_all_fn + 1e-8)
std_f1        = 2 * std_precision * std_recall / (std_precision + std_recall + 1e-8)

print(f"\n=== STANDARD METRICS ===")
print(f"  Precision: {std_precision:.4f}")
print(f"  Recall:    {std_recall:.4f}")
print(f"  F1:        {std_f1:.4f}")
print(f"  TP: {std_all_tp}  FP: {std_all_fp}  FN: {std_all_fn}")

# ─────────────────────────────────────────────
# RUN SAHI + COMPUTE METRICS
# ─────────────────────────────────────────────

images = sorted([f for f in os.listdir(TEST_IMAGES) if f.endswith(".PNG")])
print(f"Running SAHI on {len(images)} images...")

all_tp, all_fp, all_fn = 0, 0, 0

for i, img_name in enumerate(tqdm(images, desc="SAHI inference")):
    img_path   = os.path.join(TEST_IMAGES, img_name)
    label_path = os.path.join(TEST_LABELS, img_name.replace(".PNG", ".txt"))

    # Get image size to convert YOLO labels to pixel coords
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # Load ground truth boxes from YOLO label file
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    gt_boxes.append([
                        (cx - w / 2) * W,
                        (cy - h / 2) * H,
                        (cx + w / 2) * W,
                        (cy + h / 2) * H,
                    ])

    # Run SAHI sliced inference
    result = get_sliced_prediction(
        image=img_path,
        detection_model=model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        verbose=0,
    )
    pred_boxes = [det.bbox.to_xyxy() for det in result.object_prediction_list]

    # Match predictions to ground truth using IoU
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_j = 0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            # Compute IoU
            xi1, yi1 = max(pred[0], gt[0]), max(pred[1], gt[1])
            xi2, yi2 = min(pred[2], gt[2]), min(pred[3], gt[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_pred = (pred[2]-pred[0]) * (pred[3]-pred[1])
            area_gt   = (gt[2]-gt[0])   * (gt[3]-gt[1])
            iou = inter / (area_pred + area_gt - inter + 1e-8)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= IOU_THRESH:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)

    all_tp += tp
    all_fp += fp
    all_fn += fn

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(images)}")

# ─────────────────────────────────────────────
# COMPUTE AND SAVE METRICS
# ─────────────────────────────────────────────

precision = all_tp / (all_tp + all_fp + 1e-8)
recall    = all_tp / (all_tp + all_fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

metrics = {
    "model":                MODEL_PATH,
    "slice_height":         SLICE_SIZE,
    "slice_width":          SLICE_SIZE,
    "overlap":              OVERLAP,
    "confidence_threshold": CONFIDENCE_THRESH,
    "iou_threshold":        IOU_THRESH,
    "total_images":         len(images),
    "tp": all_tp, "fp": all_fp, "fn": all_fn,
    "precision":            round(precision, 4),
    "recall":               round(recall, 4),
    "f1":                   round(f1, 4),
}

file_name = f"sahi_metrics_{int(time.time())}.json"

metrics_path = os.path.join(OUTPUT_DIR, file_name)
with open(metrics_path, "w") as f:
        json.dump({
        "standard": {
            "summary": {
                "model": MODEL_PATH,
                "confidence_threshold": CONFIDENCE_THRESH,
                "iou_threshold": IOU_THRESH,
                "total_images": len(images),
                "tp": std_all_tp, "fp": std_all_fp, "fn": std_all_fn,
                "precision": round(std_precision, 4),
                "recall":    round(std_recall, 4),
                "f1":        round(std_f1, 4),
            },
        },
        "sahi": {
            "summary": metrics,
        },
    }, f, indent=2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal inference time: {elapsed_time:.2f} seconds")

print(f"\n=== SAHI METRICS ===")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  TP: {all_tp}  FP: {all_fp}  FN: {all_fn}")
print(f"  Saved → {metrics_path}")  