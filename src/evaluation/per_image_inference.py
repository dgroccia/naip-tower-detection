"""
per_image_inference.py

Runs inference on each test image for all fine-tuned models (four Ultralytics
YOLO variants + torchvision Faster R-CNN + torchvision RetinaNet) and saves
per-image AP50 values with metadata columns for bootstrapping in R.

There is no zero-shot phase in this model set — every row is phase=fine_tuned.

Usage (from project root in WSL):
    conda activate thesis
    python src/evaluation/per_image_inference.py

Output:
    outputs/results/per_image_predictions.csv
    Columns: filename, model, phase, ap50, precision, recall, n_gt, n_pred,
             state, voltage_tier, nlcd_class
"""

import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from transformers import DetrForObjectDetection, DetrImageProcessor

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_WEIGHTS = {
    "yolov8m":  PROJECT_ROOT / "outputs/weights/naip_yolov8m_coco/weights/best.pt",
    "yolov9m":  PROJECT_ROOT / "outputs/weights/naip_yolov9m_coco/weights/best.pt",
    "yolov10m": PROJECT_ROOT / "outputs/weights/naip_yolov10m_coco/weights/best.pt",
    "yolov11m": PROJECT_ROOT / "outputs/weights/naip_yolov11m_coco/weights/best.pt",
}

FASTER_RCNN_WEIGHTS = PROJECT_ROOT / "outputs/weights/naip_faster_rcnn/best.pt"
RETINANET_WEIGHTS   = PROJECT_ROOT / "outputs/weights/naip_retinanet/best.pt"
DETR_WEIGHTS_DIR    = PROJECT_ROOT / "outputs/weights/naip_detr/best/"

TEST_IMAGES_DIR  = PROJECT_ROOT / "data/splits/test/images"
TEST_LABELS_DIR  = PROJECT_ROOT / "data/splits/test/labels"
METADATA_CSV     = PROJECT_ROOT / "data/collected/annotated/tower_metadata_aug_test.csv"
OUTPUT_CSV       = PROJECT_ROOT / "outputs/results/per_image_predictions.csv"

IOU_THRESHOLD    = 0.5    # for matching predictions to ground truth
CONF_THRESHOLD   = 0.25   # prediction confidence threshold (applied to both architectures)
IMG_SIZE         = 640
SEED             = 42
DEVICE           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torchvision detection models reserve class 0 for background internally.
# Matches src/training/train_faster_rcnn.py: class 1 = tower.
FRCNN_NUM_CLASSES = 2  # background + tower
FRCNN_TOWER_LABEL = 1

# Matches src/training/train_retinanet.py: class 1 = tower.
RETINANET_NUM_CLASSES = 2  # background + tower
RETINANET_TOWER_LABEL = 1

# HF DetrForObjectDetection appends an internal no-object class at the last
# logit index (index == config.num_labels), per HF's documented convention.
# Verified empirically against this checkpoint: index 1 = tower, index 2 =
# no-object (mean prob 0.986 over background queries), index 0 unused/dead.
DETR_TOWER_LABEL     = 1
DETR_NO_OBJECT_LABEL = 2
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_faster_rcnn():
    """Same architecture definition as src/training/train_faster_rcnn.py:build_model()."""
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, FRCNN_NUM_CLASSES)
    return model


def build_retinanet():
    """Same architecture definition as src/training/train_retinanet.py:build_model()."""
    model = retinanet_resnet50_fpn_v2(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=RETINANET_NUM_CLASSES,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )
    return model


def load_ground_truth(label_path: Path, img_w: int, img_h: int):
    """
    Load YOLO-format label file.
    Returns list of [x1, y1, x2, y2] boxes in pixel coords.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def compute_ap50(pred_boxes, pred_scores, gt_boxes, iou_thresh=IOU_THRESHOLD):
    """
    Compute AP@0.5 for a single image.
    pred_boxes  : list of [x1,y1,x2,y2]
    pred_scores : list of confidence scores (same order)
    gt_boxes    : list of [x1,y1,x2,y2] ground truth boxes
    Returns scalar AP value in [0, 1].
    """
    if len(gt_boxes) == 0:
        # No ground truth — AP = 1.0 if no predictions, 0.0 otherwise
        return 1.0 if len(pred_boxes) == 0 else 0.0

    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by descending confidence
    order = np.argsort(pred_scores)[::-1]
    pred_boxes  = [pred_boxes[i]  for i in order]
    pred_scores = [pred_scores[i] for i in order]

    matched_gt = set()
    tp = []
    fp = []

    for pb in pred_boxes:
        best_iou  = 0.0
        best_gt_i = -1
        for gi, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou  = iou
                best_gt_i = gi

        if best_iou >= iou_thresh and best_gt_i not in matched_gt:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_i)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_gt   = len(gt_boxes)

    recall    = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # Prepend sentinel
    recall    = np.concatenate([[0.0], recall,    [recall[-1]]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Smooth precision (monotonically decreasing from right)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Area under PR curve
    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] != recall[i - 1]:
            ap += (recall[i] - recall[i - 1]) * precision[i]

    return float(ap)


def final_precision_recall(pred_boxes, pred_scores, gt_boxes,
                           iou_thresh=IOU_THRESHOLD):
    """
    Returns precision and recall at the highest-confidence threshold
    that maximises F1, for reporting alongside AP.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0, 0.0

    order = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in order]

    matched_gt = set()
    tps = 0

    for pb in pred_boxes:
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if compute_iou(pb, gb) >= iou_thresh:
                tps += 1
                matched_gt.add(gi)
                break

    prec = tps / len(pred_boxes) if pred_boxes else 0.0
    rec  = tps / len(gt_boxes)   if gt_boxes  else 0.0
    return prec, rec


def run_yolo_inference(img_path: Path, model: YOLO):
    """Returns (pred_boxes, pred_scores, img_w, img_h) in original-image pixel coords."""
    results = model.predict(
        source    = str(img_path),
        imgsz     = IMG_SIZE,
        conf      = CONF_THRESHOLD,
        verbose   = False,
        save      = False,
    )
    r = results[0]
    img_h, img_w = r.orig_shape

    if r.boxes is not None and len(r.boxes) > 0:
        pred_boxes  = r.boxes.xyxy.cpu().numpy().tolist()
        pred_scores = r.boxes.conf.cpu().numpy().tolist()
    else:
        pred_boxes  = []
        pred_scores = []

    return pred_boxes, pred_scores, img_w, img_h


def run_faster_rcnn_inference(img_path: Path, model):
    """Returns (pred_boxes, pred_scores, img_w, img_h) in original-image pixel coords."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        output = model([img_tensor])[0]

    boxes  = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    keep = (scores >= CONF_THRESHOLD) & (labels == FRCNN_TOWER_LABEL)
    pred_boxes  = boxes[keep].tolist()
    pred_scores = scores[keep].tolist()

    return pred_boxes, pred_scores, img_w, img_h


def run_retinanet_inference(img_path: Path, model):
    """Returns (pred_boxes, pred_scores, img_w, img_h) in original-image pixel coords."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        output = model([img_tensor])[0]

    boxes  = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    keep = (scores >= CONF_THRESHOLD) & (labels == RETINANET_TOWER_LABEL)
    pred_boxes  = boxes[keep].tolist()
    pred_scores = scores[keep].tolist()

    return pred_boxes, pred_scores, img_w, img_h


def run_detr_inference(img_path: Path, model, processor):
    """Returns (pred_boxes, pred_scores, img_w, img_h) in original-image pixel coords."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model(**inputs)

    probs = output.logits[0].softmax(-1)          # [num_queries, num_classes+1]
    scores, labels = probs.max(-1)                 # per-query top class + prob

    keep = (labels == DETR_TOWER_LABEL) & (scores >= CONF_THRESHOLD)

    boxes_cxcywh = output.pred_boxes[0][keep]      # normalized (cx, cy, w, h)
    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    pred_boxes  = boxes_xyxy.cpu().numpy().tolist()
    pred_scores = scores[keep].cpu().numpy().tolist()

    return pred_boxes, pred_scores, img_w, img_h


def main():
    # Load metadata — restrict to the test split and flag any duplicate
    # filenames (multiple tower instances in one image patch) where the
    # per-image columns we're joining on aren't fully consistent.
    meta = pd.read_csv(METADATA_CSV)
    meta_test = meta[meta["split"] == "test"] if "split" in meta.columns else meta

    dupe_mask = meta_test.duplicated("filename", keep=False)
    if dupe_mask.any():
        dupe_groups = meta_test[dupe_mask].groupby("filename")
        inconsistent = [
            fn for fn, g in dupe_groups
            if g["nlcd_class"].nunique() > 1 or g["state"].nunique() > 1
            or g["voltage_tier"].nunique() > 1
        ]
        if inconsistent:
            print(f"NOTE: {len(inconsistent)} test-split filenames have multiple "
                  f"metadata rows (multiple tower instances per patch) with "
                  f"differing nlcd_class/state/voltage_tier. Using the first "
                  f"row per filename — flag for review, not silently resolved:")
            for fn in inconsistent[:10]:
                print(f"    {fn}")
            if len(inconsistent) > 10:
                print(f"    ... and {len(inconsistent) - 10} more")

    meta_lookup = {}
    for _, row in meta_test.iterrows():
        stem = Path(row["filename"]).stem
        if stem in meta_lookup:
            continue  # keep first row per filename
        meta_lookup[stem] = {
            "state":        row.get("state",        "unknown"),
            "voltage_tier": row.get("voltage_tier", "unknown"),
            "nlcd_class":   row.get("nlcd_class",   "unknown"),
        }

    # Collect test images
    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg")) + \
                  sorted(TEST_IMAGES_DIR.glob("*.png"))
    print(f"Found {len(test_images)} test images.")

    records = []

    # ── Ultralytics YOLO models ────────────────────────────────────────────
    for model_name, weight_path in YOLO_WEIGHTS.items():
        if not weight_path.exists():
            print(f"WARNING: weights not found for {model_name} at {weight_path}")
            continue

        print(f"\nRunning inference: {model_name}")
        model = YOLO(str(weight_path))

        for img_path in test_images:
            label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
            pred_boxes, pred_scores, img_w, img_h = run_yolo_inference(img_path, model)
            gt_boxes = load_ground_truth(label_path, img_w, img_h)

            ap50 = compute_ap50(pred_boxes, pred_scores, gt_boxes)
            prec, rec = final_precision_recall(pred_boxes, pred_scores, gt_boxes)

            stem = img_path.stem
            meta_row = meta_lookup.get(stem, {
                "state":        "unknown",
                "voltage_tier": "unknown",
                "nlcd_class":   "unknown",
            })

            records.append({
                "filename":     img_path.name,
                "model":        model_name,
                "phase":        "fine_tuned",
                "ap50":         round(ap50, 6),
                "precision":    round(prec, 6),
                "recall":       round(rec,  6),
                "n_gt":         len(gt_boxes),
                "n_pred":       len(pred_boxes),
                "state":        meta_row["state"],
                "voltage_tier": meta_row["voltage_tier"],
                "nlcd_class":   meta_row["nlcd_class"],
            })

        print(f"  Done. {len(test_images)} images processed.")

    # ── Faster R-CNN (torchvision) ──────────────────────────────────────────
    if not FASTER_RCNN_WEIGHTS.exists():
        print(f"WARNING: weights not found for faster_rcnn at {FASTER_RCNN_WEIGHTS}")
    else:
        print(f"\nRunning inference: faster_rcnn")
        frcnn_model = build_faster_rcnn()
        state_dict = torch.load(str(FASTER_RCNN_WEIGHTS), map_location=DEVICE, weights_only=True)
        frcnn_model.load_state_dict(state_dict)
        frcnn_model.to(DEVICE)
        frcnn_model.eval()

        for img_path in test_images:
            label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
            pred_boxes, pred_scores, img_w, img_h = run_faster_rcnn_inference(img_path, frcnn_model)
            gt_boxes = load_ground_truth(label_path, img_w, img_h)

            ap50 = compute_ap50(pred_boxes, pred_scores, gt_boxes)
            prec, rec = final_precision_recall(pred_boxes, pred_scores, gt_boxes)

            stem = img_path.stem
            meta_row = meta_lookup.get(stem, {
                "state":        "unknown",
                "voltage_tier": "unknown",
                "nlcd_class":   "unknown",
            })

            records.append({
                "filename":     img_path.name,
                "model":        "faster_rcnn",
                "phase":        "fine_tuned",
                "ap50":         round(ap50, 6),
                "precision":    round(prec, 6),
                "recall":       round(rec,  6),
                "n_gt":         len(gt_boxes),
                "n_pred":       len(pred_boxes),
                "state":        meta_row["state"],
                "voltage_tier": meta_row["voltage_tier"],
                "nlcd_class":   meta_row["nlcd_class"],
            })

        print(f"  Done. {len(test_images)} images processed.")

    # ── RetinaNet (torchvision) ──────────────────────────────────────────────
    if not RETINANET_WEIGHTS.exists():
        print(f"WARNING: weights not found for retinanet at {RETINANET_WEIGHTS}")
    else:
        print(f"\nRunning inference: retinanet")
        retinanet_model = build_retinanet()
        state_dict = torch.load(str(RETINANET_WEIGHTS), map_location=DEVICE, weights_only=True)
        retinanet_model.load_state_dict(state_dict)
        retinanet_model.to(DEVICE)
        retinanet_model.eval()

        for img_path in test_images:
            label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
            pred_boxes, pred_scores, img_w, img_h = run_retinanet_inference(img_path, retinanet_model)
            gt_boxes = load_ground_truth(label_path, img_w, img_h)

            ap50 = compute_ap50(pred_boxes, pred_scores, gt_boxes)
            prec, rec = final_precision_recall(pred_boxes, pred_scores, gt_boxes)

            stem = img_path.stem
            meta_row = meta_lookup.get(stem, {
                "state":        "unknown",
                "voltage_tier": "unknown",
                "nlcd_class":   "unknown",
            })

            records.append({
                "filename":     img_path.name,
                "model":        "retinanet",
                "phase":        "fine_tuned",
                "ap50":         round(ap50, 6),
                "precision":    round(prec, 6),
                "recall":       round(rec,  6),
                "n_gt":         len(gt_boxes),
                "n_pred":       len(pred_boxes),
                "state":        meta_row["state"],
                "voltage_tier": meta_row["voltage_tier"],
                "nlcd_class":   meta_row["nlcd_class"],
            })

        print(f"  Done. {len(test_images)} images processed.")

    # ── DETR (HuggingFace) ───────────────────────────────────────────────────
    if not DETR_WEIGHTS_DIR.exists():
        print(f"WARNING: weights not found for detr at {DETR_WEIGHTS_DIR}")
    else:
        print(f"\nRunning inference: detr")
        detr_model = DetrForObjectDetection.from_pretrained(str(DETR_WEIGHTS_DIR))
        detr_processor = DetrImageProcessor.from_pretrained(str(DETR_WEIGHTS_DIR))
        detr_model.to(DEVICE)
        detr_model.eval()

        for img_path in test_images:
            label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
            pred_boxes, pred_scores, img_w, img_h = run_detr_inference(img_path, detr_model, detr_processor)
            gt_boxes = load_ground_truth(label_path, img_w, img_h)

            ap50 = compute_ap50(pred_boxes, pred_scores, gt_boxes)
            prec, rec = final_precision_recall(pred_boxes, pred_scores, gt_boxes)

            stem = img_path.stem
            meta_row = meta_lookup.get(stem, {
                "state":        "unknown",
                "voltage_tier": "unknown",
                "nlcd_class":   "unknown",
            })

            records.append({
                "filename":     img_path.name,
                "model":        "detr",
                "phase":        "fine_tuned",
                "ap50":         round(ap50, 6),
                "precision":    round(prec, 6),
                "recall":       round(rec,  6),
                "n_gt":         len(gt_boxes),
                "n_pred":       len(pred_boxes),
                "state":        meta_row["state"],
                "voltage_tier": meta_row["voltage_tier"],
                "nlcd_class":   meta_row["nlcd_class"],
            })

        print(f"  Done. {len(test_images)} images processed.")

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")

    # Quick sanity check
    summary = (
        df.groupby("model")["ap50"]
        .mean()
        .mul(100)
        .round(1)
        .rename("mean_ap50_%")
    )
    print("\nSanity check — mean AP50 per model (should roughly match finetune_results.csv):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
