"""
collect_raw_predictions.py

Runs inference on all 461 test images for all 7 models (yolov8m, yolov9m,
yolov10m, yolov11m, faster_rcnn, retinanet, detr) and saves every raw
detection with its confidence score and TP/FP status against ground truth
at IoU >= 0.5. This is the per-detection data needed to sweep confidence
thresholds and build proper precision-recall curves — per_image_inference.py
only keeps the aggregate AP50/precision/recall at a single fixed threshold
(CONF_THRESHOLD=0.25), which isn't enough to trace out a full curve.

Inference here uses a near-zero confidence floor (COLLECT_CONF_THRESHOLD)
instead of the 0.25 used elsewhere in the eval pipeline. Sweeping thresholds
over detections that were already filtered at 0.25 would truncate the curve
at whatever recall/precision that fixed threshold happens to land on.

Ground-truth accounting: every test image gets at least one row in the
output, even if a model produces zero detections for it. Zero-detection
images get a placeholder row with confidence=-1 and is_tp=0 so that
plot_pr_curves.py can still recover the correct total ground-truth count
per model (needed for the recall denominator) without that placeholder
ever being selected as a "kept" detection during threshold sweeping
(thresholds are swept over confidence in [0, 1], so -1 is never >= threshold).

Usage (from project root in WSL):
    conda activate thesis
    python src/evaluation/collect_raw_predictions.py

Output:
    outputs/results/raw_predictions.csv
    Columns: model, filename, confidence, is_tp, n_gt
"""

import logging
import random
from functools import partial
from pathlib import Path
from typing import Optional

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

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
DETR_WEIGHTS_DIR     = PROJECT_ROOT / "outputs/weights/naip_detr/best/"

TEST_IMAGES_DIR  = PROJECT_ROOT / "data/splits/test/images"
TEST_LABELS_DIR  = PROJECT_ROOT / "data/splits/test/labels"
OUTPUT_CSV       = PROJECT_ROOT / "outputs/results/raw_predictions.csv"

IOU_THRESHOLD          = 0.5     # for matching predictions to ground truth
COLLECT_CONF_THRESHOLD = 0.001   # near-zero floor so the PR sweep isn't truncated
IMG_SIZE                = 640
SEED                    = 42
DEVICE                  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torchvision detection models reserve class 0 for background internally.
# Matches src/training/train_faster_rcnn.py: class 1 = tower.
FRCNN_NUM_CLASSES = 2  # background + tower
FRCNN_TOWER_LABEL = 1

# Matches src/training/train_retinanet.py: class 1 = tower.
RETINANET_NUM_CLASSES = 2  # background + tower
RETINANET_TOWER_LABEL = 1

# HF DetrForObjectDetection appends an internal no-object class at the last
# logit index. Matches src/evaluation/per_image_inference.py.
DETR_TOWER_LABEL     = 1
DETR_NO_OBJECT_LABEL = 2
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_faster_rcnn() -> torch.nn.Module:
    """Same architecture definition as src/training/train_faster_rcnn.py:build_model().

    Returns:
        Uninitialized Faster R-CNN model with a 2-class box predictor head.
    """
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, FRCNN_NUM_CLASSES)
    return model


def build_retinanet() -> torch.nn.Module:
    """Same architecture definition as src/training/train_retinanet.py:build_model().

    Returns:
        Uninitialized RetinaNet model with a 2-class classification head.
    """
    model = retinanet_resnet50_fpn_v2(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=RETINANET_NUM_CLASSES,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )
    return model


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Load YOLO-format ground truth boxes for one image.

    Args:
        label_path: Path to the YOLO .txt label file.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List of [x1, y1, x2, y2] boxes in pixel coordinates.
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


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes.

    Args:
        box_a: First box in pixel coordinates.
        box_b: Second box in pixel coordinates.

    Returns:
        IoU as a float in [0, 1].
    """
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


def match_detections(pred_boxes: list[list[float]], pred_scores: list[float],
                      gt_boxes: list[list[float]],
                      iou_thresh: float = IOU_THRESHOLD) -> list[int]:
    """Greedily match predictions to ground truth, highest confidence first.

    Each prediction is matched to its best-IoU unmatched ground truth box;
    a match counts as a true positive only if that IoU clears iou_thresh.
    Matches the logic in per_image_inference.py:compute_ap50 so is_tp here
    is consistent with the AP50 numbers already reported elsewhere.

    Args:
        pred_boxes: Predicted [x1,y1,x2,y2] boxes, any order.
        pred_scores: Confidence score per predicted box, same order.
        gt_boxes: Ground truth [x1,y1,x2,y2] boxes for this image.
        iou_thresh: Minimum IoU for a match to count as a true positive.

    Returns:
        List of 0/1 flags (is_tp), in the same order as pred_boxes/pred_scores.
    """
    if not pred_boxes:
        return []

    order = np.argsort(pred_scores)[::-1]
    is_tp = [0] * len(pred_boxes)
    matched_gt = set()

    for i in order:
        pb = pred_boxes[i]
        best_iou, best_gt_i = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_gt_i = iou, gi
        if best_iou >= iou_thresh and best_gt_i != -1:
            is_tp[i] = 1
            matched_gt.add(best_gt_i)

    return is_tp


def run_yolo_inference(img_path: Path, model: YOLO) -> tuple[list, list, int, int]:
    """Returns (pred_boxes, pred_scores, img_w, img_h) in original-image pixel coords."""
    results = model.predict(
        source  = str(img_path),
        imgsz   = IMG_SIZE,
        conf    = COLLECT_CONF_THRESHOLD,
        verbose = False,
        save    = False,
    )
    r = results[0]
    img_h, img_w = r.orig_shape

    if r.boxes is not None and len(r.boxes) > 0:
        pred_boxes  = r.boxes.xyxy.cpu().numpy().tolist()
        pred_scores = r.boxes.conf.cpu().numpy().tolist()
    else:
        pred_boxes, pred_scores = [], []

    return pred_boxes, pred_scores, img_w, img_h


def run_torchvision_inference(img_path: Path, model: torch.nn.Module,
                               tower_label: int) -> tuple[list, list, int, int]:
    """Returns (pred_boxes, pred_scores, img_w, img_h) in original-image pixel coords.

    Shared by Faster R-CNN and RetinaNet — both are torchvision detection
    models with the same output dict shape.
    """
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    img_tensor = torchvision.transforms.functional.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        output = model([img_tensor])[0]

    boxes  = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    keep = (scores >= COLLECT_CONF_THRESHOLD) & (labels == tower_label)
    return boxes[keep].tolist(), scores[keep].tolist(), img_w, img_h


def run_detr_inference(img_path: Path, model: DetrForObjectDetection,
                        processor: DetrImageProcessor) -> tuple[list, list, int, int]:
    """Returns (pred_boxes, pred_scores, img_w, img_h) in original-image pixel coords."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model(**inputs)

    probs = output.logits[0].softmax(-1)
    scores, labels = probs.max(-1)

    keep = (labels == DETR_TOWER_LABEL) & (scores >= COLLECT_CONF_THRESHOLD)

    boxes_cxcywh = output.pred_boxes[0][keep]
    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    return boxes_xyxy.cpu().numpy().tolist(), scores[keep].cpu().numpy().tolist(), img_w, img_h


def collect_model_predictions(model_name: str, test_images: list[Path],
                               infer_fn) -> list[dict]:
    """Run one model over all test images and build raw detection rows.

    Args:
        model_name: Name to tag every row with (e.g. "yolov8m").
        test_images: Sorted list of test image paths.
        infer_fn: Callable(img_path) -> (pred_boxes, pred_scores, img_w, img_h).

    Returns:
        List of row dicts with keys model, filename, confidence, is_tp, n_gt.
        Images with zero detections still contribute one placeholder row
        (confidence=-1, is_tp=0) so the ground-truth count isn't lost.
    """
    rows = []
    for img_path in test_images:
        label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
        pred_boxes, pred_scores, img_w, img_h = infer_fn(img_path)
        gt_boxes = load_ground_truth(label_path, img_w, img_h)
        n_gt = len(gt_boxes)

        if not pred_boxes:
            rows.append({
                "model": model_name, "filename": img_path.name,
                "confidence": -1.0, "is_tp": 0, "n_gt": n_gt,
            })
            continue

        is_tp_flags = match_detections(pred_boxes, pred_scores, gt_boxes)
        for score, is_tp in zip(pred_scores, is_tp_flags):
            rows.append({
                "model": model_name, "filename": img_path.name,
                "confidence": round(float(score), 6), "is_tp": is_tp, "n_gt": n_gt,
            })

    return rows


def main() -> None:
    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg")) + \
                  sorted(TEST_IMAGES_DIR.glob("*.png"))
    log.info(f"Found {len(test_images)} test images.")

    all_rows: list[dict] = []

    # ── Ultralytics YOLO models ────────────────────────────────────────────
    for model_name, weight_path in YOLO_WEIGHTS.items():
        if not weight_path.exists():
            log.warning(f"weights not found for {model_name} at {weight_path}")
            continue
        log.info(f"Running inference: {model_name}")
        model = YOLO(str(weight_path))
        infer_fn = partial(run_yolo_inference, model=model)
        all_rows.extend(collect_model_predictions(model_name, test_images, infer_fn))
        log.info(f"  Done. {len(test_images)} images processed.")

    # ── Faster R-CNN (torchvision) ──────────────────────────────────────────
    if not FASTER_RCNN_WEIGHTS.exists():
        log.warning(f"weights not found for faster_rcnn at {FASTER_RCNN_WEIGHTS}")
    else:
        log.info("Running inference: faster_rcnn")
        frcnn_model = build_faster_rcnn()
        state_dict = torch.load(str(FASTER_RCNN_WEIGHTS), map_location=DEVICE, weights_only=True)
        frcnn_model.load_state_dict(state_dict)
        frcnn_model.to(DEVICE)
        frcnn_model.eval()
        infer_fn = partial(run_torchvision_inference, model=frcnn_model, tower_label=FRCNN_TOWER_LABEL)
        all_rows.extend(collect_model_predictions("faster_rcnn", test_images, infer_fn))
        log.info(f"  Done. {len(test_images)} images processed.")

    # ── RetinaNet (torchvision) ──────────────────────────────────────────────
    if not RETINANET_WEIGHTS.exists():
        log.warning(f"weights not found for retinanet at {RETINANET_WEIGHTS}")
    else:
        log.info("Running inference: retinanet")
        retinanet_model = build_retinanet()
        state_dict = torch.load(str(RETINANET_WEIGHTS), map_location=DEVICE, weights_only=True)
        retinanet_model.load_state_dict(state_dict)
        retinanet_model.to(DEVICE)
        retinanet_model.eval()
        infer_fn = partial(run_torchvision_inference, model=retinanet_model, tower_label=RETINANET_TOWER_LABEL)
        all_rows.extend(collect_model_predictions("retinanet", test_images, infer_fn))
        log.info(f"  Done. {len(test_images)} images processed.")

    # ── DETR (HuggingFace) ───────────────────────────────────────────────────
    if not DETR_WEIGHTS_DIR.exists():
        log.warning(f"weights not found for detr at {DETR_WEIGHTS_DIR}")
    else:
        log.info("Running inference: detr")
        detr_model = DetrForObjectDetection.from_pretrained(str(DETR_WEIGHTS_DIR))
        detr_processor = DetrImageProcessor.from_pretrained(str(DETR_WEIGHTS_DIR))
        detr_model.to(DEVICE)
        detr_model.eval()
        infer_fn = partial(run_detr_inference, model=detr_model, processor=detr_processor)
        all_rows.extend(collect_model_predictions("detr", test_images, infer_fn))
        log.info(f"  Done. {len(test_images)} images processed.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved {len(df)} rows to {OUTPUT_CSV}")

    log.info("Sanity check — detections per model (excludes placeholder rows):")
    real = df[df["confidence"] >= 0]
    print(real.groupby("model").size().rename("n_detections").to_string())


if __name__ == "__main__":
    main()
