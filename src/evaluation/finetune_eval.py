"""
Evaluate fine-tuned NAIP models on the test set.

Four Ultralytics YOLO models are evaluated with the standard .val() API.
Faster R-CNN (torchvision) can't use .val(), so its mAP is computed with
torchmetrics.detection.MeanAveragePrecision instead; precision/recall for
that row are computed by aggregate IoU matching at CONF_THRESHOLD, to keep
the results table schema consistent across all five models.
"""
import logging
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "outputs/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "yolov8m":  PROJECT_ROOT / "outputs/weights/naip_yolov8m_coco/weights/best.pt",
    "yolov9m":  PROJECT_ROOT / "outputs/weights/naip_yolov9m_coco/weights/best.pt",
    "yolov10m": PROJECT_ROOT / "outputs/weights/naip_yolov10m_coco/weights/best.pt",
    "yolov11m": PROJECT_ROOT / "outputs/weights/naip_yolov11m_coco/weights/best.pt",
}
FASTER_RCNN_WEIGHTS = PROJECT_ROOT / "outputs/weights/naip_faster_rcnn/best.pt"

TEST_IMAGES_DIR = PROJECT_ROOT / "data/splits/test/images"
TEST_LABELS_DIR = PROJECT_ROOT / "data/splits/test/labels"

CONF_THRESHOLD = 0.25   # matches per_image_inference.py, for the precision/recall row only
IOU_THRESHOLD  = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FRCNN_NUM_CLASSES = 2  # background + tower
FRCNN_TOWER_LABEL = 1


def build_faster_rcnn():
    """Same architecture definition as src/training/train_faster_rcnn.py:build_model()."""
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, FRCNN_NUM_CLASSES)
    return model


def compute_iou(box_a, box_b):
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


def aggregate_precision_recall(preds, targets, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    """
    Corpus-level precision/recall (not averaged per image) at a fixed
    confidence threshold, via greedy IoU matching.
    preds/targets: torchvision-style dict lists with "boxes"/"scores"/"labels".
    """
    tp = fp = fn = 0
    for pred, target in zip(preds, targets):
        keep = pred["scores"] >= conf_thresh
        pred_boxes = pred["boxes"][keep].tolist()
        pred_scores = pred["scores"][keep].tolist()
        gt_boxes = target["boxes"].tolist()

        order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        matched_gt = set()
        for i in order:
            pb = pred_boxes[i]
            best_iou, best_gi = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def load_yolo_ground_truth(label_path: Path, img_w: int, img_h: int, label_value: int):
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels
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
            labels.append(label_value)
    return boxes, labels


def evaluate_faster_rcnn():
    """Runs FRCNN over the test set and computes mAP (torchmetrics) plus
    aggregate precision/recall (manual IoU matching)."""
    model = build_faster_rcnn()
    state_dict = torch.load(str(FASTER_RCNN_WEIGHTS), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg")) + sorted(TEST_IMAGES_DIR.glob("*.png"))

    preds, targets = [], []
    for img_path in test_images:
        label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img_tensor = torchvision.transforms.functional.to_tensor(img).to(DEVICE)

        with torch.no_grad():
            output = model([img_tensor])[0]

        preds.append({
            "boxes":  output["boxes"].cpu(),
            "scores": output["scores"].cpu(),
            "labels": output["labels"].cpu(),
        })

        gt_boxes, gt_labels = load_yolo_ground_truth(label_path, img_w, img_h, FRCNN_TOWER_LABEL)
        if gt_boxes:
            targets.append({
                "boxes":  torch.tensor(gt_boxes, dtype=torch.float32),
                "labels": torch.tensor(gt_labels, dtype=torch.long),
            })
        else:
            targets.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long),
            })

    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    computed = metric.compute()

    precision, recall = aggregate_precision_recall(preds, targets)

    return {
        "model":     "faster_rcnn",
        "phase":     "fine_tuned",
        "map50":     round(float(computed["map_50"]), 4),
        "map50_95":  round(float(computed["map"]), 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }


def main():
    results = []

    for name, weights in WEIGHTS.items():
        log.info(f"\nEvaluating fine-tuned {name}...")
        try:
            model = YOLO(str(weights))
            metrics = model.val(
                data="configs/dataset.yaml",
                split="test",
                imgsz=640,
                batch=16,
                device=0,
                verbose=False,
                name=f"finetune_{name}",
                project="outputs/results",
            )
            row = {
                "model":     name,
                "phase":     "fine_tuned",
                "map50":     round(float(metrics.box.map50), 4),
                "map50_95":  round(float(metrics.box.map), 4),
                "precision": round(float(metrics.box.mp), 4),
                "recall":    round(float(metrics.box.mr), 4),
            }
            results.append(row)
            log.info(f"  mAP@0.5={row['map50']}  P={row['precision']}  R={row['recall']}")
        except Exception as e:
            log.error(f"  {name}: FAILED — {e}")
            results.append({
                "model": name, "phase": "fine_tuned",
                "map50": None, "map50_95": None,
                "precision": None, "recall": None,
            })

    log.info(f"\nEvaluating fine-tuned faster_rcnn (torchmetrics)...")
    try:
        row = evaluate_faster_rcnn()
        results.append(row)
        log.info(f"  mAP@0.5={row['map50']}  P={row['precision']}  R={row['recall']}")
    except Exception as e:
        log.error(f"  faster_rcnn: FAILED — {e}")
        results.append({
            "model": "faster_rcnn", "phase": "fine_tuned",
            "map50": None, "map50_95": None,
            "precision": None, "recall": None,
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "finetune_results.csv", index=False)
    log.info(f"\nResults saved to {RESULTS_DIR}/finetune_results.csv")
    log.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
