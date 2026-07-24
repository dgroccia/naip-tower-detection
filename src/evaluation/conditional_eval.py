"""
Conditional evaluation by geographic region, voltage tier, and land use.
Runs fine-tuned models on test set subsets stratified by each dimension.

Covers all 7 models in the current benchmark: yolov8m, yolov9m, yolov10m,
yolov11m (Ultralytics), faster_rcnn, retinanet (torchvision), and detr
(HuggingFace) — no yolov8l, no rtdetr_l, matching the rest of the
evaluation pipeline.
"""
import logging
from functools import partial

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from transformers import DetrForObjectDetection, DetrImageProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "yolov8m":  "outputs/weights/naip_yolov8m_coco/weights/best.pt",
    "yolov9m":  "outputs/weights/naip_yolov9m_coco/weights/best.pt",
    "yolov10m": "outputs/weights/naip_yolov10m_coco/weights/best.pt",
    "yolov11m": "outputs/weights/naip_yolov11m_coco/weights/best.pt",
}
FASTER_RCNN_WEIGHTS = "outputs/weights/naip_faster_rcnn/best.pt"
RETINANET_WEIGHTS   = "outputs/weights/naip_retinanet/best.pt"
DETR_WEIGHTS_DIR     = "outputs/weights/naip_detr/best/"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FRCNN_NUM_CLASSES = 2  # background + tower
RETINANET_NUM_CLASSES = 2  # background + tower

# HF DetrForObjectDetection: class 1 = tower, class 2 = no-object.
# Matches src/evaluation/per_image_inference.py.
DETR_TOWER_LABEL     = 1
DETR_NO_OBJECT_LABEL = 2


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


def compute_map50(preds, targets):
    from torchmetrics.detection import MeanAveragePrecision
    if not targets:
        return None
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    return metric.compute()["map_50"].item()

def get_predictions(model, image_paths, label_dir):
    """Ultralytics YOLO models — ground truth labels kept at YOLO class id (0=tower)."""
    preds, targets = [], []
    for img_path in image_paths:
        results = model(str(img_path), verbose=False, device=0)
        boxes   = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            preds.append({
                "boxes":  boxes.xyxy.cpu(),
                "scores": boxes.conf.cpu(),
                "labels": boxes.cls.cpu().long(),
            })
        else:
            preds.append({
                "boxes":  torch.zeros((0,4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            })
        lbl = Path(label_dir) / f"{img_path.stem}.txt"
        gt_boxes, gt_labels = [], []
        if lbl.exists():
            for line in lbl.read_text().strip().split('\n'):
                p = line.strip().split()
                if len(p) == 5:
                    cls, xc, yc, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    gt_boxes.append([(xc-bw/2)*640, (yc-bh/2)*640, (xc+bw/2)*640, (yc+bh/2)*640])
                    gt_labels.append(cls)
        if gt_boxes:
            targets.append({"boxes": torch.tensor(gt_boxes, dtype=torch.float32), "labels": torch.tensor(gt_labels, dtype=torch.long)})
        else:
            targets.append({"boxes": torch.zeros((0,4), dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.long)})
    return preds, targets


def get_torchvision_predictions(model, image_paths, label_dir):
    """torchvision Faster R-CNN / RetinaNet — both return a list of dicts with
    boxes/scores/labels per image. Ground truth labels kept at torchvision
    class id (1=tower, 0=background), matching the model's own prediction
    label convention."""
    preds, targets = [], []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img).to(DEVICE)
        with torch.no_grad():
            output = model([img_tensor])[0]
        preds.append({
            "boxes":  output["boxes"].cpu(),
            "scores": output["scores"].cpu(),
            "labels": output["labels"].cpu(),
        })

        lbl = Path(label_dir) / f"{img_path.stem}.txt"
        gt_boxes, gt_labels = [], []
        if lbl.exists():
            for line in lbl.read_text().strip().split('\n'):
                p = line.strip().split()
                if len(p) == 5:
                    _, xc, yc, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    gt_boxes.append([(xc-bw/2)*640, (yc-bh/2)*640, (xc+bw/2)*640, (yc+bh/2)*640])
                    gt_labels.append(1)  # torchvision class 1 = tower
        if gt_boxes:
            targets.append({"boxes": torch.tensor(gt_boxes, dtype=torch.float32), "labels": torch.tensor(gt_labels, dtype=torch.long)})
        else:
            targets.append({"boxes": torch.zeros((0,4), dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.long)})
    return preds, targets


def get_detr_predictions(model, processor, image_paths, label_dir):
    """HuggingFace DETR — softmax over logits, class 1 = tower, class 2 =
    no-object (filtered out); pred_boxes converted from normalized cxcywh to
    pixel xyxy, class 1 probability used as confidence score. Ground truth
    labels kept at the same convention (1=tower) as the model's predictions."""
    preds, targets = [], []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model(**inputs)

        probs = output.logits[0].softmax(-1)  # [num_queries, num_classes+1]
        labels = probs.argmax(-1)
        keep = labels != DETR_NO_OBJECT_LABEL

        boxes_cxcywh = output.pred_boxes[0][keep]
        cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        tower_scores = probs[keep, DETR_TOWER_LABEL]
        n_kept = int(keep.sum().item())

        preds.append({
            "boxes":  boxes_xyxy.cpu(),
            "scores": tower_scores.cpu(),
            "labels": torch.full((n_kept,), DETR_TOWER_LABEL, dtype=torch.long),
        })

        lbl = Path(label_dir) / f"{img_path.stem}.txt"
        gt_boxes, gt_labels = [], []
        if lbl.exists():
            for line in lbl.read_text().strip().split('\n'):
                p = line.strip().split()
                if len(p) == 5:
                    _, xc, yc, bw_, bh_ = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    gt_boxes.append([(xc-bw_/2)*640, (yc-bh_/2)*640, (xc+bw_/2)*640, (yc+bh_/2)*640])
                    gt_labels.append(DETR_TOWER_LABEL)
        if gt_boxes:
            targets.append({"boxes": torch.tensor(gt_boxes, dtype=torch.float32), "labels": torch.tensor(gt_labels, dtype=torch.long)})
        else:
            targets.append({"boxes": torch.zeros((0,4), dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.long)})
    return preds, targets

def main():
    meta     = pd.read_csv("data/collected/annotated/tower_metadata.csv")
    test_dir = Path("data/splits/test/images")
    lbl_dir  = Path("data/splits/test/labels")

    # Only keep test split metadata
    test_meta = meta[meta["split"] == "test"].copy()
    log.info(f"Test set: {len(test_meta)} annotations across {test_meta['filename'].nunique()} images")

    dimensions = {
        "region":       "state",
        "voltage_tier": "voltage_tier",
        "land_use":     "land_use",
    }

    all_results = []

    def run_conditional(model_name, predict_fn):
        for dim_name, col in dimensions.items():
            for group_val in sorted(test_meta[col].dropna().unique()):
                subset = test_meta[test_meta[col] == group_val]
                img_stems = subset["filename"].unique()
                img_paths = [test_dir / f"{s}.jpg" for s in img_stems if (test_dir / f"{s}.jpg").exists()]

                if len(img_paths) < 3:
                    log.warning(f"  {model_name} {dim_name}={group_val}: only {len(img_paths)} images, skipping")
                    continue

                preds, targets = predict_fn(img_paths, lbl_dir)
                map50 = compute_map50(preds, targets)

                log.info(f"  {model_name} {dim_name}={group_val}: mAP@0.5={map50:.4f} ({len(img_paths)} images)")
                all_results.append({
                    "model":     model_name,
                    "dimension": dim_name,
                    "group":     group_val,
                    "n_images":  len(img_paths),
                    "map50":     round(map50, 4) if map50 is not None else None,
                })

    for model_name, weights in WEIGHTS.items():
        log.info(f"\nLoading {model_name}...")
        model = YOLO(weights)
        model.model.eval()
        run_conditional(model_name, lambda paths, lbls, m=model: get_predictions(m, paths, lbls))

    log.info(f"\nLoading faster_rcnn...")
    frcnn_model = build_faster_rcnn()
    state_dict = torch.load(FASTER_RCNN_WEIGHTS, map_location=DEVICE, weights_only=True)
    frcnn_model.load_state_dict(state_dict)
    frcnn_model.to(DEVICE)
    frcnn_model.eval()
    run_conditional("faster_rcnn", lambda paths, lbls: get_torchvision_predictions(frcnn_model, paths, lbls))

    log.info(f"\nLoading retinanet...")
    retinanet_model = build_retinanet()
    state_dict = torch.load(RETINANET_WEIGHTS, map_location=DEVICE, weights_only=True)
    retinanet_model.load_state_dict(state_dict)
    retinanet_model.to(DEVICE)
    retinanet_model.eval()
    run_conditional("retinanet", lambda paths, lbls: get_torchvision_predictions(retinanet_model, paths, lbls))

    log.info(f"\nLoading detr...")
    detr_model = DetrForObjectDetection.from_pretrained(DETR_WEIGHTS_DIR)
    detr_processor = DetrImageProcessor.from_pretrained(DETR_WEIGHTS_DIR)
    detr_model.to(DEVICE)
    detr_model.eval()
    run_conditional("detr", lambda paths, lbls: get_detr_predictions(detr_model, detr_processor, paths, lbls))

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "conditional_results.csv", index=False)
    log.info(f"\nSaved to {RESULTS_DIR}/conditional_results.csv")
    log.info("\n" + df.to_string(index=False))

if __name__ == "__main__":
    main()
