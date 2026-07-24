"""
train_yolo_coco.py

Trains all four YOLO-family architectures (YOLOv8m, YOLOv9m, YOLOv10m,
YOLOv11m) directly on the NAIP dataset, initialized from official COCO
pretrained weights. Replaces the prior two-stage TTPLA-then-NAIP approach
entirely -- no TTPLA pretraining step anywhere in this pipeline.

Trains on the augmented training set (data/augmented/train), evaluates
on the held-out val/test splits unchanged from the original thesis split.

Usage:
    conda activate thesis
    python src/training/train_yolo_coco.py
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Official Ultralytics COCO-pretrained weights -- auto-downloaded if not cached
WEIGHTS = {
    "yolov8m":  "yolov8m.pt",
    "yolov9m":  "yolov9m.pt",
    "yolov10m": "yolov10m.pt",
    "yolov11m": "yolo11m.pt",   # note: Ultralytics naming is "yolo11" not "yolov11"
}

NAIP_DATA   = "configs/dataset.yaml"
OUTPUT_BASE = Path("outputs/weights")

TRAIN_CONFIG = {
    "epochs":   100,
    "patience": 20,
    "batch":    16,
    "imgsz":    640,
    "device":   0,
    "seed":     42,
    "workers":  4,
    "exist_ok": True,
}
# ─────────────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log.info(f"Fine-tuning {len(WEIGHTS)} YOLO-family models on NAIP")
    log.info("Initialization: official COCO-pretrained weights (no TTPLA step)")
    log.info(f"Training data: {NAIP_DATA}\n")

    for name, weights in WEIGHTS.items():
        log.info(f"{'='*60}")
        log.info(f"Training {name} from {weights}...")
        model = YOLO(weights)
        model.train(
            data=NAIP_DATA,
            project=str(OUTPUT_BASE),
            name=f"naip_{name}_coco",
            **TRAIN_CONFIG,
        )
        best = OUTPUT_BASE / f"naip_{name}_coco" / "weights" / "best.pt"
        log.info(f"{name} complete. Weights: {best}")

    log.info("\nAll four YOLO-family models trained from COCO init.")
    log.info("Next: run RT-DETR-L and Faster R-CNN scripts (already COCO-initialized).")


if __name__ == "__main__":
    main()
