"""
train_yolov10m.py

Standalone YOLOv10m retraining script. Previous run was interrupted at
epoch 17/100 when the parent train_naip.py process was killed to stop
a separate, unrelated RT-DETR failure further down in that script's
sequence. This restarts YOLOv10m cleanly from the TTPLA-pretrained
weights, same config as the original interrupted run.

Usage:
    conda activate thesis
    python src/training/train_yolov10m.py
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG — matches the original train_naip.py entry for yolov10m ──────────
WEIGHTS    = "outputs/weights/ttpla_yolov10m/weights/best.pt"
NAIP_DATA  = "configs/dataset.yaml"
OUTPUT_BASE = Path("outputs/weights")
RUN_NAME   = "naip_yolov10m"  # same name — will overwrite the interrupted run

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
    log.info("Retraining YOLOv10m on expanded NAIP dataset (clean restart)")
    log.info(f"Training data: {NAIP_DATA}")

    model = YOLO(WEIGHTS)
    model.train(
        data=NAIP_DATA,
        project=str(OUTPUT_BASE),
        name=RUN_NAME,
        **TRAIN_CONFIG,
    )

    best = OUTPUT_BASE / RUN_NAME / "weights" / "best.pt"
    log.info(f"YOLOv10m training complete. Weights: {best}")


if __name__ == "__main__":
    main()
