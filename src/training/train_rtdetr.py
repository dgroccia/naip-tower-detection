"""
train_rtdetr.py

Standalone RT-DETR-L training script with corrected optimizer configuration.
Previous run failed with all-NaN losses from epoch 1, consistent with the
default SGD optimizer being unstable for transformer attention layers at
the learning rate tuned for YOLO-family CNNs.

Fix: AdamW optimizer, lower learning rate, explicit warmup.

Usage:
    conda activate thesis
    python src/training/train_rtdetr.py
"""
import logging
from pathlib import Path
from ultralytics import RTDETR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Start from TTPLA-pretrained RT-DETR weights (same as original failed run)
WEIGHTS    = "/home/dante/thesis/outputs/yolo_runs/rtdetr_l_20251217_233410/weights/best.pt"
NAIP_DATA  = "configs/dataset.yaml"
OUTPUT_BASE = Path("outputs/weights")
RUN_NAME   = "naip_rtdetr_l_fixed"

TRAIN_CONFIG = {
    "epochs":     100,
    "patience":   20,
    "batch":      16,
    "imgsz":      640,
    "device":     0,
    "seed":       42,
    "workers":    4,
    "exist_ok":   True,
    # ── The fix ──
    "optimizer":  "AdamW",
    "lr0":        0.0001,    # much lower than YOLO's default SGD lr0
    "lrf":        0.01,      # final LR as a fraction of lr0 (cosine decay)
    "warmup_epochs": 3.0,    # gradual ramp-up to avoid early instability
    "weight_decay": 0.0001,
}
# ─────────────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log.info("Fine-tuning RT-DETR-L on expanded NAIP dataset (corrected config)")
    log.info(f"Training data: {NAIP_DATA}")
    log.info(f"Optimizer: {TRAIN_CONFIG['optimizer']}, lr0={TRAIN_CONFIG['lr0']}")

    model = RTDETR(WEIGHTS)
    model.train(
        data=NAIP_DATA,
        project=str(OUTPUT_BASE),
        name=RUN_NAME,
        **TRAIN_CONFIG,
    )

    best = OUTPUT_BASE / RUN_NAME / "weights" / "best.pt"
    log.info(f"RT-DETR-L fine-tuning complete. Weights: {best}")
    log.info("\nVerify results.csv shows non-NaN losses before proceeding to evaluation.")


if __name__ == "__main__":
    main()
