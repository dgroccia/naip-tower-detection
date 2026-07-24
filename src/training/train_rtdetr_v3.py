"""
train_rtdetr.py (v3)

Full RT-DETR-L training run, AdamW, matching standard DETR-family practice.
Smoke test (10 epochs, 20 images) confirmed the environment can train without
NaN under AdamW, so the full-scale instability seen in two previous attempts
is likely scale-dependent, not a fundamental environment incompatibility.

Adds a NaN watchdog via a custom Ultralytics callback that raises immediately
when NaN first appears in training loss, instead of letting the run grind
through 90+ wasted epochs before being noticed.

Usage:
    conda activate thesis
    python src/training/train_rtdetr.py
"""
import logging
import math
from pathlib import Path
from ultralytics import RTDETR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

WEIGHTS    = "rtdetr-l.pt"   # official COCO-pretrained weights
NAIP_DATA  = "configs/dataset.yaml"
OUTPUT_BASE = Path("outputs/weights")
RUN_NAME   = "naip_rtdetr_l_v3"

TRAIN_CONFIG = {
    "epochs":        100,
    "patience":       20,
    "batch":          16,
    "imgsz":         640,
    "device":           0,
    "seed":            42,
    "workers":          4,
    "exist_ok":      True,
    "optimizer":  "AdamW",
    "lr0":        0.0001,
    "lrf":          0.01,
    "warmup_epochs": 3.0,
    "weight_decay": 0.0001,
}


def on_train_batch_end(trainer):
    """NaN watchdog — stop immediately rather than burning epochs on a dead run."""
    loss = trainer.loss
    if loss is not None:
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        if math.isnan(loss_val):
            log.error(f"NaN detected at epoch {trainer.epoch}, batch loss. Stopping training.")
            trainer.stop_training = True
            raise RuntimeError(
                f"NaN loss detected at epoch {trainer.epoch}. "
                f"Training halted by watchdog instead of running to completion."
            )


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log.info("Fine-tuning RT-DETR-L on full NAIP dataset (v3: AdamW + NaN watchdog)")
    log.info(f"Training data: {NAIP_DATA}")
    log.info(f"Optimizer: {TRAIN_CONFIG['optimizer']}, lr0={TRAIN_CONFIG['lr0']}")
    log.info("NaN watchdog active — will halt immediately if NaN reappears, not run to epoch 100")

    model = RTDETR(WEIGHTS)
    model.add_callback("on_train_batch_end", on_train_batch_end)

    try:
        model.train(
            data=NAIP_DATA,
            project=str(OUTPUT_BASE),
            name=RUN_NAME,
            **TRAIN_CONFIG,
        )
    except RuntimeError as e:
        log.error(f"Training halted: {e}")
        log.error("NaN reappeared at full scale despite clean smoke test.")
        log.error("This points to a scale-dependent instability (large batch size,")
        log.error("full dataset gradient statistics, or long-schedule LR decay)")
        log.error("rather than a fundamental environment incompatibility.")
        return

    best = OUTPUT_BASE / RUN_NAME / "weights" / "best.pt"
    log.info(f"RT-DETR-L fine-tuning complete. Weights: {best}")


if __name__ == "__main__":
    main()
