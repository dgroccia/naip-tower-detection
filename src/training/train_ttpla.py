"""
Train YOLOv8l and YOLOv10m on TTPLA for apples-to-apples zero-shot comparison.
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

TTPLA_DATA  = "/home/dante/thesis/data/ttpla_yolo/data.yaml"
OUTPUT_BASE = Path("outputs/weights")

MODELS = {
    "yolov8l":  "yolov8l.pt",
    "yolov10m": "yolov10m.pt",
}

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

def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log.info(f"TTPLA data: {TTPLA_DATA}")
    log.info(f"Training: {list(MODELS.keys())}")
    log.info("Estimated time: ~40-80 min per model\n")

    for name, weights in MODELS.items():
        log.info(f"{'='*50}")
        log.info(f"Training {name} on TTPLA...")
        model = YOLO(weights)
        model.train(
            data=TTPLA_DATA,
            project=str(OUTPUT_BASE),
            name=f"ttpla_{name}",
            **TRAIN_CONFIG,
        )
        best = OUTPUT_BASE / f"ttpla_{name}" / "weights" / "best.pt"
        log.info(f"{name} complete. Weights: {best}")

    log.info("\nAll done. Next: python src/evaluation/zero_shot_eval.py")

if __name__ == "__main__":
    main()
