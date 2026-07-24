"""
Fine-tune all 5 models on NAIP training split.
Initialize from TTPLA-pretrained weights for fair comparison.
"""
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NAIP_DATA   = "configs/dataset.yaml"
OUTPUT_BASE = Path("outputs/weights")

WEIGHTS = {
    "yolov8m":  "/home/dante/thesis/outputs/yolo_runs/yolov8m_ep100_bs16_img640_20251204_195740/weights/best.pt",
    "yolov8l":  "outputs/weights/ttpla_yolov8l/weights/best.pt",
    "yolov9m":  "/home/dante/thesis/outputs/yolo_runs/yolov9m_20251205_095002/weights/best.pt",
    "yolov10m": "outputs/weights/ttpla_yolov10m/weights/best.pt",
    "rtdetr_l": "/home/dante/thesis/outputs/yolo_runs/rtdetr_l_20251217_233410/weights/best.pt",
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
    log.info(f"Fine-tuning {len(WEIGHTS)} models on NAIP")
    log.info(f"Training data: {NAIP_DATA}")
    log.info(f"Augmented samples: 6,352\n")

    for name, weights in WEIGHTS.items():
        log.info(f"{'='*50}")
        log.info(f"Fine-tuning {name}...")
        model = YOLO(weights)
        model.train(
            data=NAIP_DATA,
            project=str(OUTPUT_BASE),
            name=f"naip_{name}",
            **TRAIN_CONFIG,
        )
        best = OUTPUT_BASE / f"naip_{name}" / "weights" / "best.pt"
        log.info(f"{name} complete. Weights: {best}")

    log.info("\nAll fine-tuning complete.")
    log.info("Next: python src/evaluation/finetune_eval.py")

if __name__ == "__main__":
    main()
