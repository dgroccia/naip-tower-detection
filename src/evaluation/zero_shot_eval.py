"""
Zero-shot evaluation: TTPLA-pretrained weights on NAIP test set.
No fine-tuning — measures the raw domain gap.
"""
import logging
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "yolov8m":  "/home/dante/thesis/outputs/yolo_runs/yolov8m_ep100_bs16_img640_20251204_195740/weights/best.pt",
    "yolov8l":  "outputs/weights/ttpla_yolov8l/weights/best.pt",
    "yolov9m":  "/home/dante/thesis/outputs/yolo_runs/yolov9m_20251205_095002/weights/best.pt",
    "yolov10m": "outputs/weights/ttpla_yolov10m/weights/best.pt",
    "rtdetr_l": "/home/dante/thesis/outputs/yolo_runs/rtdetr_l_20251217_233410/weights/best.pt",
}

results = []

for name, weights in WEIGHTS.items():
    log.info(f"\nEvaluating {name} zero-shot on NAIP test set...")
    try:
        model = YOLO(weights)
        metrics = model.val(
            data="configs/dataset.yaml",
            split="test",
            imgsz=640,
            batch=16,
            device=0,
            verbose=False,
            name=f"zeroshot_{name}",
            project="outputs/results",
        )
        row = {
            "model":     name,
            "phase":     "zero_shot",
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
            "model": name, "phase": "zero_shot",
            "map50": None, "map50_95": None,
            "precision": None, "recall": None,
        })

df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / "zero_shot_results.csv", index=False)
log.info(f"\nResults saved to {RESULTS_DIR}/zero_shot_results.csv")
log.info("\n" + df.to_string(index=False))
