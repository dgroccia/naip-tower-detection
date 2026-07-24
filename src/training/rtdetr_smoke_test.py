"""
rtdetr_smoke_test.py

Minimal diagnostic: trains RT-DETR-L with the standard AdamW config on a
tiny subset of real data for a handful of epochs, watched live, to isolate
whether the NaN instability is environment-level (PyTorch nightly + Blackwell)
or specific to the full dataset/training scale.

Usage:
    conda activate thesis
    python src/training/rtdetr_smoke_test.py
"""
import logging
import shutil
from pathlib import Path
from ultralytics import RTDETR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Build a tiny smoke-test dataset: 20 images from the real training set ────
SRC_IMAGES = Path("data/augmented/train/images")
SRC_LABELS = Path("data/augmented/train/labels")
SMOKE_DIR  = Path("data/smoke_test")
N_IMAGES   = 20

def build_smoke_dataset():
    img_out = SMOKE_DIR / "images"
    lbl_out = SMOKE_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images = sorted(SRC_IMAGES.glob("*.jpg"))[:N_IMAGES]
    for img in images:
        shutil.copy(img, img_out / img.name)
        lbl = SRC_LABELS / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, lbl_out / lbl.name)

    # Minimal dataset yaml — train and val point to the same tiny set,
    # this is purely a NaN/no-NaN diagnostic, not a real training run
    yaml_content = f"""path: {SMOKE_DIR.resolve()}
train: images
val: images
nc: 1
names:
  0: tower
"""
    yaml_path = SMOKE_DIR / "smoke.yaml"
    yaml_path.write_text(yaml_content)
    log.info(f"Smoke test dataset: {len(images)} images at {SMOKE_DIR}")
    return yaml_path


def main():
    yaml_path = build_smoke_dataset()

    log.info("Starting RT-DETR-L smoke test: AdamW, 10 epochs, 20 images")
    log.info("Watching for NaN appearance — original failures occurred by epoch 1-5")

    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=str(yaml_path),
        project="outputs/weights",
        name="rtdetr_smoke_test",
        epochs=10,
        batch=4,
        imgsz=640,
        device=0,
        seed=42,
        workers=2,
        optimizer="AdamW",
        lr0=0.0001,
        warmup_epochs=3.0,
        weight_decay=0.0001,
        exist_ok=True,
        plots=False,
        val=False,  # skip val to keep this fast and focused purely on train loss NaN
    )

    log.info("\nSmoke test complete. Check results.csv for NaN.")
    log.info("If NaN appears here too, this confirms environment-level instability,")
    log.info("not a dataset-scale or hyperparameter-specific issue.")


if __name__ == "__main__":
    main()
