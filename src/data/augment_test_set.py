"""
augment_test_set.py

Applies geometric augmentation (rotation + flips only) to the 97-image
test set, producing a 3,492-image augmented test pool for evaluation.

Per Ed's suggestion: 36 rotations (0-350 degrees, 10-degree steps) x
97 base images = 3,492 augmented test images. No photometric or noise
augmentation -- geometric transforms only to preserve evaluation integrity.

Usage:
    conda activate thesis
    python src/data/augment_test_set.py
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TEST_IMAGES  = Path("data/splits/test/images")
TEST_LABELS  = Path("data/splits/test/labels")
OUT_IMAGES   = Path("data/augmented/test/images")
OUT_LABELS   = Path("data/augmented/test/labels")
PATCH_SIZE   = 640
SEED         = 42
MIN_VIS      = 0.3
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

bbox_params = A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=MIN_VIS,
    clip=True,
)

def build_pipeline(angle):
    return A.Compose([
        A.Rotate(limit=(angle, angle),
                 border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], bbox_params=bbox_params)

def read_yolo(path):
    class_ids, bboxes = [], []
    if not path.exists():
        return class_ids, bboxes
    for line in path.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 5:
            class_ids.append(int(float(parts[0])))
            bboxes.append([float(x) for x in parts[1:]])
    return class_ids, bboxes

def write_yolo(path, class_ids, bboxes):
    lines = [f"{int(cid)} {' '.join(f'{v:.6f}' for v in box)}"
             for cid, box in zip(class_ids, bboxes)]
    path.write_text("\n".join(lines))

angles = list(range(0, 360, 10))  # 36 angles
images = sorted(TEST_IMAGES.glob("*.jpg"))
log.info(f"Base test images: {len(images)}")
log.info(f"Target augmented count: {len(images) * len(angles)}")

written = 0
skipped = 0

for img_path in tqdm(images, desc="Augmenting test images"):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    label_path = TEST_LABELS / f"{img_path.stem}.txt"
    class_ids, bboxes = read_yolo(label_path)

    for angle in angles:
        pipeline = build_pipeline(angle)
        aug = pipeline(
            image=img_rgb,
            bboxes=bboxes,
            class_labels=class_ids,
        )

        out_stem = f"{img_path.stem}_aug_{angle:03d}"
        out_img  = OUT_IMAGES / f"{out_stem}.jpg"
        out_lbl  = OUT_LABELS / f"{out_stem}.txt"

        if out_img.exists():
            written += 1
            continue

        bgr = cv2.cvtColor(aug["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img), bgr)

        if aug["bboxes"]:
            write_yolo(out_lbl, aug["class_labels"], aug["bboxes"])
            written += 1
        else:
            # Write empty label file so image is included in eval
            out_lbl.write_text("")
            written += 1

log.info(f"Done. {written} augmented test images written to {OUT_IMAGES}")
log.info(f"Skipped (already existed): {skipped}")
