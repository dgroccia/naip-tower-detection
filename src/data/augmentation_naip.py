import argparse
import logging
import yaml
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def build_bbox_params():
    return A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3, clip=True)

def build_pipeline(angle, cfg):
    geo  = cfg["geometric"]
    phot = cfg["photometric"]
    nocc = cfg["noise_occlusion"]
    return A.Compose([
        A.Rotate(limit=(angle, angle), border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.HorizontalFlip(p=geo["horizontal_flip_p"]),
        A.VerticalFlip(p=geo["vertical_flip_p"]),
        A.RandomResizedCrop(size=(640, 640), scale=(geo["crop_scale_min"], 1.0), ratio=(0.9, 1.1), p=geo["crop_p"]),
        A.CLAHE(clip_limit=(phot["clahe_clip_min"], phot["clahe_clip_max"]), p=phot["clahe_p"]),
        A.RandomBrightnessContrast(brightness_limit=phot["brightness_limit"], contrast_limit=phot["contrast_limit"], p=phot["brightness_contrast_p"]),
        A.HueSaturationValue(hue_shift_limit=phot["hue_shift_limit"], sat_shift_limit=phot["sat_shift_limit"], val_shift_limit=10, p=phot["hue_sat_p"]),
        A.ChannelShuffle(p=phot["channel_shuffle_p"]),
        A.GaussNoise(var_limit=(nocc["gauss_var_min"], nocc["gauss_var_max"]), p=nocc["gauss_p"]),
        A.CoarseDropout(max_holes=nocc["dropout_max_holes"], max_height=nocc["dropout_max_size"], max_width=nocc["dropout_max_size"], p=nocc["dropout_p"]),
        A.RandomShadow(p=nocc["shadow_p"]),
        A.GridDistortion(distort_limit=nocc["grid_distort_limit"], border_mode=cv2.BORDER_REFLECT_101, p=nocc["grid_distort_p"]),
    ], bbox_params=build_bbox_params())

def read_yolo(path):
    class_ids, bboxes = [], []
    if not Path(path).exists():
        return class_ids, bboxes
    for line in Path(path).read_text().strip().split('\n'):
        parts = line.strip().split()
        if len(parts) == 5:
            class_ids.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])
    return class_ids, bboxes

def write_yolo(path, class_ids, bboxes):
    with open(path, 'w') as f:
        for cid, bbox in zip(class_ids, bboxes):
            f.write(f"{int(cid)} {' '.join(f'{v:.6f}' for v in bbox)}\n")

def smoke_test(image_dir, label_dir, cfg, n=5):
    log.info(f"Running smoke test on {n} images...")
    images = sorted(list(Path(image_dir).glob("*.jpg")))[:n]
    for img_path in images:
        image = cv2.imread(str(img_path))
        class_ids, bboxes = read_yolo(Path(label_dir) / f"{img_path.stem}.txt")
        if not bboxes:
            continue
        pipeline = build_pipeline(90, cfg)
        result = pipeline(image=image, bboxes=bboxes, class_labels=class_ids)
        assert result["image"] is not None
        for bbox in result["bboxes"]:
            x, y, w, h = bbox
            assert 0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0
        log.info(f"  {img_path.stem}: OK — {len(result['bboxes'])} boxes")
    log.info("Smoke test PASSED")

def augment(image_dir, label_dir, output_dir, cfg, seed=42):
    np.random.seed(seed)
    out_img = Path(output_dir) / "images"
    out_lbl = Path(output_dir) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(list(Path(image_dir).glob("*.jpg")))
    angles = list(range(0, 360, cfg["geometric"]["rotation_step_degrees"]))
    total = 0

    for img_path in tqdm(images, desc="Augmenting"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        class_ids, bboxes = read_yolo(Path(label_dir) / f"{img_path.stem}.txt")
        stem = img_path.stem

        # Save original
        cv2.imwrite(str(out_img / f"{stem}_aug_000.jpg"), image)
        write_yolo(out_lbl / f"{stem}_aug_000.txt", class_ids, bboxes)
        total += 1

        for angle in angles[1:]:
            pipeline = build_pipeline(angle, cfg)
            try:
                result = pipeline(image=image, bboxes=bboxes, class_labels=class_ids)
            except Exception as e:
                continue
            if not result["bboxes"]:
                continue
            a = f"{angle:03d}"
            cv2.imwrite(str(out_img / f"{stem}_aug_{a}.jpg"), result["image"])
            write_yolo(out_lbl / f"{stem}_aug_{a}.txt", result["class_labels"], result["bboxes"])
            total += 1

    log.info(f"Done. {total} augmented images written.")
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  type=str, required=True)
    parser.add_argument("--label_dir",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config",     type=str, default="configs/augmentation.yaml")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke_test:
        smoke_test(args.image_dir, args.label_dir, cfg)
        return

    augment(args.image_dir, args.label_dir, args.output_dir, cfg, args.seed)

if __name__ == "__main__":
    main()
