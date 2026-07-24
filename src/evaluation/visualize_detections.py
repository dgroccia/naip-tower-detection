"""
visualize_detections.py

Generates a 3x2 qualitative detection-example figure for the paper: for each
of three land use classes (agriculture, forest, suburban), one YOLOv8m test
image with a strong AP@0.5 (>0.8) and one with a weak AP@0.5 (<0.3), both with
ground truth (green) and predicted (red) boxes drawn on the NAIP patch.

This is the one figure in the pipeline generated directly in Python
(matplotlib) rather than R/ggplot2 — it overlays inference output on raw
image patches rather than plotting summary statistics, which ggplot2 handles
awkwardly. All other figures remain R/ggplot2 per project convention.

Usage (from project root in WSL):
    conda activate thesis
    python src/evaluation/visualize_detections.py

Output:
    outputs/figures/fig8_detection_examples.png
    logs/visualize_detections.log
"""

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_WEIGHTS      = PROJECT_ROOT / "outputs/weights/naip_yolov8m_coco/weights/best.pt"
TEST_IMAGES_DIR   = PROJECT_ROOT / "data/splits/test/images"
TEST_LABELS_DIR   = PROJECT_ROOT / "data/splits/test/labels"
PREDICTIONS_CSV   = PROJECT_ROOT / "outputs/results/per_image_predictions.csv"
METADATA_CSV      = PROJECT_ROOT / "data/collected/annotated/tower_metadata.csv"
OUTPUT_PNG        = PROJECT_ROOT / "outputs/figures/fig8_detection_examples.png"
LOG_FILE          = PROJECT_ROOT / "logs/visualize_detections.log"

LAND_USES        = ["agriculture", "forest", "suburban"]
GOOD_AP_THRESH   = 0.8
BAD_AP_THRESH    = 0.3
CONF_THRESHOLD   = 0.25
SEED             = 42
DPI              = 300
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_predictions() -> pd.DataFrame:
    """
    Load per-image AP@0.5 predictions for YOLOv8m, fine-tuned phase.

    Returns:
        DataFrame with columns filename, ap50, n_gt (indexed by filename stem).
    """
    df = pd.read_csv(PREDICTIONS_CSV)
    df = df[(df["model"] == "yolov8m") & (df["phase"] == "fine_tuned")].copy()
    df["stem"] = df["filename"].apply(lambda f: Path(f).stem)
    return df


def load_land_use_lookup() -> dict[str, str]:
    """
    Build a filename-stem -> land_use lookup from the tower metadata CSV,
    restricted to the test split.

    Returns:
        Dict mapping image filename stem to land_use string.
    """
    meta = pd.read_csv(METADATA_CSV)
    meta_test = meta[meta["split"] == "test"]
    lookup: dict[str, str] = {}
    for _, row in meta_test.iterrows():
        lookup.setdefault(row["filename"], row["land_use"])
    return lookup


def select_examples(preds: pd.DataFrame, land_use_lookup: dict[str, str]) -> dict[str, dict[str, pd.Series]]:
    """
    Select one good (AP@0.5 > 0.8) and one bad (AP@0.5 < 0.3) test image per
    land use class, restricted to images with at least one ground truth box.

    Args:
        preds: Per-image YOLOv8m predictions (from load_predictions).
        land_use_lookup: filename stem -> land_use (from load_land_use_lookup).

    Returns:
        Nested dict: {land_use: {"good": row, "bad": row}}, rows are pandas Series.
    """
    preds = preds.copy()
    preds["land_use"] = preds["stem"].map(land_use_lookup)

    rng = np.random.default_rng(SEED)
    selections: dict[str, dict[str, pd.Series]] = {}

    for land_use in LAND_USES:
        subset = preds[(preds["land_use"] == land_use) & (preds["n_gt"] > 0)]
        good_pool = subset[subset["ap50"] > GOOD_AP_THRESH]
        bad_pool  = subset[subset["ap50"] < BAD_AP_THRESH]

        if good_pool.empty or bad_pool.empty:
            raise RuntimeError(
                f"Could not find both a good and bad example for land use "
                f"'{land_use}' (good pool={len(good_pool)}, bad pool={len(bad_pool)})"
            )

        good_row = subset.loc[rng.choice(good_pool.index.to_numpy())]
        bad_row  = subset.loc[rng.choice(bad_pool.index.to_numpy())]
        selections[land_use] = {"good": good_row, "bad": bad_row}

        log.info(
            "%-11s good=%s (AP50=%.3f)  bad=%s (AP50=%.3f)",
            land_use, good_row["filename"], good_row["ap50"],
            bad_row["filename"], bad_row["ap50"],
        )

    return selections


def load_ground_truth_boxes(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """
    Load YOLO-format ground truth boxes and convert to pixel [x1, y1, x2, y2].

    Args:
        label_path: Path to the YOLO .txt label file.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List of [x1, y1, x2, y2] boxes in pixel coordinates.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def run_yolo_inference(model: YOLO, img_path: Path) -> list[list[float]]:
    """
    Run YOLOv8m inference on one image at the project's confidence threshold.

    Args:
        model: Loaded Ultralytics YOLO model.
        img_path: Path to the test image.

    Returns:
        List of predicted [x1, y1, x2, y2] boxes in pixel coordinates.
    """
    results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False, save=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    return r.boxes.xyxy.cpu().numpy().tolist()


def draw_panel(ax: plt.Axes, img_path: Path, gt_boxes: list, pred_boxes: list, title: str) -> None:
    """
    Draw one detection example panel: image with GT (green) and Pred (red) boxes.

    Args:
        ax: Matplotlib axes to draw into.
        img_path: Path to the NAIP patch image.
        gt_boxes: List of [x1, y1, x2, y2] ground truth boxes.
        pred_boxes: List of [x1, y1, x2, y2] predicted boxes.
        title: Subplot title (AP@0.5, land use class, good/bad label).
    """
    img = Image.open(img_path).convert("RGB")
    ax.imshow(np.array(img))

    for x1, y1, x2, y2 in gt_boxes:
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor="#00CC00", facecolor="none", linewidth=2))
        ax.text(x1, y1 - 4, "GT", color="#00CC00", fontsize=8, fontweight="bold")

    for x1, y1, x2, y2 in pred_boxes:
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor="red", facecolor="none", linewidth=2))
        ax.text(x1, y2 + 12, "Pred", color="red", fontsize=8, fontweight="bold")

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def build_figure(model: YOLO, selections: dict[str, dict[str, pd.Series]]) -> None:
    """
    Assemble the 3x2 (land use x good/bad) detection example grid and save it.

    Args:
        model: Loaded Ultralytics YOLO model, used for inference on each panel.
        selections: Output of select_examples().
    """
    fig, axes = plt.subplots(3, 2, figsize=(9, 13))

    for row, land_use in enumerate(LAND_USES):
        for col, label in enumerate(["good", "bad"]):
            example = selections[land_use][label]
            img_path = TEST_IMAGES_DIR / example["filename"]
            label_path = TEST_LABELS_DIR / (example["stem"] + ".txt")

            img = Image.open(img_path)
            gt_boxes = load_ground_truth_boxes(label_path, *img.size)
            pred_boxes = run_yolo_inference(model, img_path)

            title = (f"{land_use.title()} — {'Good' if label == 'good' else 'Poor'} "
                     f"(AP@0.5 = {example['ap50']:.2f})")
            draw_panel(axes[row, col], img_path, gt_boxes, pred_boxes, title)

    handles = [
        mpatches.Patch(edgecolor="#00CC00", facecolor="none", linewidth=2, label="Ground Truth"),
        mpatches.Patch(edgecolor="red", facecolor="none", linewidth=2, label="Prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Figure 8 — YOLOv8m Detection Examples by Land Use Class", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", OUTPUT_PNG)


def main() -> None:
    """Select examples, run inference, and save the detection example figure."""
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLOv8m weights not found at {YOLO_WEIGHTS}")

    preds = load_predictions()
    land_use_lookup = load_land_use_lookup()
    selections = select_examples(preds, land_use_lookup)

    model = YOLO(str(YOLO_WEIGHTS))
    build_figure(model, selections)


if __name__ == "__main__":
    main()
