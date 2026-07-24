"""
train_faster_rcnn.py

Trains Faster R-CNN (ResNet-50 FPN, COCO-pretrained) on the NAIP transmission
tower dataset for the two-stage detector leg of the architecture comparison.

Matches YOLO training config where the paradigm allows:
  - Same train/val data (data/augmented/train, data/splits/val)
  - epochs=100 equivalent budget, early stopping patience=20
  - imgsz=640, seed=42, batch=16 (capped lower if GPU memory requires)

Usage:
    conda activate thesis
    python src/training/train_faster_rcnn.py
"""
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG — mirrors train_naip.py where the paradigm allows ────────────────
TRAIN_IMAGES = Path("data/augmented/train/images")
TRAIN_LABELS = Path("data/augmented/train/labels")
VAL_IMAGES   = Path("data/splits/val/images")
VAL_LABELS   = Path("data/splits/val/labels")

OUTPUT_DIR   = Path("outputs/weights/naip_faster_rcnn")
IMG_SIZE     = 640
BATCH_SIZE   = 8          # lower than YOLO's 16 — two-stage detectors are heavier per-sample
EPOCHS       = 100
PATIENCE     = 20
LR           = 0.0001
WEIGHT_DECAY = 0.0005
SEED         = 42
NUM_WORKERS  = 4
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torchvision detection models reserve class 0 for background internally.
# YOLO class 0 (tower) maps to torchvision class 1; background is implicit at 0.
NUM_CLASSES  = 2  # background + tower
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TowerDataset(Dataset):
    """Loads YOLO-format labels and converts to torchvision detection format."""

    def __init__(self, image_dir: Path, label_dir: Path):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(image_dir.glob("*.jpg"))
        if not self.images:
            self.images = sorted(image_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = (
                    int(float(parts[0])), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                )
                # Only tower (YOLO class 0) — map to torchvision class 1
                if cls_id != 0:
                    continue
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # torchvision class 1 = tower, 0 = background

        boxes_t  = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)  if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    """COCO-pretrained Faster R-CNN, head replaced for 2-class (bg + tower) output."""
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def evaluate_loss(model, val_loader):
    """Quick val loss for early stopping — full mAP eval happens separately."""
    model.train()  # torchvision detection models need train() mode to return losses
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss for loss in loss_dict.values()).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Device: {DEVICE}")
    log.info(f"Train images: {TRAIN_IMAGES}")
    log.info(f"Val images:   {VAL_IMAGES}")

    train_ds = TowerDataset(TRAIN_IMAGES, TRAIN_LABELS)
    val_ds   = TowerDataset(VAL_IMAGES, VAL_LABELS)
    log.info(f"Train samples: {len(train_ds)}")
    log.info(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, targets in pbar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader)

        log.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
            log.info(f"  New best val_loss — saved checkpoint")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch} (patience={PATIENCE} exceeded)")
            break

    torch.save(model.state_dict(), OUTPUT_DIR / "last.pt")
    log.info(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    log.info(f"Weights saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
