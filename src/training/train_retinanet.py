"""
train_retinanet.py

Trains RetinaNet (ResNet-50 FPN v2, COCO-pretrained) on the NAIP
transmission tower dataset. RetinaNet is a single-stage anchor-based
detector, providing a paradigm contrast to the anchor-free YOLO family
within the single-stage category.

Matches training configuration across all torchvision models:
  - Same train/val data (data/augmented/train, data/splits/val)
  - epochs=100, patience=20, batch=8, seed=42, AdamW lr=0.0001

Usage:
    conda activate thesis
    python src/training/train_retinanet.py
"""
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from PIL import Image
from tqdm import tqdm
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_IMAGES = Path("data/augmented/train/images")
TRAIN_LABELS = Path("data/augmented/train/labels")
VAL_IMAGES   = Path("data/splits/val/images")
VAL_LABELS   = Path("data/splits/val/labels")

OUTPUT_DIR   = Path("outputs/weights/naip_retinanet")
IMG_SIZE     = 640
BATCH_SIZE   = 8
EPOCHS       = 100
PATIENCE     = 20
LR           = 0.0001
WEIGHT_DECAY = 0.0005
SEED         = 42
NUM_WORKERS  = 4
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 2  # background + tower (torchvision convention)
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TowerDataset(Dataset):
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
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                # YOLO class 0 (tower) maps to torchvision class 1 (background=0)
                if cls_id != 0:
                    continue
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)

        boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
    # Replace classification head for 2-class output (background + tower)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )
    return model


def evaluate_loss(model, val_loader):
    model.train()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss for loss in loss_dict.values()).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {DEVICE}")

    train_ds = TowerDataset(TRAIN_IMAGES, TRAIN_LABELS)
    val_ds   = TowerDataset(VAL_IMAGES, VAL_LABELS)
    log.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss    = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, targets in pbar:
            images  = [img.to(DEVICE) for img in images]
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
            best_val_loss      = val_loss
            epochs_no_improve  = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
            log.info("  New best val_loss — saved checkpoint")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), OUTPUT_DIR / "last.pt")
    log.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    log.info(f"Weights: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
