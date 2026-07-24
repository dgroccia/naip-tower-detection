"""
train_detr.py

Trains DETR (Detection Transformer, ResNet-50 backbone, COCO-pretrained)
on the NAIP transmission tower dataset. DETR uses standard multi-head
self-attention rather than deformable attention, confirmed stable on
RTX 5060 Ti sm_120 via forward pass smoke test.

Key differences from RT-DETR:
  - Standard (not deformable) attention -- no sm_120 instability
  - HuggingFace transformers implementation
  - DETR natively predicts a fixed set of N queries (default 100);
    only predictions assigned to tower class are retained
  - Trained longer (patience=30) since DETR is known to converge slowly

Usage:
    conda activate thesis
    python src/training/train_detr.py
"""
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_IMAGES  = Path("data/augmented/train/images")
TRAIN_LABELS  = Path("data/augmented/train/labels")
VAL_IMAGES    = Path("data/splits/val/images")
VAL_LABELS    = Path("data/splits/val/labels")

OUTPUT_DIR    = Path("outputs/weights/naip_detr")
BATCH_SIZE    = 4     # DETR is heavier per sample than Faster R-CNN
EPOCHS        = 100
PATIENCE      = 30    # DETR converges slowly; give it more patience
LR            = 0.0001
LR_BACKBONE   = 0.00001  # lower LR for pretrained backbone, standard DETR practice
WEIGHT_DECAY  = 0.0001
SEED          = 42
NUM_WORKERS   = 2
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DETR class mapping:
# DETR's COCO head has 92 classes (0=N/A, 1-91=COCO classes).
# We replace the head so our classes are:
#   0 = no object (background / N/A)
#   1 = tower
NUM_CLASSES   = 2
MODEL_ID      = "facebook/detr-resnet-50"
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TowerDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.images = sorted(image_dir.glob("*.jpg"))
        if not self.images:
            self.images = sorted(image_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path   = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id != 0:   # YOLO class 0 = tower
                    continue
                cx, cy, bw, bh = map(float, parts[1:])
                # Convert YOLO normalized xywh -> absolute xyxy for DETR
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)   # tower = class 1 in our 2-class scheme

        # DETR processor expects pascal_voc format boxes and handles resizing
        target = {
            "image_id":    idx,
            "annotations": [
                {
                    "image_id":    idx,
                    "category_id": labels[i],
                    "bbox":        [boxes[i][0], boxes[i][1],
                                    boxes[i][2] - boxes[i][0],
                                    boxes[i][3] - boxes[i][1]],  # xywh absolute
                    "area":        (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1]),
                    "iscrowd":     0,
                    "id":          i,
                }
                for i in range(len(boxes))
            ],
        }

        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        encoding = {k: v.squeeze(0) if hasattr(v, "squeeze") else v for k, v in encoding.items()}
        return encoding


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [{"class_labels": b["labels"][0]["class_labels"],
                "boxes":        b["labels"][0]["boxes"]} for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def evaluate_loss(model, val_loader):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {DEVICE}")
    log.info(f"Model: {MODEL_ID}, NUM_CLASSES={NUM_CLASSES}")

    processor = DetrImageProcessor.from_pretrained(MODEL_ID)

    train_ds = TowerDataset(TRAIN_IMAGES, TRAIN_LABELS, processor)
    val_ds   = TowerDataset(VAL_IMAGES,   VAL_LABELS,   processor)
    log.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Load COCO pretrained DETR and replace classification head for 2 classes
    model = DetrForObjectDetection.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,   # replaces the 92-class COCO head
    )
    model = model.to(DEVICE)

    # Standard DETR fine-tuning: lower LR on backbone, higher on transformer head
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": LR_BACKBONE},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    best_val_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
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
            model.save_pretrained(OUTPUT_DIR / "best")
            processor.save_pretrained(OUTPUT_DIR / "best")
            log.info("  New best val_loss — saved checkpoint")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch}")
            break

    model.save_pretrained(OUTPUT_DIR / "last")
    log.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    log.info(f"Weights: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
