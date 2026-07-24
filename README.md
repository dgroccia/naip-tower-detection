# Deep Learning Detection of Electrical Transmission Towers from NAIP Aerial Imagery

---

## Project Overview

This project benchmarks seven deep learning object detection architectures across four detection paradigms for automated detection of high-voltage electrical transmission towers from National Agriculture Imagery Program (NAIP) aerial imagery at 0.6m ground sampling distance. The work addresses the generalization gap between existing drone-based detection methods and nationally available aerial imagery sources.

**Three research questions:**
1. How does fine-tuned detection performance vary across single-stage, transformer-based, and two-stage architectures at sub-meter aerial resolution?
2. To what extent does tower detectability vary with physical size as a function of voltage class?
3. How does the best-performing model's output compare to OSM-derived tower inventory at state scale? (Virginia pilot — pipeline built, inference not yet run)

---

## Dataset Summary

| Split | Images | Instances | States |
|---|---|---|---|
| Train (base) | 288 | 418 | TX, TN, WV, GA, NY |
| Train (augmented) | 9,670 | ~14,000 | same |
| Validation | 67 | 87 | TX, TN, WV |
| Test (expanded) | 461 | 839 | TX, TN, WV, GA, NY, AZ, NM |

**Single class:** `tower` = class index **0** throughout. This is critical — CVAT single-class task exports use index 0 but the raw export files sometimes contain class 1. Always verify and fix with `sed -i 's/^1 /0 /' *.txt` before integrating new annotations.

**Voltage classes:** 345kV and 500kV  
**Land use:** agriculture, forest, suburban (desert excluded from primary benchmark)  
**Metadata:** `data/collected/annotated/tower_metadata.csv` — 798 rows, fully NLCD-verified

---

## Model Benchmark

| Model | Paradigm | Weights Path |
|---|---|---|
| YOLOv8m | Single-stage anchor-free | `outputs/weights/naip_yolov8m_coco/weights/best.pt` |
| YOLOv9m | Single-stage anchor-free | `outputs/weights/naip_yolov9m_coco/weights/best.pt` |
| YOLOv10m | Single-stage anchor-free | `outputs/weights/naip_yolov10m_coco/weights/best.pt` |
| YOLOv11m | Single-stage anchor-free | `outputs/weights/naip_yolov11m_coco/weights/best.pt` |
| RetinaNet | Single-stage anchor-based | `outputs/weights/naip_retinanet/best.pt` |
| Faster R-CNN | Two-stage | `outputs/weights/naip_faster_rcnn/best.pt` |
| DETR | Transformer | `outputs/weights/naip_detr/best/` (HuggingFace format) |

**RT-DETR was excluded:** Persistent NaN loss from deformable attention kernels on RTX 5060 Ti sm_120 Blackwell GPU. Not a configuration problem — a hardware/PyTorch nightly compatibility issue. See Hardware section.

---

## Key Results

**Primary metric: corridor-level bootstrapped AP@0.5** (1,000 iterations, 122 corridors, seed 42)

| Model | Boot AP@0.5 | SE | Corpus mAP@0.5 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Faster R-CNN | 0.701 | 0.065 | 0.729 | 0.842 | 0.751 | 0.794 |
| YOLOv8m | 0.677 | 0.064 | 0.776 | 0.857 | 0.716 | 0.781 |
| RetinaNet | 0.672 | 0.063 | - | 0.857 | 0.737 | 0.792 |
| YOLOv9m | 0.663 | 0.066 | 0.759 | 0.852 | 0.721 | 0.781 |
| YOLOv10m | 0.654 | 0.063 | 0.731 | 0.818 | 0.681 | 0.743 |
| YOLOv11m | 0.641 | 0.064 | 0.766 | 0.858 | 0.683 | 0.760 |
| DETR | 0.565 | 0.060 | - | 0.708 | 0.561 | 0.626 |

Precision/recall reported at max-F1 confidence threshold per model.

---

## Environment Setup

### Hardware
- **GPU:** NVIDIA RTX 5060 Ti (sm_120 Blackwell architecture)
- **CRITICAL:** This GPU requires the PyTorch nightly cu128 build. Stable PyTorch releases do not support sm_120 and will silently fall back to degraded operation.
- **OS:** Windows 11 with WSL2 Ubuntu 24.04

### Software Installation

```bash
# 1. Create conda environment
conda create -n thesis python=3.10 -y
conda activate thesis

# 2. Install PyTorch nightly (EXACT VERSION REQUIRED for RTX 5060 Ti)
pip install torch==2.10.0.dev20251204+cu128 torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install Albumentations (PINNED - do not upgrade)
pip install albumentations==1.4.18

# 4. Install remaining dependencies
pip install ultralytics transformers==5.3.0 \
    geopandas rasterio pyproj pystac-client \
    planetary-computer torchmetrics pyserial \
    pandas numpy opencv-python pillow tqdm

# 5. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 5060 Ti
```

### R Setup (for figures)
```r
install.packages(c("tidyverse", "extrafont", "zoo", "patchwork"))
library(extrafont)
font_import()  # run once to import Times New Roman
loadfonts(device = "win")
```

---

## Project Structure

```
thesis_infrastructure_detection/
├── configs/
│   ├── dataset.yaml          # train/val/test paths, nc=1, names={0: tower}
│   └── augmentation.yaml     # rotation_step_degrees: 10
├── data/
│   ├── splits/               # train/val/test image and label splits
│   ├── augmented/            # augmented training set (9,670 images)
│   ├── naip/                 # raw and clipped NAIP tiles
│   ├── nlcd_*.tif            # NLCD rasters per state
│   └── collected/
│       └── annotated/
│           └── tower_metadata.csv
├── src/
│   ├── data/                 # download, clip, tile, augment scripts
│   ├── training/             # one training script per model
│   └── evaluation/           # inference, bootstrap, conditional eval
├── outputs/
│   ├── weights/              # trained model checkpoints
│   └── results/              # all evaluation CSVs
└── project_plots.R           # all 7 figures (run from RStudio)
```

---

## Evaluation Pipeline — Run in This Order

```bash
conda activate thesis
cd ~/projects/thesis_infrastructure_detection

# 1. Per-image inference (all 7 models, GPU required, ~90 seconds)
python src/evaluation/per_image_inference.py

# 2. Corpus mAP (5 YOLO/FRCNN models, GPU required)
python src/evaluation/finetune_eval.py

# 3. Overall corridor bootstrap (CPU, <1 minute)
python src/evaluation/bootstrap_corridor.py

# 4. Conditional eval — voltage and land use (GPU, ~15 minutes)
python src/evaluation/conditional_eval.py

# 5. Conditional corridor bootstrap (CPU, <1 minute)
python src/evaluation/bootstrap_corridor_conditional.py

# 6. Raw predictions for PR curves (GPU, ~90 seconds)
python src/evaluation/collect_raw_predictions.py

# 7. PR curve data (CPU, fast)
python src/evaluation/plot_pr_curves.py
```

Results land in `outputs/results/`. Copy CSVs to RStudio working directory and run `project_plots.R` to regenerate all figures.

---

## Data Acquisition Pipeline (for new corridors)

```bash
# 1. Draw corridor polygons in ArcPro, export as shapefile
# 2. Download NAIP tiles
python src/data/download_naip_{state}.py

# 3. Filter overlapping tiles (manual bounding box check)
# 4. Clip to corridor polygons
python src/data/clip_naip.py

# 5. Tile to 640x640 patches
python src/data/slice_naip_tiles.py \
  --input_dir data/naip/clipped_{state} \
  --output_dir data/naip/patches/{STATE}_corridor \
  --patch_size 640 --overlap 0.1 --min_valid 0.1

# 6. Zip images and upload to CVAT for annotation
# CVAT settings: single class 'tower', overlap=1
# CRITICAL: After export, verify class index = 0, not 1
# Fix if needed: find obj_train_data -name "*.txt" -exec sed -i 's/^1 /0 /' {} \;

# 7. Integrate annotations
python src/data/integrate_{state}_metadata.py
```

---

## Critical Conventions and Known Gotchas

### Class Index
Tower is **always class 0**. CVAT single-class tasks sometimes export class 1. Every time you integrate new annotations, verify with:
```bash
find data/splits/test/labels -name "*.txt" | xargs grep -l "^1 " | wc -l
# Should be 0. If not, fix: sed -i 's/^1 /0 /' each file
```

### DETR Loading
DETR weights are saved in HuggingFace `save_pretrained` format at `outputs/weights/naip_detr/best/`. Load with:
```python
from transformers import DetrForObjectDetection, DetrImageProcessor
model = DetrForObjectDetection.from_pretrained('outputs/weights/naip_detr/best/')
processor = DetrImageProcessor.from_pretrained('outputs/weights/naip_detr/best/')
```
DETR's no-object class is **index 2** (not 0), confirmed empirically. Filter predictions where `argmax(logits) == 2`.

### Bootstrapping Methodology
Use **corridor-level** bootstrap, not image-level. Images within a corridor are spatially correlated. The corridor identifier is extracted from the filename by stripping the `_rXXX_cXXX.jpg` suffix. See `src/evaluation/bootstrap_corridor.py`.

### Augmentation
Albumentations must be **pinned to 1.4.18**. Do not upgrade. The augmentation pipeline runs offline before training:
```bash
python src/data/augmentation_naip.py
```
Output: 9,670 images in `data/augmented/train/`. Min visibility threshold 0.3 drops annotations that become too small after rotation.

### RT-DETR Exclusion
RT-DETR-L was attempted four times with different configurations. All attempts produced NaN loss from epoch 1. Root cause: deformable attention CUDA kernels are unstable in PyTorch nightly cu128 on sm_120 Blackwell. Standard DETR (this repo) uses regular self-attention and works fine. Do not attempt RT-DETR on this hardware until stable PyTorch Blackwell support is released.

---

## Outstanding Tasks

**Critical for submission:**
- [ ] Write discussion section (structure: RQ1 architecture comparison → RQ2 voltage → RQ3 Virginia → limitations → future work)
- [ ] Write conclusion
- [ ] Write glossary (~30 terms, flagged by Matt Rice at thesis defense)
- [ ] Run Virginia inference pipeline (scripts exist, baseline exists, inference not yet run)
- [ ] Add RetinaNet and DETR to `finetune_eval.py` for complete precision/recall via corpus eval

**Virginia Inference Pipeline (RQ3):**
- OSM baseline: 10,873 towers associated with 500kV+ lines in Virginia
- HIFLD lines: `data/collected/osm_va_hv_lines.geojson`
- Next step: draw corridor polygons around Virginia 500kV+ lines in ArcPro, download NAIP tiles, run best model inference, compare detection count to OSM baseline
- Recommended model for this task: Faster R-CNN (highest recall, minimizes missed towers)

**Desert evaluation (future work, not blocking):**
- AZ and NM corridors showed near-zero performance (1-12% mAP)
- Likely causes: poor corridor selection, high-albedo terrain, annotation sparsity
- Recommended fix: identify corridors where towers cast clear shadows, use shadow geometry for annotation
- Longer term: multispectral or NIR imagery would provide better contrast in desert terrain

