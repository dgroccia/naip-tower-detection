# Project Specification Document
## Deep Learning Detection of Electrical Transmission Towers from NAIP Imagery
### Version: July 2026 | Author: Dante Groccia | Handoff Document

---

## 1. Research Questions

1. How does fine-tuned detection performance vary across single-stage, transformer-based, and two-stage architectures at sub-meter aerial resolution?
2. To what extent does tower detectability vary with physical size as a function of voltage class, once geographic confounding is resolved?
3. How does the best-performing model's output compare to OSM-derived inventory at state scale? (Virginia pilot)

---

## 2. Dataset

### 2.1 Annotation Corpus (Training + Original Test)
- **677 instances** across **452 images** from 5 states: TX, TN, WV, GA, NY
- Single class: `tower` = class index **0**
- Two voltage classes: 345kV and 500kV
- NAIP imagery at 0.6m GSD, RGB+NIR (4-band), patches at 640x640 pixels
- Annotated in CVAT using YOLO bounding box format

### 2.2 Splits
| Split | Images | Instances | Notes |
|---|---|---|---|
| Train | 288 | 418 | Augmented to 9,670 for training |
| Val | 67 | 87 | Unaugmented, used during training |
| Test (original) | 97 | 152 | TX, TN, WV corridors |
| Test (expanded) | 461 | 839 | Added NY, WV-new, AZ, NM after training |

**Important:** Test set was expanded after all model training was complete. No test images were seen during training or model selection. The expansion was done to increase geographic evaluation depth.

### 2.3 Land Use Scheme (NLCD-derived)
| Category | NLCD Classes Included | Test Images |
|---|---|---|
| Agriculture | Pasture, cultivated crops, grassland | 57 |
| Forest | Deciduous, evergreen, mixed forest, woody wetlands | 126 |
| Suburban | Developed open, low, medium, high | 163 |
| Desert | AZ/NM corridors (excluded from primary benchmark) | 114 |

### 2.4 Metadata
- File: `data/collected/annotated/tower_metadata.csv`
- 798 rows (one per instance), fully NLCD-verified including GA and NY
- Key columns: filename, state, split, lon, lat, nlcd_code, nlcd_class, land_use, voltage_tier, dist_m

### 2.5 Augmentation Pipeline
- Library: **Albumentations 1.4.18** (pinned — do not upgrade)
- Config: `configs/augmentation.yaml`, rotation_step_degrees: 10
- 4 tiers: rotation (36 steps, 0-350 degrees), photometric, noise/occlusion, online (YOLO only)
- Output: 9,670 training images (33.6x expansion from 288 base)
- Min visibility threshold: 0.3 (annotations dropped if box becomes too small after rotation)
- Script: `src/data/augmentation_naip.py`

---

## 3. Model Specifications

### 3.1 Architecture Overview
All models initialized from COCO pretrained weights. Single class output (tower).

| Model | Paradigm | Parameters | Source | Weights Format |
|---|---|---|---|---|
| YOLOv8m | Single-stage anchor-free | 25.9M | Ultralytics | `.pt` |
| YOLOv9m | Single-stage anchor-free | 20.0M | Ultralytics | `.pt` |
| YOLOv10m | Single-stage anchor-free | 15.4M | Ultralytics | `.pt` |
| YOLOv11m | Single-stage anchor-free | 20.1M | Ultralytics | `.pt` |
| RetinaNet | Single-stage anchor-based | 36.4M | torchvision | `.pt` state dict |
| Faster R-CNN | Two-stage | 43.7M | torchvision | `.pt` state dict |
| DETR | Transformer | 41.3M | HuggingFace | `save_pretrained` dir |

### 3.2 Training Configuration
| Parameter | YOLO models | RetinaNet | Faster R-CNN | DETR |
|---|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Learning rate | 0.0001 | 0.0001 | 0.0001 | 0.0001 (head), 0.00001 (backbone) |
| Epochs | 100 | 100 | 100 | 100 |
| Patience | 20 | 20 | 20 | 30 |
| Batch size | 16 | 8 | 8 | 4 |
| Image size | 640 | 640 | 640 | 800 (processor default) |
| Weight decay | 0.0005 | 0.0005 | 0.0005 | 0.0001 |
| Seed | 42 | 42 | 42 | 42 |

### 3.3 Special Notes Per Model

**RetinaNet:**
- Classification head replaced: `RetinaNetClassificationHead(in_channels=256, num_anchors=9, num_classes=2, norm_layer=partial(GroupNorm, 32))`
- num_classes=2 because torchvision convention: 0=background, 1=tower
- Early stopped at epoch 2 — val_loss diverged after epoch 2 but best checkpoint is still competitive

**Faster R-CNN:**
- Uses `fasterrcnn_resnet50_fpn_v2` with `FastRCNNPredictor` head
- num_classes=2 (background + tower)
- Precision reported at fixed threshold 0.25 in finetune_eval.py — lower than YOLO's swept threshold. mAP@0.5 is unaffected. See evaluation notes.

**DETR:**
- Loaded via HuggingFace: `DetrForObjectDetection.from_pretrained('outputs/weights/naip_detr/best/')`
- No-object class is index **2** (not 0) — confirmed empirically. Filter `argmax(logits) == 2` to remove background predictions
- Trained for 67 epochs before early stopping (patience=30)
- Known limitation: hard recall ceiling at ~56%, suggesting query underfitting within 67-epoch budget

**RT-DETR (excluded):**
- Attempted 4 times with different configurations
- All attempts: NaN loss from epoch 1 due to deformable attention kernel instability on sm_120 Blackwell GPU
- Not a configuration problem. Do not retry until stable PyTorch Blackwell support releases.

---

## 4. Evaluation Methodology

### 4.1 Primary Metric: Corridor-Level Bootstrapped AP@0.5
- **Why corridor-level:** Individual patches within a corridor share terrain, acquisition date, lighting, and tower density. Resampling patches treats correlated observations as independent, inflating confidence. Resampling corridors treats each geographic segment as one observation.
- **Implementation:** Extract corridor ID by stripping `_rXXX_cXXX.jpg` from filename stem. Bootstrap by drawing 122 corridors with replacement, pool all patch AP@0.5 values from drawn corridors, take mean. Repeat 1,000 times. Report mean and SD of 1,000 means.
- **122 corridors** from 461 test images (~3.8 images per corridor average)
- **Script:** `src/evaluation/bootstrap_corridor.py`
- **Output:** `outputs/results/bootstrap_corridor_results.csv`

### 4.2 Secondary Metrics
- **Corpus mAP@0.5:** Standard COCO-style pooled evaluation. Available for 5 models (YOLO family + Faster R-CNN) via Ultralytics `.val()` and torchmetrics. Not available for DETR and RetinaNet via this pathway.
- **Precision/Recall/F1:** Reported at max-F1 confidence threshold per model from PR curve analysis. Available for all 7 models via `src/evaluation/collect_raw_predictions.py` + `src/evaluation/plot_pr_curves.py`
- **Conditional mAP@0.5:** Per voltage class and land use stratum. Script: `src/evaluation/conditional_eval.py`
- **Conditional corridor bootstrap:** SE per stratum. Script: `src/evaluation/bootstrap_corridor_conditional.py`

### 4.3 Evaluation Run Order
```
per_image_inference.py → finetune_eval.py → bootstrap_corridor.py →
conditional_eval.py → bootstrap_corridor_conditional.py →
collect_raw_predictions.py → plot_pr_curves.py
```

---

## 5. Key Decisions and Rationale

| Decision | Rationale |
|---|---|
| Single class (tower only) | Cable class was annotated early but dropped — too visually ambiguous, insufficient instances |
| COCO initialization (not TTPLA) | TTPLA pretraining produced worse results than COCO direct fine-tune |
| Corridor-level bootstrap over image-level | Spatial autocorrelation within corridors violates image-level independence assumption |
| Desert excluded from primary benchmark | Near-zero performance caused by corridor selection and annotation issues, not model failure |
| Shrubland excluded from figures | Only 8 test images, small sample artifact including apparent 1.000 mAP for YOLOv11m |
| Augmentation Albumentations pinned at 1.4.18 | Later versions have API changes that break the pipeline |
| RetinaNet GroupNorm in classification head | Recommended for fine-tuning on small datasets; BatchNorm performs poorly |
| DETR patience=30 (not 20) | DETR converges slowly; 20 epochs was insufficient to allow meaningful learning |

---

## 6. File Inventory

### Scripts (`src/`)
```
src/data/
├── augmentation_naip.py          # 4-tier augmentation pipeline
├── slice_naip_tiles.py           # clip GeoTIFFs to 640x640 patches
├── download_naip_*.py            # state-specific NAIP downloaders
├── integrate_*_metadata.py       # add new corridors to metadata CSV
├── augment_test_set.py           # geometric augmentation of test set (deprecated)
├── fill_nlcd_pending.py          # NLCD API query (unreliable, use ArcPro instead)
├── propagate_test_metadata.py    # copy metadata to augmented test images
└── osm_va_*.py                   # Virginia OSM tower and line extraction

src/training/
├── train_retinanet.py
├── train_faster_rcnn.py
└── train_detr.py
# YOLO models trained via ultralytics CLI / train_naip.py

src/evaluation/
├── per_image_inference.py        # inference on all 7 models, produces per_image_predictions.csv
├── finetune_eval.py              # corpus mAP for 5 models
├── bootstrap_corridor.py         # overall corridor bootstrap
├── conditional_eval.py           # conditional mAP by voltage and land use
├── bootstrap_corridor_conditional.py  # corridor bootstrap per stratum
├── collect_raw_predictions.py    # raw detections for PR curves
└── plot_pr_curves.py             # PR curve data from raw predictions
```

### Results CSVs (`outputs/results/`)
| File | Contents | Rows |
|---|---|---|
| `per_image_predictions.csv` | AP50, precision, recall per image per model | 3,227 |
| `finetune_results.csv` | Corpus mAP, precision, recall for 5 models | 5 |
| `bootstrap_corridor_results.csv` | Corridor bootstrap mean and SE, all 7 models | 7 |
| `conditional_results.csv` | mAP by dimension/group/model | 98 |
| `bootstrap_corridor_conditional_results.csv` | Corridor bootstrap SE per stratum | 35 |
| `raw_predictions.csv` | Per-detection confidence and TP/FP status | 19,849 |
| `pr_curve_data.csv` | Precision/recall/F1 at each threshold per model | 17,765 |
| `corridor_heatmap_labeled.csv` | Per-corridor AP50, filtered to 56 intermediate corridors | 56 |

---

## 7. Virginia Inference Pipeline (RQ3 — Incomplete)

**What exists:**
- OSM tower baseline: `data/collected/osm_va_hv_towers.geojson` — 10,873 towers on 345kV+ lines
- OSM line geometry: `data/collected/osm_va_hv_lines.geojson`
- HIFLD transmission line layer was used for corridor delineation

**What needs to be done:**
1. Draw corridor polygons in ArcPro around Virginia 345kV+ transmission lines
2. Download NAIP tiles via Planetary Computer (same pipeline as other states)
3. Clip and tile to 640x640 patches
4. Run Faster R-CNN inference (highest recall, best for counting tasks)
5. Convert bounding box detections to geographic coordinates using tile geotransforms
6. Compare detection count to 10,873 OSM baseline
7. Produce a map figure showing detected towers vs OSM towers in the same corridor

**Recommended approach:** Start with a 20-30km pilot corridor rather than full state, verify detections look real, then scale up.

---

## 8. Outstanding Tasks for Next Researcher

### Critical for submission
- [ ] Discussion section (structure below)
- [ ] Conclusion
- [ ] Glossary (~30 terms, flagged by Matt Rice at thesis defense)
- [ ] Virginia inference pipeline (RQ3)
- [ ] Add RetinaNet and DETR to finetune_eval.py

### Discussion section structure
1. RQ1: Architecture comparison — Faster R-CNN leads on recall and corridor bootstrap, YOLO leads on corpus mAP, paradigm matters more in hard environments (suburban) than easy ones (forest)
2. RQ2: Voltage class — 500kV consistently more detectable, geographic confound acknowledged, RetinaNet smallest gap suggests anchor-scale encoding helps for smaller structures
3. RQ3: Virginia pilot (fill in once inference is run)
4. Land use findings — forest solved, suburban discriminative, desert boundary condition
5. DETR — slow convergence interpretation, hard recall ceiling finding
6. Limitations — training budget, desert corridor selection, voltage-geography confound in test set
7. Future work — desert multispectral imagery, cross-validation design, transformer training at scale, Virginia full-state inference

### Paper target
- Primary: IEEE JSTARS
- Secondary: Remote Sensing (MDPI)
- Template: IEEE JSTARS Word template (in project folder)

---

## 9. Contact and Continuity

**Original researcher:** Dante Groccia (dgroccia@gmu.edu)  
**Advisor:** Dr. Edward Oughton (eoughton@gmu.edu)  
**Lab:** Infrastructure and Climate Risk Lab, George Mason University  

This project was developed over approximately 8 months. The full conversation history including all debugging, design decisions, and pipeline development is archived in transcript files. Key decisions are summarized in this document but the transcripts contain the full rationale for edge cases.
