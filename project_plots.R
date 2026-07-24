# ============================================================
# results_figures_v2.R
# All 7 models in every figure.
# Overall benchmark: bootstrap_results.csv (all 7 models)
# Conditional: conditional_results.csv (will show all 7 once
#   RetinaNet and DETR are added to conditional_eval.py and rerun)
# Times New Roman, Okabe-Ito, 300 DPI.
# ============================================================

library(tidyverse)
library(extrafont)
if (!requireNamespace("zoo", quietly = TRUE)) {
  install.packages("zoo", repos = "https://cloud.r-project.org")
}
library(zoo)
if (!requireNamespace("ggrepel", quietly = TRUE)) {
  install.packages("ggrepel", repos = "https://cloud.r-project.org")
}
library(ggrepel)
loadfonts(device = "win", quiet = TRUE)

FONT <- "Times New Roman"
DPI  <- 300
OUT <- "figures/"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── Read CSVs from project outputs ──────────────────────────────────────────
boot      <- read_csv("bootstrap_corridor_results.csv",    show_col_types = FALSE)
cond      <- read_csv("conditional_results.csv",  show_col_types = FALSE)
finet     <- read_csv("finetune_results.csv",     show_col_types = FALSE)
pr_curve  <- read_csv("pr_curve_data.csv",        show_col_types = FALSE)
boot_cond <- read_csv("bootstrap_corridor_conditional_results.csv", show_col_types = FALSE)

# ── Clean model names for display ────────────────────────────────────────────
model_labels <- c(
  "yolov8m"      = "YOLOv8m",
  "yolov9m"      = "YOLOv9m",
  "yolov10m"     = "YOLOv10m",
  "yolov11m"     = "YOLOv11m",
  "retinanet"    = "RetinaNet",
  "faster_rcnn"  = "Faster R-CNN",
  "detr"         = "DETR"
)

model_order <- c("YOLOv8m","YOLOv9m","YOLOv10m","YOLOv11m",
                 "RetinaNet","Faster R-CNN","DETR")

paradigm_map <- c(
  "YOLOv8m"      = "Single-stage anchor-free",
  "YOLOv9m"      = "Single-stage anchor-free",
  "YOLOv10m"     = "Single-stage anchor-free",
  "YOLOv11m"     = "Single-stage anchor-free",
  "RetinaNet"    = "Single-stage anchor-based",
  "Faster R-CNN" = "Two-stage",
  "DETR"         = "Transformer"
)

# Okabe-Ito palette
model_colors <- c(
  "YOLOv8m"      = "#E69F00",
  "YOLOv9m"      = "#56B4E9",
  "YOLOv10m"     = "#009E73",
  "YOLOv11m"     = "#0072B2",
  "RetinaNet"    = "#882255",
  "Faster R-CNN" = "#CC79A7",
  "DETR"         = "#D55E00"
)

# Recode model names
boot  <- boot  %>% mutate(model = recode(model, !!!model_labels),
                          model = factor(model, levels = model_order),
                          paradigm = paradigm_map[as.character(model)])
cond  <- cond  %>% mutate(model = recode(model, !!!model_labels),
                          model = factor(model, levels = model_order))
finet <- finet %>% mutate(model = recode(model, !!!model_labels),
                          model = factor(model, levels = model_order))
pr_curve <- pr_curve %>% mutate(model = recode(model, !!!model_labels),
                                model = factor(model, levels = model_order),
                                paradigm = paradigm_map[as.character(model)],
                                paradigm = factor(paradigm, levels = c(
                                  "Single-stage anchor-free",
                                  "Single-stage anchor-based",
                                  "Two-stage",
                                  "Transformer")))
boot_cond <- boot_cond %>% mutate(model = recode(model, !!!model_labels),
                                  model = factor(model, levels = model_order))

# ── Shared theme ─────────────────────────────────────────────────────────────
thesis_theme <- function(base = 12) {
  theme_minimal(base_size = base, base_family = FONT) +
    theme(
      panel.grid.minor    = element_blank(),
      panel.grid.major.x  = element_blank(),
      axis.title          = element_text(size = base,     family = FONT),
      axis.text           = element_text(size = base - 1, family = FONT),
      legend.title        = element_blank(),
      legend.text         = element_text(size = base - 1, family = FONT),
      legend.position     = "bottom",
      strip.text          = element_text(face = "bold",   family = FONT),
      plot.caption        = element_text(size = 8, colour = "grey50",
                                         hjust = 0, family = FONT),
      plot.title          = element_text(size = base + 1, face = "bold",
                                         family = FONT),
      plot.margin         = margin(8, 8, 8, 8)
    )
}


# ============================================================
# FIGURE 1 -- Bootstrapped AP@0.5 all 7 models (PRIMARY BENCHMARK)
# ============================================================
fig1 <- ggplot(boot, aes(x = reorder(model, -ap50_mean),
                         y = ap50_mean * 100, fill = model)) +
  geom_col(width = 0.65, alpha = 0.9) +
  geom_errorbar(aes(ymin = (ap50_mean - 1 * ap50_se) * 100,
                    ymax = (ap50_mean + 1 * ap50_se) * 100),
                width = 0.25, linewidth = 0.5, colour = "black") +
  scale_fill_manual(values = model_colors, breaks = model_order) +
  coord_cartesian(ylim = c(50, 80)) +
  scale_y_continuous(breaks = seq(50, 80, by = 5),
                     labels = function(x) paste0(x, "%")) +
  labs(x = NULL, y = "Mean AP@0.5",
       title = "Figure 1 - Architecture Benchmark: Bootstrapped Per-Image AP@0.5",
       caption = paste0("All 7 architectures across 4 detection paradigms. ",
                        "n = 122 corridors, 461 test images. ",
                        "Error bars = +-1 SE, ",
                        "1000 corridor-level bootstrap iterations, seed 42.")) +
  thesis_theme() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 15, hjust = 1))

ggsave(paste0(OUT, "fig1_benchmark_bootstrap.png"),
       fig1, width = 9, height = 5.5, dpi = DPI)
message("Saved fig1_benchmark_bootstrap.png")


# ============================================================
# FIGURE 2 -- Corpus mAP@0.5 (5 models with Ultralytics/.val())
# ============================================================
fig2 <- ggplot(finet, aes(x = reorder(model, -map50),
                          y = map50 * 100, fill = model)) +
  geom_col(width = 0.65, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f%%", map50 * 100)),
            vjust = -0.4, size = 3.5, family = FONT) +
  scale_fill_manual(values = model_colors, breaks = model_order) +
  coord_cartesian(ylim = c(60, 85)) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(x = NULL, y = "mAP@0.5",
       title = "Figure 2 - Corpus mAP@0.5 (5 Architectures)",
       caption = paste0("DETR and RetinaNet excluded -- corpus mAP@0.5 not available via ",
                        "their evaluation pathway; see Figure 1 for all-7-model comparison. ",
                        "n = 461 test images.")) +
  thesis_theme() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 15, hjust = 1))

ggsave(paste0(OUT, "fig2_corpus_map50.png"),
       fig2, width = 8, height = 5, dpi = DPI)
message("Saved fig2_corpus_map50.png")


# ============================================================
# FIGURE 3 -- Precision-Recall curves by architecture, faceted by
# detection paradigm (2x2 panel layout, 7 models)
# ============================================================

# Raw max-F1 operating point per model (unsmoothed)
max_f1_pts <- pr_curve %>%
  group_by(model) %>%
  slice_max(f1, n = 1, with_ties = FALSE) %>%
  ungroup()

# Smooth precision with a 50-step rolling mean (per model, ordered by
# descending threshold = non-decreasing recall) to reduce jaggedness
pr_smooth <- pr_curve %>%
  arrange(model, desc(threshold)) %>%
  group_by(model) %>%
  mutate(precision_smooth = zoo::rollmean(precision, k = 50,
                                          fill = NA, align = "center")) %>%
  ungroup() %>%
  filter(!is.na(precision_smooth))

# Direct end-of-line labels: rightmost (max-recall) point per model on the
# smoothed curve
label_pts <- pr_smooth %>%
  group_by(model) %>%
  filter(recall == max(recall)) %>%
  slice(1) %>%
  ungroup()

random_line <- data.frame(x = 1, y = 0, xend = 0, yend = 1)

facet_titles <- c(
  "Single-stage anchor-free"  = "Single-Stage Anchor-Free",
  "Single-stage anchor-based" = "Single-Stage Anchor-Based",
  "Two-stage"                 = "Two-Stage",
  "Transformer"                = "Transformer"
)

fig3 <- ggplot(pr_smooth, aes(x = recall, y = precision_smooth, colour = model)) +
  geom_segment(data = random_line, aes(x = x, y = y, xend = xend, yend = yend),
               inherit.aes = FALSE, linetype = "dashed",
               colour = "grey50", linewidth = 0.6) +
  geom_line(linewidth = 1.0) +
  geom_point(data = max_f1_pts, aes(x = recall, y = precision), size = 3) +
  geom_text_repel(data = label_pts, aes(label = model), size = 3,
                  family = FONT, fontface = "bold", show.legend = FALSE,
                  direction = "y", hjust = 0, nudge_x = 0.03,
                  segment.size = 0.3, segment.color = "grey60",
                  min.segment.length = 0, max.overlaps = Inf, seed = 42) +
  scale_colour_manual(values = model_colors, breaks = model_order) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = function(x) paste0(x * 100, "%"),
                     expand = expansion(mult = c(0.01, 0.1))) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                     labels = function(x) paste0(x * 100, "%")) +
  facet_wrap(~paradigm, nrow = 2, ncol = 2, scales = "fixed",
             labeller = labeller(paradigm = facet_titles)) +
  labs(x = "Recall", y = "Precision",
       title = "Figure 3 - Precision-Recall Curves by Architecture",
       caption = paste0("Lines smoothed via 50-threshold-step rolling mean of precision. ",
                        "Points mark the confidence threshold at which each model achieves maximum F1 score (raw, unsmoothed). ",
                        "Dashed line = random classifier baseline. ",
                        "n = 461 test images, 839 ground truth instances.")) +
  thesis_theme() +
  theme(legend.position = "none",
        panel.spacing = unit(1.4, "lines"))

ggsave(paste0(OUT, "fig3_pr_curves.png"),
       fig3, width = 10, height = 8, dpi = DPI)
message("Saved fig3_pr_curves.png")


# ============================================================
# FIGURE 4 -- Voltage class conditional (all models in CSV)
# ============================================================
volt_boot <- boot_cond %>%
  filter(dimension == "voltage_tier") %>%
  select(model, group, ap50_mean, ap50_se)

volt_data <- cond %>%
  filter(dimension == "voltage_tier") %>%
  left_join(volt_boot, by = c("model", "group")) %>%
  mutate(group = factor(group, levels = c("345kV", "500kV")))

n_volt_models <- n_distinct(volt_data$model)
message(paste("Voltage figure: models present =", n_volt_models))

fig4 <- ggplot(volt_data, aes(x = group, y = ap50_mean * 100, fill = model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, alpha = 0.9) +
  geom_errorbar(aes(ymin = (ap50_mean - ap50_se) * 100, ymax = (ap50_mean + ap50_se) * 100),
                width = 0.25, linewidth = 0.5, colour = "black",
                position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = model_colors, breaks = model_order) +
  coord_cartesian(ylim = c(0, 100)) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(x = "Transmission voltage class", y = "mAP@0.5",
       title = "Figure 4 - Performance by Voltage Class",
       caption = paste0("345kV: n = 213 images. 500kV: n = 248 images. ",
                        "Note: voltage class and geographic origin remain partially correlated. ",
                        "Error bars = +-1 SE, corridor-level bootstrap.")) +
  thesis_theme() +
  guides(fill = guide_legend(nrow = 2))

ggsave(paste0(OUT, "fig4_voltage.png"),
       fig4, width = 9, height = 5.5, dpi = DPI)
message("Saved fig4_voltage.png")


# ============================================================
# FIGURE 5 -- Land use conditional (desert excluded, all models in CSV)
# ============================================================
lulc_order <- c("Agriculture", "Forest", "Suburban")

lulc_boot <- boot_cond %>%
  filter(dimension == "land_use") %>%
  select(model, group, ap50_mean, ap50_se)

lulc_data <- cond %>%
  filter(dimension == "land_use",
         tolower(group) != "desert",
         tolower(group) != "shrubland",
         tolower(group) != "other") %>%
  left_join(lulc_boot, by = c("model", "group")) %>%
  mutate(group = str_to_title(group),
         group = factor(group, levels = lulc_order))

n_lulc_models <- n_distinct(lulc_data$model)
message(paste("Land use figure: models present =", n_lulc_models))

fig5 <- ggplot(lulc_data, aes(x = group, y = ap50_mean * 100, fill = model)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, alpha = 0.9) +
  geom_errorbar(aes(ymin = (ap50_mean - ap50_se) * 100, ymax = (ap50_mean + ap50_se) * 100),
                width = 0.25, linewidth = 0.5, colour = "black",
                position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = model_colors, breaks = model_order) +
  coord_cartesian(ylim = c(40, 110)) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(x = "Land use class (NLCD)", y = "mAP@0.5",
       title = "Figure 5 - Performance by Land Use Class",
       caption = paste0("Desert stratum excluded (AZ/NM, n = 114) pending improved corridor selection. ",
                        "Shrubland n = 8 -- interpret with caution. ",
                        "Error bars = +-1 SE, corridor-level bootstrap.")) +
  thesis_theme() +
  guides(fill = guide_legend(nrow = 2))

ggsave(paste0(OUT, "fig5_land_use.png"),
       fig5, width = 9, height = 5.5, dpi = DPI)
message("Saved fig5_land_use.png")


# ============================================================
# FIGURE 6 -- Paradigm comparison: bootstrapped AP50 by paradigm
# (replaces the interaction heatmap until conditional has all 7 models)
# ============================================================
paradigm_data <- boot %>%
  mutate(paradigm = paradigm_map[as.character(model)],
         paradigm = factor(paradigm, levels = c(
           "Single-stage anchor-free",
           "Single-stage anchor-based",
           "Two-stage",
           "Transformer")))

fig6 <- ggplot(paradigm_data,
               aes(x = reorder(model, -ap50_mean),
                   y = ap50_mean * 100,
                   fill = paradigm)) +
  geom_col(width = 0.65, alpha = 0.9) +
  geom_errorbar(aes(ymin = (ap50_mean - 1 * ap50_se) * 100,
                    ymax = (ap50_mean + 1 * ap50_se) * 100),
                width = 0.25, linewidth = 0.5, colour = "black") +
  scale_fill_manual(values = c(
    "Single-stage anchor-free"  = "#56B4E9",
    "Single-stage anchor-based" = "#E69F00",
    "Two-stage"                 = "#CC79A7",
    "Transformer"               = "#D55E00"
  )) +
  coord_cartesian(ylim = c(45, 80)) +
  scale_y_continuous(breaks = seq(45, 80, by = 5),
                     labels = function(x) paste0(x, "%")) +
  labs(x = NULL, y = "Mean AP@0.5",
       title = "Figure 6 - Bootstrapped AP@0.5 by Detection Paradigm",
       caption = paste0("Fill colour encodes detection paradigm. ",
                        "Error bars = +-1 SE. ",
                        "n = 122 corridors, 461 test images.")) +
  thesis_theme() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1)) +
  guides(fill = guide_legend(nrow = 2))

ggsave(paste0(OUT, "fig6_paradigm.png"),
       fig6, width = 9, height = 5.5, dpi = DPI)
message("Saved fig6_paradigm.png")

message("\nAll 6 figures saved to ", OUT)
message("Note: Figs 4 and 5 will automatically include RetinaNet and DETR")
message("once conditional_eval.py is updated and rerun by Claude Code.")


# ============================================================
# FIGURE 7 -- Environment-Level Detection Performance Heatmap
# (4 land cover strata x 7 models)
# ============================================================
env_heat_boot <- boot_cond %>%
  filter(dimension == "land_use", group %in% c("agriculture", "forest", "suburban")) %>%
  select(model, group, ap50_mean)

env_heat_desert <- cond %>%
  filter(dimension == "land_use", group == "desert") %>%
  transmute(model, group, ap50_mean = map50)

env_heat <- bind_rows(env_heat_boot, env_heat_desert) %>%
  mutate(group = str_to_title(group),
         group = factor(group, levels = c("Desert", "Suburban", "Agriculture", "Forest")))

fig7_caption <- str_wrap(paste0(
  "Mean AP@0.5 per land cover stratum per model. Desert stratum excluded ",
  "from primary benchmark pending improved corridor selection. ",
  "Corridor-level bootstrapped values for agriculture, forest, and ",
  "suburban; per-image conditional mAP@0.5 for desert."), width = 100)

fig7 <- ggplot(env_heat, aes(x = model, y = group, fill = ap50_mean)) +
  geom_tile(colour = "white", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.1f%%", ap50_mean * 100)),
            size = 4, colour = "black", family = FONT) +
  scale_fill_gradient2(low = "#D55E00", mid = "#F0F0F0", high = "#0072B2",
                       midpoint = 0.5, limits = c(0, 1)) +
  labs(x = NULL, y = NULL,
       title = "Figure 7 - Detection Performance by Land Cover and Architecture",
       caption = fig7_caption) +
  thesis_theme() +
  theme(legend.position    = "right",
        legend.title       = element_blank(),
        axis.text.x        = element_text(angle = 30, hjust = 1),
        plot.margin        = margin(8, 16, 8, 8))

ggsave(paste0(OUT, "fig7_environment_heatmap.png"),
       fig7, width = 9, height = 5, dpi = DPI)
message("Saved fig7_environment_heatmap.png")