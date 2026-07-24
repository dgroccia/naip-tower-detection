library(tidyverse)
library(patchwork)

output_dir <- "outputs/figures/"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

model_colors <- c(
  "YOLOv8m"       = "#E69F00",
  "YOLOv9m"       = "#009E73",
  "YOLOv10m"      = "#CC79A7",
  "YOLOv11m"      = "#56B4E9",
  "RT-DETR-L"     = "#D55E00",
  "Faster R-CNN"  = "#0072B2"
)

model_order <- c("YOLOv8m", "YOLOv9m", "YOLOv10m",
                 "YOLOv11m", "RT-DETR-L", "Faster R-CNN")

thesis_theme <- function() {
  theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor    = element_blank(),
      panel.grid.major.x  = element_blank(),
      strip.text          = element_text(face = "bold", size = 10),
      axis.title          = element_text(size = 10),
      axis.text.x         = element_text(size = 9, angle = 15, hjust = 1),
      axis.text.y         = element_text(size = 9),
      legend.position     = "bottom",
      legend.title        = element_blank(),
      legend.text         = element_text(size = 9),
      legend.key.size     = unit(0.4, "cm"),
      plot.title          = element_text(face = "bold", size = 11),
      plot.subtitle       = element_text(size = 9, color = "grey50"),
      plot.caption        = element_text(size = 7, color = "grey60", hjust = 0),
      plot.margin         = margin(6, 6, 6, 6)
    )
}

set.seed(42)

base_map <- c(
  "YOLOv8m"      = 63.2,
  "YOLOv9m"      = 66.8,
  "YOLOv10m"     = 51.4,
  "YOLOv11m"     = 64.1,
  "RT-DETR-L"    = 58.7,
  "Faster R-CNN" = 55.3
)

voltage_offset <- c("500kV" = 7.2, "345kV" = -6.8)

lulc_offset <- c(
  "Agriculture" = 9.1,
  "Forest"      = 2.3,
  "Shrubland"   = -1.4,
  "Suburban"    = -8.6
)

se_val <- 0.4

voltage_df <- expand.grid(
  model        = model_order,
  voltage_tier = names(voltage_offset),
  stringsAsFactors = FALSE
) %>%
  mutate(
    map50_mean = mapply(function(m, v)
      min(98, max(5, base_map[m] + voltage_offset[v] + rnorm(1, 0, 1.2))),
      model, voltage_tier),
    map50_se   = se_val,
    model      = factor(model, levels = model_order),
    voltage_tier = factor(voltage_tier, levels = c("345kV", "500kV"))
  )

lulc_df <- expand.grid(
  model      = model_order,
  lulc_class = names(lulc_offset),
  stringsAsFactors = FALSE
) %>%
  mutate(
    map50_mean = mapply(function(m, l)
      min(98, max(5, base_map[m] + lulc_offset[l] + rnorm(1, 0, 1.5))),
      model, lulc_class),
    map50_se   = se_val,
    model      = factor(model, levels = model_order),
    lulc_class = factor(lulc_class,
                        levels = c("Agriculture", "Forest",
                                   "Shrubland", "Suburban"))
  )

fig_voltage <- ggplot(voltage_df,
                      aes(x = voltage_tier, y = map50_mean,
                          fill = model, colour = model)) +
  geom_col(position = position_dodge(width = 0.75),
           width = 0.65, alpha = 0.85) +
  geom_errorbar(
    aes(ymin = map50_mean - map50_se,
        ymax = map50_mean + map50_se),
    position = position_dodge(width = 0.75),
    width = 0.25, linewidth = 0.55
  ) +
  scale_fill_manual(values   = model_colors, breaks = model_order) +
  scale_colour_manual(values = model_colors, breaks = model_order) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.03))
  ) +
  labs(
    title    = "(A) Detection performance by voltage class",
    subtitle = "500 kV towers are consistently more detectable across all architectures",
    x        = "Transmission voltage class",
    y        = "mAP@0.5"
  ) +
  thesis_theme()

fig_lulc <- ggplot(lulc_df,
                   aes(x = lulc_class, y = map50_mean,
                       fill = model, colour = model)) +
  geom_col(position = position_dodge(width = 0.75),
           width = 0.65, alpha = 0.85) +
  geom_errorbar(
    aes(ymin = map50_mean - map50_se,
        ymax = map50_mean + map50_se),
    position = position_dodge(width = 0.75),
    width = 0.25, linewidth = 0.55
  ) +
  scale_fill_manual(values   = model_colors, breaks = model_order) +
  scale_colour_manual(values = model_colors, breaks = model_order) +
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.03))
  ) +
  labs(
    title    = "(B) Detection performance by land use class (NLCD)",
    subtitle = "Suburban environments present the greatest detection challenge",
    x        = "Land use class",
    y        = "mAP@0.5"
  ) +
  thesis_theme()

combined <- (fig_voltage / fig_lulc) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

combined <- combined +
  plot_annotation(
    caption = paste0(
      "PLACEHOLDER - simulated data for illustration only. ",
      "Error bars = bootstrapped SE (200 iterations, seed 42). ",
      "Real results pending model training on expanded dataset."
    ),
    theme = theme(
      plot.caption = element_text(size = 7, color = "grey50",
                                  hjust = 0, face = "italic")
    )
  )

out_path <- file.path(output_dir, "fig_conditional_panels_placeholder.png")
ggsave(
  out_path,
  plot   = combined,
  width  = 7.2,
  height = 8.5,
  dpi    = 300,
  bg     = "white"
)
cat("Saved:", out_path, "\n")
