#!/usr/bin/env Rscript

# Reproducible ICLR figures for the consistency-gap manuscript.
#
# Figure 2: seed-resolved FID learning curves for all arms, budgets, and NFE.
# Figure 4: seed-resolved NFE1 factorial contrasts across training budgets.
#
# Run from the repository root:
#   Rscript analysis/plot_iclr_gap_geometry.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(ragg)
  library(scales)
  library(svglite)
  library(tidyr)
})

input_path <- file.path(
  "docs", "figure_source_data", "q256_seed3_5_fidkid50k_source.csv"
)
output_dir <- file.path("docs", "figures")
source_dir <- file.path("docs", "figure_source_data")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

required_columns <- c(
  "seed", "arm", "budget_kimg", "nfe", "fid50k_full", "kid50k_full",
  "sample_count", "status"
)

raw <- read_csv(input_path, show_col_types = FALSE, progress = FALSE)

missing_columns <- setdiff(required_columns, names(raw))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}

expected_grid <- expand_grid(
  seed = 3:5,
  arm = c("A", "B", "C", "D"),
  budget_kimg = c(256, 384, 512, 640, 768, 896, 1024),
  nfe = c(1, 2)
)

observed_grid <- raw |>
  distinct(seed, arm, budget_kimg, nfe)

missing_cells <- anti_join(
  expected_grid,
  observed_grid,
  by = c("seed", "arm", "budget_kimg", "nfe")
)
unexpected_cells <- anti_join(
  observed_grid,
  expected_grid,
  by = c("seed", "arm", "budget_kimg", "nfe")
)
duplicate_cells <- raw |>
  count(seed, arm, budget_kimg, nfe) |>
  filter(n != 1)

if (nrow(raw) != 168L) {
  stop("Expected 168 formal evaluation rows; found ", nrow(raw))
}
if (nrow(missing_cells) > 0 || nrow(unexpected_cells) > 0) {
  stop("Observed seed-arm-budget-NFE grid does not match the frozen design.")
}
if (nrow(duplicate_cells) > 0) {
  stop("Duplicate seed-arm-budget-NFE cells found.")
}
if (any(raw$status != "PASS")) {
  stop("At least one formal evaluation row is not PASS.")
}
if (any(raw$sample_count != 50000L)) {
  stop("At least one evaluation row does not use 50,000 samples.")
}
if (any(!is.finite(raw$fid50k_full)) || any(raw$fid50k_full <= 0)) {
  stop("FID values must be finite and positive.")
}

palette_arms <- c(
  A = "#3F3F3F",
  B = "#0F4D92",
  C = "#42949E",
  D = "#E28E2C"
)

theme_iclr <- function(base_size = 6.4, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = "#222222"),
      axis.ticks = element_line(linewidth = 0.32, colour = "#222222"),
      axis.ticks.length = grid::unit(1.2, "mm"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.4, colour = "#222222"),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.5),
      legend.key.width = grid::unit(5.5, "mm"),
      legend.key.height = grid::unit(2.8, "mm"),
      strip.background = element_rect(
        fill = "#F2F2F2", colour = "#D0D0D0", linewidth = 0.25
      ),
      strip.text = element_text(size = base_size - 0.1, face = "bold"),
      panel.spacing = grid::unit(2.2, "mm"),
      plot.margin = margin(2.2, 2.2, 1.5, 2.2, unit = "mm")
    )
}

save_publication_plot <- function(
    plot,
    stem,
    width_mm = 139.7,
    height_mm = 88,
    dpi = 600) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4

  svg_path <- paste0(stem, ".svg")
  pdf_path <- paste0(stem, ".pdf")
  png_path <- paste0(stem, ".png")

  svglite(svg_path, width = width_in, height = height_in, bg = "white")
  print(plot)
  dev.off()

  cairo_pdf(
    pdf_path,
    width = width_in,
    height = height_in,
    family = "Arial",
    bg = "white",
    onefile = TRUE
  )
  print(plot)
  dev.off()

  agg_png(
    png_path,
    width = width_in,
    height = height_in,
    units = "in",
    res = dpi,
    background = "white",
    scaling = 1
  )
  print(plot)
  dev.off()
}

# ---------------------------------------------------------------------------
# Figure 2: every formal seed, arm, budget, and NFE.
# ---------------------------------------------------------------------------

figure2_data <- raw |>
  transmute(
    seed = as.integer(seed),
    arm = factor(arm, levels = c("A", "B", "C", "D")),
    budget_kimg = as.integer(budget_kimg),
    nfe = as.integer(nfe),
    fid50k_full = as.numeric(fid50k_full),
    kid50k_full = as.numeric(kid50k_full),
    seed_label = factor(
      paste("Seed", seed),
      levels = paste("Seed", 3:5)
    ),
    nfe_label = factor(
      paste("NFE", nfe),
      levels = c("NFE 1", "NFE 2")
    )
  ) |>
  arrange(nfe, seed, arm, budget_kimg)

write_csv(
  figure2_data |>
    select(seed, arm, budget_kimg, nfe, fid50k_full, kid50k_full),
  file.path(source_dir, "figure2_learning_curves_source.csv")
)

figure2 <- ggplot(
  figure2_data,
  aes(
    x = budget_kimg,
    y = fid50k_full,
    colour = arm,
    group = arm,
    linetype = arm,
    linewidth = arm,
    shape = arm,
    size = arm,
    alpha = arm
  )
) +
  geom_line(lineend = "round") +
  geom_point(stroke = 0.15) +
  facet_grid(rows = vars(nfe_label), cols = vars(seed_label)) +
  scale_x_continuous(
    breaks = c(256, 512, 768, 1024),
    limits = c(240, 1040),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_log10(
    breaks = c(3, 10, 30, 100, 300),
    labels = label_number(accuracy = 1),
    limits = c(2.5, 380),
    expand = expansion(mult = c(0.03, 0.05))
  ) +
  scale_colour_manual(
    values = palette_arms,
    labels = c(
      A = "A: baseline",
      B = "B: complete",
      C = "C: target only",
      D = "D: denominator only"
    )
  ) +
  scale_linetype_manual(
    values = c(A = "solid", B = "solid", C = "22", D = "42")
  ) +
  scale_linewidth_manual(
    values = c(A = 0.68, B = 0.84, C = 0.38, D = 0.38)
  ) +
  scale_shape_manual(values = c(A = 16, B = 17, C = 15, D = 18)) +
  scale_size_manual(values = c(A = 1.05, B = 1.20, C = 0.78, D = 0.78)) +
  scale_alpha_manual(values = c(A = 0.95, B = 1.00, C = 0.72, D = 0.72)) +
  guides(
    colour = guide_legend(
      nrow = 1,
      byrow = TRUE,
      override.aes = list(
        linewidth = c(0.68, 0.84, 0.50, 0.50),
        size = c(1.05, 1.20, 0.90, 0.90),
        alpha = 1,
        linetype = c("solid", "solid", "22", "42"),
        shape = c(16, 17, 15, 18)
      )
    ),
    linewidth = "none",
    linetype = "none",
    shape = "none",
    size = "none",
    alpha = "none"
  ) +
  labs(
    x = "Training budget (kimg)",
    y = "FID-50k (log scale)"
  ) +
  theme_iclr()

save_publication_plot(
  figure2,
  file.path(output_dir, "figure2_seed_resolved_learning_curves"),
  width_mm = 139.7,
  height_mm = 88
)

# ---------------------------------------------------------------------------
# Figure 4: seed-resolved NFE1 quality contrasts and their cohort mean.
# ---------------------------------------------------------------------------

wide_fid <- raw |>
  select(seed, arm, budget_kimg, nfe, fid50k_full) |>
  pivot_wider(names_from = arm, values_from = fid50k_full)

figure4_seed_data <- wide_fid |>
  filter(nfe == 1) |>
  transmute(
    seed = as.integer(seed),
    budget_kimg = as.integer(budget_kimg),
    complete = B - A,
    target = C - A,
    denominator = D - A,
    interaction = B - C - D + A
  ) |>
  pivot_longer(
    cols = c(complete, target, denominator, interaction),
    names_to = "contrast",
    values_to = "fid_contrast"
  ) |>
  mutate(
    contrast = factor(
      contrast,
      levels = c("complete", "target", "denominator", "interaction"),
      labels = c(
        "Complete: B - A",
        "Target: C - A",
        "Denominator: D - A",
        "Interaction: B - C - D + A"
      )
    ),
    series = factor(
      paste("Seed", seed),
      levels = c("Seed 3", "Seed 4", "Seed 5", "Cohort mean")
    )
  ) |>
  arrange(contrast, seed, budget_kimg)

figure4_mean_data <- figure4_seed_data |>
  group_by(contrast, budget_kimg) |>
  summarise(fid_contrast = mean(fid_contrast), .groups = "drop") |>
  mutate(
    seed = NA_integer_,
    series = factor(
      "Cohort mean",
      levels = c("Seed 3", "Seed 4", "Seed 5", "Cohort mean")
    )
  )

figure4_data <- bind_rows(figure4_seed_data, figure4_mean_data) |>
  arrange(contrast, series, budget_kimg)

write_csv(
  figure4_seed_data |>
    select(seed, budget_kimg, contrast, fid_contrast),
  file.path(source_dir, "figure4_nfe1_factorial_contrasts_source.csv")
)

series_colours <- c(
  "Seed 3" = "#5C5C5C",
  "Seed 4" = "#8A8A8A",
  "Seed 5" = "#B0B0B0",
  "Cohort mean" = "#0F4D92"
)

figure4 <- ggplot(
  figure4_data,
  aes(
    x = budget_kimg,
    y = fid_contrast,
    colour = series,
    group = series,
    linewidth = series,
    alpha = series,
    shape = series,
    linetype = series,
    size = series
  )
) +
  geom_hline(
    yintercept = 0,
    colour = "#2A2A2A",
    linewidth = 0.30,
    linetype = "dotted"
  ) +
  geom_line(lineend = "round") +
  geom_point(stroke = 0.15) +
  facet_wrap(vars(contrast), ncol = 2) +
  scale_x_continuous(
    breaks = c(256, 512, 768, 1024),
    limits = c(240, 1040),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 1, base = 10),
    breaks = c(-200, -100, -30, -10, -3, 0, 3, 10, 30, 100, 200),
    labels = label_number(accuracy = 1),
    limits = c(-235, 215),
    expand = expansion(mult = c(0.03, 0.03))
  ) +
  scale_colour_manual(values = series_colours) +
  scale_linewidth_manual(
    values = c(
      "Seed 3" = 0.38,
      "Seed 4" = 0.38,
      "Seed 5" = 0.38,
      "Cohort mean" = 0.86
    )
  ) +
  scale_alpha_manual(
    values = c(
      "Seed 3" = 0.78,
      "Seed 4" = 0.78,
      "Seed 5" = 0.78,
      "Cohort mean" = 1
    )
  ) +
  scale_shape_manual(
    values = c(
      "Seed 3" = 16,
      "Seed 4" = 15,
      "Seed 5" = 18,
      "Cohort mean" = 17
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Seed 3" = "solid",
      "Seed 4" = "22",
      "Seed 5" = "42",
      "Cohort mean" = "solid"
    )
  ) +
  scale_size_manual(
    values = c(
      "Seed 3" = 0.82,
      "Seed 4" = 0.82,
      "Seed 5" = 0.82,
      "Cohort mean" = 1.18
    )
  ) +
  guides(
    colour = guide_legend(
      nrow = 1,
      byrow = TRUE,
      override.aes = list(
        linewidth = c(0.45, 0.45, 0.45, 0.90),
        size = c(0.90, 0.90, 0.90, 1.20),
        alpha = 1,
        linetype = c("solid", "22", "42", "solid"),
        shape = c(16, 15, 18, 17)
      )
    ),
    linewidth = "none",
    alpha = "none",
    shape = "none",
    linetype = "none",
    size = "none"
  ) +
  labs(
    x = "Training budget (kimg)",
    y = "Paired FID-50k contrast (pseudo-log axis)"
  ) +
  theme_iclr() +
  theme(
    panel.spacing = grid::unit(3.0, "mm"),
    strip.text = element_text(size = 6.0, face = "bold")
  )

save_publication_plot(
  figure4,
  file.path(output_dir, "figure4_budget_dependent_factorial_contrasts"),
  width_mm = 139.7,
  height_mm = 92
)

message("Rendered Figure 2 and Figure 4 to ", normalizePath(output_dir))
