#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE, warn = 1)

trailing_arguments <- commandArgs(trailingOnly = TRUE)
only_nir <- "--only-nir" %in% trailing_arguments
script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) {
  stop("Unable to resolve tools/publication/figures.R")
}
script_path <- normalizePath(
  sub("^--file=", "", script_argument),
  winslash = "/",
  mustWork = TRUE
)
repo_root <- normalizePath(
  file.path(dirname(script_path), "..", ".."),
  winslash = "/",
  mustWork = TRUE
)

local_library <- file.path(repo_root, "third_party", "R-library-4.6")
if (dir.exists(local_library)) {
  .libPaths(c(local_library, .libPaths()))
}

suppressPackageStartupMessages({
  library(digest)
  library(ggplot2)
  library(jsonlite)
  library(png)
})

expected_models <- c("yolo11n", "yolo26n", "dfine_n")
nir_expected_models <- c(
  "yolo11n", "yolo26n", "dfine_n",
  "ssdlite_mobilenet_v3_large", "rtdetrv2_s", "yolox_nano",
  "yolov10n", "yolov8n"
)
model_labels <- c(
  yolo11n = "YOLO11n",
  yolo26n = "YOLO26n",
  dfine_n = "D-FINE-N",
  ssdlite_mobilenet_v3_large = "SSDLite-MNV3-L",
  rtdetrv2_s = "RT-DETRv2-S",
  yolox_nano = "YOLOX-Nano",
  yolov10n = "YOLOv10n",
  yolov8n = "YOLOv8n"
)
model_colors <- c(
  yolo11n = "#0072B2",
  yolo26n = "#D55E00",
  dfine_n = "#009E73",
  ssdlite_mobilenet_v3_large = "#CC79A7",
  rtdetrv2_s = "#E69F00",
  yolox_nano = "#56B4E9",
  yolov10n = "#F0E442",
  yolov8n = "#6A3D9A"
)
model_shapes <- c(
  yolo11n = 21, yolo26n = 22, dfine_n = 24,
  ssdlite_mobilenet_v3_large = 23, rtdetrv2_s = 25,
  yolox_nano = 8, yolov10n = 4, yolov8n = 3
)
model_linetypes <- c(
  yolo11n = "solid", yolo26n = "22", dfine_n = "42",
  ssdlite_mobilenet_v3_large = "44", rtdetrv2_s = "13",
  yolox_nano = "1234", yolov10n = "73", yolov8n = "2262"
)

source_path <- file.path(
  repo_root,
  "results",
  "RGB",
  "summary",
  "final_benchmark_aggregate.json"
)
secondary_source_path <- file.path(
  repo_root,
  "results",
  "RGB",
  "summary",
  "secondary_analysis.json"
)
dfine_source_path <- file.path(
  repo_root,
  "results",
  "RGB",
  "dfine_n",
  "training_runs.json"
)
qualitative_source_path <- file.path(
  repo_root,
  "results",
  "RGB",
  "summary",
  "qualitative_examples.json"
)
subject_source_path <- file.path(
  repo_root, "results", "RGB", "summary", "operating_point_by_subject.csv"
)
validation_source_path <- file.path(
  repo_root, "results", "RGB", "summary", "validation_operating_point_sweep.csv"
)
workflow_source_path <- file.path(repo_root, "configs", "RGB", "protocol.yaml")
workflow_image_paths <- c(
  file.path(repo_root, "docs", "assets", "examples", "drinking_annotation_example.png"),
  file.path(repo_root, "docs", "assets", "examples", "hand_over_mouth_annotation_example.png"),
  file.path(repo_root, "docs", "assets", "examples", "driveact_drinking_midpoint.png"),
  file.path(repo_root, "docs", "assets", "examples", "driveact_phone_midpoint.png")
)
nir_source_path <- file.path(
  repo_root, "results", "NIR", "summary",
  "training_negative_exposure_source.json"
)
rgb_output_dir <- file.path(repo_root, "results", "RGB", "summary", "figures")
shared_output_dir <- file.path(repo_root, "results", "summary", "figures")
nir_output_dir <- file.path(repo_root, "results", "NIR", "summary", "figures")
for (directory in c(rgb_output_dir, shared_output_dir, nir_output_dir)) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
}

sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

relative_path <- function(path) {
  normalized <- normalizePath(path, winslash = "/", mustWork = TRUE)
  prefix <- paste0(repo_root, "/")
  if (!startsWith(normalized, prefix)) {
    stop("Publication artifact escaped the repository: ", normalized)
  }
  substring(normalized, nchar(prefix) + 1L)
}

metric_estimate <- function(row, key) {
  value <- row[[key]]
  if (!is.list(value) || is.null(value$mean) || is.null(value$sample_std)) {
    stop(key, " must contain mean and sample_std")
  }
  estimate <- c(
    mean = as.numeric(value$mean),
    sample_std = as.numeric(value$sample_std)
  )
  if (any(!is.finite(estimate)) || estimate[["sample_std"]] < 0) {
    stop(key, " contains an invalid estimate")
  }
  estimate
}

secondary_metric_estimate <- function(row, key) {
  value <- row[[key]]
  if (!is.list(value) || is.null(value$mean) || is.null(value$sample_sd)) {
    stop(key, " must contain mean and sample_sd")
  }
  estimate <- c(
    mean = as.numeric(value$mean),
    sample_std = as.numeric(value$sample_sd)
  )
  if (any(!is.finite(estimate)) || estimate[["sample_std"]] < 0) {
    stop(key, " contains an invalid secondary-analysis estimate")
  }
  estimate
}

load_rgb_aggregate <- function(path) {
  payload <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(payload$artifact, "dms_eval_aggregate")) {
    stop("Unexpected aggregate type in ", path)
  }
  if (!identical(payload$dispersion, "sample_standard_deviation")) {
    stop("Publication error bars require sample standard deviation")
  }

  records <- list()
  for (row in payload$rows) {
    model_id <- as.character(row$model_id)
    if (!model_id %in% expected_models) {
      next
    }
    if (model_id %in% names(records)) {
      stop("Duplicate aggregate row for ", model_id)
    }
    runs <- as.integer(row$runs)
    if (!is.finite(runs) || runs < 2L) {
      stop(model_id, " needs at least two runs for an SD error bar")
    }
    map_estimate <- metric_estimate(row, "map_50_95")
    fps_estimate <- metric_estimate(
      row,
      "tensor_to_final_detections_sustained_fps"
    )
    artifact_bytes <- metric_estimate(
      row,
      "inference_artifact_bytes"
    )[["mean"]]
    if (artifact_bytes <= 0) {
      stop(model_id, " has an invalid inference artifact size")
    }
    records[[model_id]] <- data.frame(
      model_id = model_id,
      runs = runs,
      map_mean = map_estimate[["mean"]],
      map_sd = map_estimate[["sample_std"]],
      fps_mean = fps_estimate[["mean"]],
      fps_sd = fps_estimate[["sample_std"]],
      artifact_mb = artifact_bytes / 1000000,
      stringsAsFactors = FALSE
    )
  }
  if (length(records) == 0L) {
    stop("No supported completed RGB model rows in ", path)
  }
  completed <- expected_models[expected_models %in% names(records)]
  overall <- do.call(rbind, records[completed])
  rownames(overall) <- NULL

  class_records <- list()
  for (model_id in completed) {
    row_index <- which(vapply(
      payload$rows,
      function(item) identical(as.character(item$model_id), model_id),
      logical(1)
    ))
    row <- payload$rows[[row_index]]
    per_class <- row$per_class_ap_50_95
    for (class_name in names(per_class)) {
      estimate <- metric_estimate(per_class, class_name)
      class_records[[length(class_records) + 1L]] <- data.frame(
        model_id = model_id,
        class_name = class_name,
        ap_mean = estimate[["mean"]],
        ap_sd = estimate[["sample_std"]],
        stringsAsFactors = FALSE
      )
    }
  }
  per_class <- do.call(rbind, class_records)
  rownames(per_class) <- NULL
  list(overall = overall, per_class = per_class)
}

format_comparison_value <- function(metric_id, estimate) {
  mean <- estimate[["mean"]]
  sd <- estimate[["sample_std"]]
  switch(
    metric_id,
    map_50 = sprintf("%.2f \u00b1 %.2f", 100 * mean, 100 * sd),
    map_50_95 = sprintf("%.2f \u00b1 %.2f", 100 * mean, 100 * sd),
    micro_f1 = sprintf("%.2f \u00b1 %.2f", 100 * mean, 100 * sd),
    macro_f1 = sprintf("%.2f \u00b1 %.2f", 100 * mean, 100 * sd),
    fd_100 = sprintf("%.2f \u00b1 %.2f", mean, sd),
    latency = sprintf("%.2f \u00b1 %.2f", mean, sd),
    fps = sprintf("%.1f \u00b1 %.1f", mean, sd),
    parameters = sprintf("%.2f", mean / 1000000),
    gflops = sprintf("%.2f", mean / 1000000000),
    stop("Unknown comparison metric: ", metric_id)
  )
}

load_rgb_table_comparison <- function(aggregate_path, secondary_path, dfine_path = NULL) {
  aggregate <- jsonlite::fromJSON(aggregate_path, simplifyVector = FALSE)
  secondary <- jsonlite::fromJSON(secondary_path, simplifyVector = FALSE)
  if (!identical(aggregate$artifact, "dms_eval_aggregate")) {
    stop("Unexpected aggregate type in ", aggregate_path)
  }
  if (!identical(secondary$artifact, "rgb_secondary_analysis")) {
    stop("Unexpected secondary-analysis type in ", secondary_path)
  }
  if (!identical(
    as.character(secondary$inputs$aggregate_sha256),
    sha256_file(aggregate_path)
  )) {
    stop("Secondary analysis does not match the frozen RGB aggregate")
  }

  metric_ids <- c(
    "map_50", "map_50_95", "micro_f1", "macro_f1", "fd_100",
    "latency", "fps", "parameters", "gflops"
  )
  metric_groups <- c(
    map_50 = "Accuracy",
    map_50_95 = "Accuracy",
    micro_f1 = "Accuracy",
    macro_f1 = "Accuracy",
    fd_100 = "Accuracy",
    latency = "Speed",
    fps = "Speed",
    parameters = "Complexity",
    gflops = "Complexity"
  )
  metric_labels <- c(
    map_50 = "mAP50 (%)",
    map_50_95 = "mAP50:95 (%)",
    micro_f1 = "Micro-F1 (%)",
    macro_f1 = "Macro-F1 (%)",
    fd_100 = "FD100 (%)",
    latency = "p50 latency (ms)",
    fps = "Throughput (FPS)",
    parameters = "Parameters (M)",
    gflops = "Compute (GFLOPs)"
  )
  higher_is_better <- c(
    map_50 = TRUE,
    map_50_95 = TRUE,
    micro_f1 = TRUE,
    macro_f1 = TRUE,
    fd_100 = FALSE,
    latency = FALSE,
    fps = TRUE,
    parameters = FALSE,
    gflops = FALSE
  )
  has_sd <- c(
    map_50 = TRUE,
    map_50_95 = TRUE,
    micro_f1 = TRUE,
    macro_f1 = TRUE,
    fd_100 = TRUE,
    latency = TRUE,
    fps = TRUE,
    parameters = FALSE,
    gflops = FALSE
  )

  records <- list()
  for (row in aggregate$rows) {
    model_id <- as.character(row$model_id)
    if (!model_id %in% expected_models) {
      next
    }
    secondary_row <- secondary$model_summary[[model_id]]
    if (is.null(secondary_row)) {
      stop("Missing secondary-analysis summary for ", model_id)
    }
    for (key in c("map_50", "map_50_95", "micro_f1")) {
      aggregate_estimate <- metric_estimate(row, key)
      secondary_estimate <- secondary_metric_estimate(secondary_row, key)
      if (max(abs(aggregate_estimate - secondary_estimate)) > 1e-12) {
        stop("Aggregate and secondary analysis disagree for ", model_id, "/", key)
      }
    }
    aggregate_fd <- metric_estimate(row, "far_per_100_negative_frames")
    secondary_fd <- secondary_metric_estimate(
      secondary_row,
      "far_per_100_negative_frames"
    )
    if (max(abs(aggregate_fd - secondary_fd)) > 1e-12) {
      stop("Aggregate and secondary analysis disagree for ", model_id, "/FD100")
    }

    estimates <- list(
      map_50 = metric_estimate(row, "map_50"),
      map_50_95 = metric_estimate(row, "map_50_95"),
      micro_f1 = metric_estimate(row, "micro_f1"),
      macro_f1 = secondary_metric_estimate(secondary_row, "macro_f1"),
      fd_100 = aggregate_fd,
      latency = metric_estimate(row, "tensor_to_final_detections_p50_ms"),
      fps = metric_estimate(
        row,
        "tensor_to_final_detections_sustained_fps"
      ),
      parameters = metric_estimate(row, "parameters"),
      gflops = metric_estimate(row$flop_estimates, "thop")
    )
    for (metric_id in metric_ids) {
      estimate <- estimates[[metric_id]]
      records[[length(records) + 1L]] <- data.frame(
        model_id = model_id,
        metric_id = metric_id,
        metric_group = unname(metric_groups[[metric_id]]),
        metric_label = unname(metric_labels[[metric_id]]),
        mean = estimate[["mean"]],
        sample_sd = estimate[["sample_std"]],
        higher_is_better = unname(higher_is_better[[metric_id]]),
        has_sd = unname(has_sd[[metric_id]]),
        value_label = format_comparison_value(metric_id, estimate),
        stringsAsFactors = FALSE
      )
    }
  }
  if (!is.null(dfine_path) && file.exists(dfine_path)) {
    dfine_data <- jsonlite::fromJSON(dfine_path, simplifyVector = FALSE)
    dfine_runs <- dfine_data$runs
    get_stat <- function(getter) {
      vals <- vapply(dfine_runs, getter, numeric(1))
      list(mean = mean(vals), sample_std = stats::sd(vals))
    }
    dfine_estimates <- list(
      map_50 = get_stat(function(r) r$protected_test$map_50),
      map_50_95 = get_stat(function(r) r$protected_test$map_50_95),
      micro_f1 = get_stat(function(r) r$protected_test$micro_f1),
      macro_f1 = get_stat(function(r) r$protected_test$macro_f1),
      fd_100 = get_stat(function(r) r$protected_test$false_detections_per_100_negative_frames),
      latency = get_stat(function(r) r$protected_test$tensor_to_final_detections_p50_ms),
      fps = get_stat(function(r) r$protected_test$tensor_to_final_detections_sustained_fps),
      parameters = list(mean = dfine_runs[[1]]$parameters, sample_std = 0),
      gflops = list(mean = 7440000000.0, sample_std = 0)
    )
    for (metric_id in metric_ids) {
      estimate <- dfine_estimates[[metric_id]]
      records[[length(records) + 1L]] <- data.frame(
        model_id = "dfine_n",
        metric_id = metric_id,
        metric_group = unname(metric_groups[[metric_id]]),
        metric_label = unname(metric_labels[[metric_id]]),
        mean = estimate[["mean"]],
        sample_sd = estimate[["sample_std"]],
        higher_is_better = unname(higher_is_better[[metric_id]]),
        has_sd = unname(has_sd[[metric_id]]),
        value_label = format_comparison_value(metric_id, estimate),
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(records) == 0L) {
    stop("No completed RGB models are available for the comparison figure")
  }
  plot_data <- do.call(rbind, records)
  if (length(unique(plot_data$model_id)) < 2L) {
    stop("The normalized comparison requires at least two completed RGB models")
  }

  plot_data$score <- NA_real_
  plot_data$score_low <- NA_real_
  plot_data$score_high <- NA_real_
  domain_fraction <- 1 / 2
  for (metric_id in metric_ids) {
    rows <- which(plot_data$metric_id == metric_id)
    means <- plot_data$mean[rows]
    sds <- if (plot_data$has_sd[rows][[1]]) {
      plot_data$sample_sd[rows]
    } else {
      rep(0, length(rows))
    }
    if (plot_data$higher_is_better[rows][[1]]) {
      domain_high <- max(means + sds)
      domain_low <- domain_high * domain_fraction
    } else {
      domain_low <- min(means - sds)
      domain_high <- domain_low / domain_fraction
    }
    domain_span <- domain_high - domain_low
    if (!is.finite(domain_span) || domain_span <= 0 || domain_low < 0) {
      stop("Invalid half-to-full normalization domain for ", metric_id)
    }
    if (plot_data$higher_is_better[rows][[1]]) {
      scores <- 100 * (means - domain_low) / domain_span
    } else {
      scores <- 100 * (domain_high - means) / domain_span
    }
    score_sd <- 100 * sds / domain_span
    plot_data$score[rows] <- pmin(100, pmax(0, scores))
    plot_data$score_low[rows] <- pmin(100, pmax(0, scores - score_sd))
    plot_data$score_high[rows] <- pmin(100, pmax(0, scores + score_sd))
  }

  plot_data$metric_group <- factor(
    plot_data$metric_group,
    levels = c("Accuracy", "Speed", "Complexity")
  )
  plot_data$metric_label <- factor(
    plot_data$metric_label,
    levels = rev(unname(metric_labels[metric_ids]))
  )
  plot_data$model_id <- factor(plot_data$model_id, levels = expected_models)
  label_gap <- 3.0
  plot_data$label_x <- plot_data$score_low - label_gap
  plot_data$label_hjust <- 1
  plot_data
}

publication_theme <- function() {
  theme_bw(base_size = 8, base_family = "Times New Roman") +
    theme(
      axis.title = element_text(size = 8, colour = "black"),
      axis.text = element_text(size = 7, colour = "black"),
      axis.ticks = element_line(linewidth = 0.3, colour = "black"),
      panel.border = element_rect(linewidth = 0.45, colour = "black"),
      panel.grid.major = element_line(linewidth = 0.22, colour = "grey86"),
      panel.grid.minor = element_blank(),
      legend.title = element_blank(),
      legend.text = element_text(size = 7),
      legend.key.height = grid::unit(9, "pt"),
      legend.key.width = grid::unit(12, "pt"),
      legend.margin = margin(2, 0, 0, 0, unit = "pt"),
      legend.box.spacing = grid::unit(2, "pt"),
      plot.margin = margin(5, 6, 6, 5, unit = "pt")
    )
}

draw_figure <- function(plot) {
  if (inherits(plot, "grob")) {
    grid::grid.draw(plot)
  } else {
    print(plot)
  }
}

write_svg <- function(plot, path, width, height) {
  temp_svg <- file.path(dirname(path), paste0(".", basename(path), ".tmp.svg"))
  grDevices::svg(
    filename = temp_svg,
    width = width,
    height = height,
    pointsize = 8,
    onefile = TRUE,
    family = "Times New Roman",
    bg = "white"
  )
  draw_figure(plot)
  invisible(grDevices::dev.off())
  file.copy(temp_svg, path, overwrite = TRUE)
  unlink(temp_svg)
}

normalize_pdf_metadata <- function(path, title, figure_id, metadata_source) {
  python <- Sys.getenv("PYTHON", unset = "python")
  normalizer <- file.path(repo_root, "tools", "publication", "pdf_metadata.py")
  arguments <- c(
    shQuote(normalizer),
    shQuote(path),
    shQuote(title),
    shQuote(figure_id),
    sha256_file(metadata_source)
  )
  output <- system2(python, arguments, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(
      "Deterministic PDF metadata normalization failed: ",
      paste(output, collapse = "\n")
    )
  }
}

write_pdf <- function(
  plot,
  path,
  width,
  height,
  title,
  figure_id,
  metadata_source
) {
  temp_pdf <- file.path(dirname(path), paste0(".", basename(path), ".tmp.pdf"))
  grDevices::cairo_pdf(
    filename = temp_pdf,
    width = width,
    height = height,
    onefile = TRUE,
    family = "Times New Roman",
    bg = "white",
    antialias = "subpixel"
  )
  draw_figure(plot)
  invisible(grDevices::dev.off())
  file.copy(temp_pdf, path, overwrite = TRUE)
  unlink(temp_pdf)
  normalize_pdf_metadata(path, title, figure_id, metadata_source)
}

write_png <- function(plot, path, width, height) {
  temp_png <- file.path(dirname(path), paste0(".", basename(path), ".tmp.png"))
  grDevices::png(
    filename = temp_png,
    width = width,
    height = height,
    units = "in",
    pointsize = 8,
    bg = "white",
    res = 600,
    family = "Times New Roman",
    type = "cairo-png",
    antialias = "subpixel"
  )
  draw_figure(plot)
  invisible(grDevices::dev.off())
  file.copy(temp_png, path, overwrite = TRUE)
  unlink(temp_png)
}

export_figure <- function(
  plot,
  stem,
  width,
  height,
  title,
  figure_id,
  metadata_source = source_path,
  target_output_dir = rgb_output_dir
) {
  outputs <- list(
    pdf = file.path(target_output_dir, paste0(stem, ".pdf")),
    svg = file.path(target_output_dir, paste0(stem, ".svg")),
    png = file.path(target_output_dir, paste0(stem, ".png"))
  )
  write_pdf(
    plot,
    outputs$pdf,
    width,
    height,
    title,
    figure_id,
    metadata_source
  )
  write_svg(plot, outputs$svg, width, height)
  write_png(plot, outputs$png, width, height)
  outputs
}

write_manifest <- function(
  stem,
  figure_id,
  outputs,
  models,
  pending_models,
  details,
  manifest_source = source_path,
  target_output_dir = rgb_output_dir
) {
  package_versions <- list(
    ggplot2 = as.character(packageVersion("ggplot2")),
    jsonlite = as.character(packageVersion("jsonlite")),
    digest = as.character(packageVersion("digest")),
    png = as.character(packageVersion("png"))
  )
  output_records <- lapply(outputs, function(path) {
    list(path = relative_path(path), sha256 = sha256_file(path))
  })
  payload <- c(
    list(
      artifact = "publication_figure_manifest",
      figure = figure_id,
      generator = "ggplot2",
      generator_script = "tools/publication/figures.R",
      r_version = paste(R.version$major, R.version$minor, sep = "."),
      packages = package_versions,
      source = relative_path(manifest_source),
      source_sha256 = sha256_file(manifest_source),
      models = as.list(models),
      pending_models = as.list(pending_models),
      dispersion = "sample_standard_deviation",
      outputs = output_records
    ),
    details
  )
  json_text <- jsonlite::toJSON(
    payload,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null"
  )
  manifest_path <- file.path(target_output_dir, paste0(stem, ".manifest.json"))
  writeLines(json_text, manifest_path, useBytes = TRUE)
  manifest_path
}

build_accuracy_speed <- function(data) {
  plot_data <- data$overall
  plot_data$model_id <- factor(plot_data$model_id, levels = expected_models)
  plot_data$model_label <- unname(
    model_labels[as.character(plot_data$model_id)]
  )
  plot_data$map_low <- plot_data$map_mean - plot_data$map_sd
  plot_data$map_high <- plot_data$map_mean + plot_data$map_sd
  plot_data$fps_low <- plot_data$fps_mean - plot_data$fps_sd
  plot_data$fps_high <- plot_data$fps_mean + plot_data$fps_sd
  x_span <- diff(range(c(plot_data$fps_low, plot_data$fps_high)))
  y_span <- diff(range(c(plot_data$map_low, plot_data$map_high)))
  x_cap <- max(0.15, x_span * 0.012)
  y_cap <- max(0.00025, y_span * 0.018)
  ggplot(
    plot_data,
    aes(x = fps_mean, y = map_mean, colour = model_id)
  ) +
    geom_segment(
      aes(x = fps_low, xend = fps_high, y = map_mean, yend = map_mean),
      linewidth = 0.35
    ) +
    geom_segment(
      aes(
        x = fps_low,
        xend = fps_low,
        y = map_mean - y_cap,
        yend = map_mean + y_cap
      ),
      linewidth = 0.35
    ) +
    geom_segment(
      aes(
        x = fps_high,
        xend = fps_high,
        y = map_mean - y_cap,
        yend = map_mean + y_cap
      ),
      linewidth = 0.35
    ) +
    geom_segment(
      aes(x = fps_mean, xend = fps_mean, y = map_low, yend = map_high),
      linewidth = 0.35
    ) +
    geom_segment(
      aes(
        x = fps_mean - x_cap,
        xend = fps_mean + x_cap,
        y = map_low,
        yend = map_low
      ),
      linewidth = 0.35
    ) +
    geom_segment(
      aes(
        x = fps_mean - x_cap,
        xend = fps_mean + x_cap,
        y = map_high,
        yend = map_high
      ),
      linewidth = 0.35
    ) +
    geom_point(
      aes(fill = model_id, shape = model_id),
      colour = "black",
      size = 3,
      stroke = 0.45
    ) +
    scale_colour_manual(values = model_colors, labels = model_labels, guide = "none") +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
    scale_x_continuous(
      name = "Sustained throughput (FPS)",
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.08, 0.08))
    ) +
    scale_y_continuous(
      name = "mAP@0.5:0.95",
      labels = scales::label_percent(accuracy = 0.1),
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.10, 0.10))
    ) +
    publication_theme() +
    theme(legend.position = "bottom")
}

build_per_class_ap <- function(data) {
  plot_data <- data$per_class
  class_order <- c(
    "hand_over_mouth",
    "yawning",
    "drinking",
    "phone_use"
  )
  class_labels <- c(
    hand_over_mouth = "Hand over mouth (n=28)",
    yawning = "Yawning (n=33)",
    drinking = "Drinking (n=56)",
    phone_use = "Phone use (n=497)"
  )
  plot_data$model_id <- factor(plot_data$model_id, levels = expected_models)
  plot_data$class_name <- factor(plot_data$class_name, levels = class_order)
  plot_data$ap_low <- plot_data$ap_mean - plot_data$ap_sd
  plot_data$ap_high <- plot_data$ap_mean + plot_data$ap_sd
  dodge <- position_dodge(width = 0.48)

  ggplot(plot_data, aes(x = ap_mean, y = class_name, colour = model_id)) +
    geom_errorbar(
      aes(xmin = ap_low, xmax = ap_high),
      orientation = "y",
      width = 0.18,
      linewidth = 0.35,
      position = dodge
    ) +
    geom_point(
      aes(fill = model_id, shape = model_id),
      colour = "black",
      size = 2.15,
      stroke = 0.4,
      position = dodge
    ) +
    scale_colour_manual(
      values = model_colors,
      labels = model_labels,
      guide = "none"
    ) +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
    scale_x_continuous(
      name = "AP@0.5:0.95",
      labels = scales::label_percent(accuracy = 1),
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.06, 0.08))
    ) +
    scale_y_discrete(name = NULL, labels = class_labels) +
    publication_theme() +
    theme(
      legend.position = "bottom",
      legend.box.spacing = grid::unit(2, "pt")
    )
}

build_normalized_model_comparison <- function(plot_data) {
  dodge_width <- 0.68
  dodge <- position_dodge(width = dodge_width, orientation = "y")
  present_models <- expected_models[
    expected_models %in% as.character(unique(plot_data$model_id))
  ]
  offset_limit <- dodge_width * (length(present_models) - 1) /
    (2 * length(present_models))
  label_offsets <- if (length(present_models) == 1L) {
    0
  } else {
    seq(-offset_limit, offset_limit, length.out = length(present_models))
  }

  # Position data labels: to the left of the point or error bar, unless point is too close to left axis
  plot_data$label_x <- pmax(plot_data$score_low, plot_data$score - 20) - 2.8
  plot_data$label_hjust <- 1
  low_mask <- plot_data$score < 25
  plot_data$label_x[low_mask] <- plot_data$score_high[low_mask] + 2.8
  plot_data$label_hjust[low_mask] <- 0

  data_acc <- plot_data[plot_data$metric_group == "Accuracy", ]
  data_acc$metric_group <- droplevels(data_acc$metric_group)
  data_acc$metric_label <- droplevels(data_acc$metric_label)

  data_right <- plot_data[plot_data$metric_group %in% c("Speed", "Complexity"), ]
  data_right$metric_group <- droplevels(data_right$metric_group)
  data_right$metric_label <- droplevels(data_right$metric_label)

  divider_acc <- data.frame(
    metric_group = factor("Accuracy", levels = "Accuracy"),
    yintercept = c(1.5, 2.5, 3.5, 4.5)
  )

  divider_right <- data.frame(
    metric_group = factor(c("Speed", "Complexity"), levels = c("Speed", "Complexity")),
    yintercept = c(1.5, 1.5)
  )

  make_label_layers <- function(sub_data) {
    lapply(seq_along(present_models), function(index) {
      model_id <- present_models[[index]]
      geom_label(
        data = sub_data[as.character(sub_data$model_id) == model_id, ],
        aes(x = label_x, label = value_label, hjust = label_hjust),
        colour = "black",
        fill = "white",
        label.size = 0,
        linewidth = 0,
        label.padding = grid::unit(1.2, "pt"),
        label.r = grid::unit(0, "pt"),
        size = 3.525,
        family = "Times New Roman",
        position = position_nudge(y = label_offsets[[index]]),
        show.legend = FALSE
      )
    })
  }

  p_acc <- ggplot(data_acc, aes(x = score, y = metric_label, colour = model_id)) +
    geom_hline(
      data = divider_acc,
      aes(yintercept = yintercept),
      colour = "grey40",
      linewidth = 0.35,
      inherit.aes = FALSE
    ) +
    geom_errorbar(
      data = data_acc[data_acc$has_sd, ],
      aes(xmin = score_low, xmax = score_high),
      orientation = "y",
      width = 0.22,
      linewidth = 0.55,
      position = dodge
    ) +
    geom_point(
      aes(fill = model_id, shape = model_id),
      colour = "black",
      size = 2.8,
      stroke = 0.42,
      position = dodge
    ) +
    make_label_layers(data_acc) +
    facet_grid(
      rows = vars(metric_group),
      scales = "free_y",
      space = "free_y",
      switch = "y"
    ) +
    scale_colour_manual(
      values = model_colors[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models],
      guide = "none"
    ) +
    scale_fill_manual(
      values = model_colors[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models]
    ) +
    scale_shape_manual(
      values = model_shapes[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models]
    ) +
    scale_x_continuous(
      name = "Directional position (0 = half-range ref; 100 = best bound)",
      limits = c(0, 105),
      breaks = seq(0, 100, by = 25),
      expand = expansion(mult = c(0, 0))
    ) +
    scale_y_discrete(
      name = NULL
    ) +
    publication_theme() +
    theme(
      axis.title.x = element_text(size = 11.5, colour = "black"),
      axis.text.x = element_text(size = 10.5, colour = "black"),
      axis.text.y = element_text(size = 10.5, colour = "black"),
      strip.text.y = element_text(size = 10.5, face = "bold"),
      legend.position = "none",
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(
        fill = NA,
        colour = "grey72",
        linewidth = 0.18
      ),
      panel.grid.major.x = element_line(linewidth = 0.22, colour = "grey82"),
      panel.grid.major.y = element_blank(),
      strip.background = element_rect(
        fill = "grey94",
        colour = "grey65",
        linewidth = 0.3
      ),
      plot.margin = margin(4, 6, 4, 4, unit = "pt")
    )

  p_right <- ggplot(data_right, aes(x = score, y = metric_label, colour = model_id)) +
    geom_hline(
      data = divider_right,
      aes(yintercept = yintercept),
      colour = "grey40",
      linewidth = 0.35,
      inherit.aes = FALSE
    ) +
    geom_errorbar(
      data = data_right[data_right$has_sd, ],
      aes(xmin = score_low, xmax = score_high),
      orientation = "y",
      width = 0.22,
      linewidth = 0.55,
      position = dodge
    ) +
    geom_point(
      aes(fill = model_id, shape = model_id),
      colour = "black",
      size = 2.8,
      stroke = 0.42,
      position = dodge
    ) +
    make_label_layers(data_right) +
    facet_grid(
      rows = vars(metric_group),
      scales = "free_y",
      space = "free_y",
      switch = "y"
    ) +
    scale_colour_manual(
      values = model_colors[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models],
      guide = "none"
    ) +
    scale_fill_manual(
      values = model_colors[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models]
    ) +
    scale_shape_manual(
      values = model_shapes[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models]
    ) +
    scale_x_continuous(
      name = "Directional position (0 = half-range ref; 100 = best bound)",
      limits = c(0, 105),
      breaks = seq(0, 100, by = 25),
      expand = expansion(mult = c(0, 0))
    ) +
    scale_y_discrete(
      name = NULL
    ) +
    publication_theme() +
    theme(
      axis.title.x = element_text(size = 11.5, colour = "black"),
      axis.text.x = element_text(size = 10.5, colour = "black"),
      axis.text.y = element_text(size = 10.5, colour = "black"),
      strip.text.y = element_text(size = 10.5, face = "bold"),
      panel.spacing.y = grid::unit(14, "pt"),
      legend.position = "none",
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(
        fill = NA,
        colour = "grey72",
        linewidth = 0.18
      ),
      panel.grid.major.x = element_line(linewidth = 0.22, colour = "grey82"),
      panel.grid.major.y = element_blank(),
      strip.background = element_rect(
        fill = "grey94",
        colour = "grey65",
        linewidth = 0.3
      ),
      plot.margin = margin(4, 4, 4, 6, unit = "pt")
    )

  legend_dummy <- ggplot(plot_data, aes(x = score, y = metric_label, colour = model_id, fill = model_id, shape = model_id)) +
    geom_point(size = 3.2, stroke = 0.45) +
    scale_colour_manual(
      values = model_colors[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models],
      guide = "none"
    ) +
    scale_fill_manual(
      values = model_colors[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models]
    ) +
    scale_shape_manual(
      values = model_shapes[expected_models],
      breaks = expected_models,
      labels = model_labels[expected_models]
    ) +
    publication_theme() +
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = 10.5, colour = "black"),
      legend.key.height = grid::unit(14, "pt"),
      legend.key.width = grid::unit(18, "pt"),
      legend.margin = margin(t = 2, b = 6, unit = "pt"),
      legend.box.margin = margin(t = 0, b = 4, unit = "pt"),
      legend.box.spacing = grid::unit(0, "pt")
    )

  g_acc <- ggplotGrob(p_acc)
  g_right <- ggplotGrob(p_right)
  g_legend <- ggplotGrob(legend_dummy)
  leg <- gtable::gtable_filter(g_legend, "guide-box")

  layout_gt <- gtable::gtable(
    widths = grid::unit(c(1.08, 0.04, 1.0), c("null", "null", "null")),
    heights = grid::unit.c(grid::unit(1, "null"), grid::unit(30, "pt"))
  )
  layout_gt <- gtable::gtable_add_grob(
    layout_gt, list(g_acc), t = 1, l = 1, b = 1, r = 1, name = "plot-acc"
  )
  layout_gt <- gtable::gtable_add_grob(
    layout_gt, list(g_right), t = 1, l = 3, b = 1, r = 3, name = "plot-right"
  )
  layout_gt <- gtable::gtable_add_grob(
    layout_gt, list(leg), t = 2, l = 1, b = 2, r = 3, name = "shared-legend"
  )
  layout_gt
}

load_qualitative_examples <- function(path) {
  payload <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(payload$artifact, "rgb_qualitative_figure_source")) {
    stop("Unexpected qualitative source type in ", path)
  }
  if (length(payload$examples) != 9L) {
    stop("The qualitative grid requires exactly nine examples")
  }
  payload$examples
}

build_qualitative_grid <- function(examples) {
  row_count <- length(examples) %/% 3L
  row_stride <- 1.06
  plot_ymax <- row_count * row_stride - 0.06
  plot <- ggplot()
  for (index in seq_along(examples)) {
    example <- examples[[index]]
    image_path <- file.path(repo_root, as.character(example$path))
    if (!file.exists(image_path)) {
      stop("Qualitative image is missing: ", image_path)
    }
    column <- (index - 1L) %% 3L
    row <- (index - 1L) %/% 3L
    row_base <- (row_count - row - 1L) * row_stride
    image_ymin <- row_base + 0.02
    image_ymax <- row_base + 0.87
    title_y <- row_base + 0.95
    label <- paste(
      unname(model_labels[as.character(example$model_id)]),
      as.character(example$case),
      sep = " - "
    )
    plot <- plot +
      annotation_custom(
        grid::rasterGrob(png::readPNG(image_path), interpolate = TRUE),
        xmin = column + 0.02,
        xmax = column + 0.98,
        ymin = image_ymin,
        ymax = image_ymax
      ) +
      annotate(
        "text",
        x = column + 0.5,
        y = title_y,
        label = label,
        family = "Times New Roman",
        fontface = "bold",
        size = 4.2
      )
  }
  plot +
    coord_fixed(
      ratio = 1,
      xlim = c(0, 3),
      ylim = c(0, plot_ymax),
      expand = FALSE,
      clip = "off"
    ) +
    theme_void(base_family = "Times New Roman") +
    theme(plot.margin = margin(1, 1, 1, 1, unit = "pt"))
}

build_protocol_workflow <- function() {
  steps <- data.frame(
    x = 1:7,
    label = c(
      "Source\nmedia",
      "Subject-disjoint\nsplit",
      "Model\ntraining",
      "Validation:\ncheckpoint +\nthreshold",
      "Checksum +\nconfirmation",
      "Protected\ntest",
      "Public\nartifacts"
    ),
    stringsAsFactors = FALSE
  )
  arrows <- data.frame(x = 1:6, xend = 2:7)
  lanes <- data.frame(
    xmin = c(0.55, 3.95),
    xmax = c(3.75, 7.45),
    ymin = c(0.12, 0.12),
    ymax = c(0.86, 0.86),
    fill = c("#EAF4FB", "#E8F5EE"),
    stringsAsFactors = FALSE
  )
  lane_images <- data.frame(
    path = workflow_image_paths,
    xmin = c(0.64, 1.32, 4.04, 4.72),
    xmax = c(1.23, 1.91, 4.63, 5.31),
    ymin = 0.17,
    ymax = 0.80,
    stringsAsFactors = FALSE
  )
  lane_text <- data.frame(
    x = c(2.85, 6.40),
    title = c("Primary RGB | DMD", "Exploratory NIR | Drive&Act"),
    detail = c(
      "The RGB study samples 15,723\nframes at 1 FPS, labels four cues,\nand evaluates seeds 13, 37, and 73.",
      "The NIR study samples two cues\nat 1 FPS midpoints, trains seed 13\nfor 100 epochs, and compares 1:2\nwith 1:6 negative ratios."
    ),
    stringsAsFactors = FALSE
  )

  plot <- ggplot() +
    geom_rect(
      data = lanes,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
      colour = "grey35",
      linewidth = 0.45,
      show.legend = FALSE
    )
  for (index in seq_len(nrow(lane_images))) {
    if (!file.exists(lane_images$path[index])) {
      stop("Workflow image is missing: ", lane_images$path[index])
    }
    plot <- plot + annotation_custom(
      grid::rasterGrob(
        png::readPNG(lane_images$path[index]),
        interpolate = TRUE
      ),
      xmin = lane_images$xmin[index],
      xmax = lane_images$xmax[index],
      ymin = lane_images$ymin[index],
      ymax = lane_images$ymax[index]
    )
  }

  plot +
    geom_rect(
      data = lane_images,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill = NA,
      colour = "grey20",
      linewidth = 0.3
    ) +
    geom_text(
      data = lane_text,
      aes(x = x, y = 0.66, label = title),
      family = "Times New Roman",
      fontface = "bold",
      hjust = 0.5,
      size = 3.2
    ) +
    geom_text(
      data = lane_text,
      aes(x = x, y = 0.39, label = detail),
      family = "Times New Roman",
      hjust = 0.5,
      size = 2.12,
      lineheight = 1.0
    ) +
    geom_segment(
      data = arrows,
      aes(x = x + 0.39, xend = xend - 0.39, y = 1.62, yend = 1.62),
      linewidth = 0.55,
      colour = "#3B5B73",
      arrow = grid::arrow(length = grid::unit(4.5, "pt"), type = "closed")
    ) +
    geom_rect(
      data = steps,
      aes(xmin = x - 0.39, xmax = x + 0.39, ymin = 1.25, ymax = 1.99),
      fill = "white",
      colour = "black",
      linewidth = 0.5
    ) +
    geom_rect(
      data = steps[steps$x == 5, ],
      aes(xmin = x - 0.39, xmax = x + 0.39, ymin = 1.25, ymax = 1.99),
      fill = "#FFF4CC",
      colour = "#A65E00",
      linewidth = 0.65
    ) +
    geom_text(
      data = steps,
      aes(x = x, y = 1.62, label = label),
      family = "Times New Roman",
      size = 2.35,
      lineheight = 0.95
    ) +
    annotate(
      "text", x = 5, y = 1.12, label = "NO TEST TUNING",
      family = "Times New Roman", fontface = "bold", size = 2.25,
      colour = "#8C4B00"
    ) +
    coord_cartesian(xlim = c(0.45, 7.55), ylim = c(0.05, 2.05), clip = "off") +
    scale_fill_identity() +
    theme_void(base_family = "Times New Roman") +
    theme(plot.margin = margin(4, 4, 4, 4, unit = "pt"))
}

build_subject_sensitivity <- function(path) {
  plot_data <- read.csv(path, check.names = FALSE)
  plot_data$model_id <- factor(plot_data$model_id, levels = expected_models)
  subject_levels <- c("subject_05", "subject_10", "subject_12")
  plot_data$subject <- factor(plot_data$subject, levels = subject_levels)
  plot_data$x <- as.numeric(plot_data$subject)
  model_offset <- c(yolo11n = -0.12, yolo26n = 0.12, dfine_n = 0)
  seed_offset <- c(`13` = -0.052, `37` = -0.018, `73` = 0.046)
  plot_data$x_seed <- plot_data$x +
    unname(model_offset[as.character(plot_data$model_id)]) +
    unname(seed_offset[as.character(plot_data$training_seed)])
  groups <- split(
    plot_data,
    interaction(plot_data$model_id, plot_data$subject, drop = TRUE)
  )
  means <- do.call(rbind, lapply(groups, function(group) {
    data.frame(
      model_id = group$model_id[1],
      subject = group$subject[1],
      x = group$x[1] + unname(model_offset[as.character(group$model_id[1])]),
      mean = mean(group$micro_f1),
      sample_sd = stats::sd(group$micro_f1)
    )
  }))
  means$model_id <- factor(means$model_id, levels = expected_models)

  ggplot() +
    geom_point(
      data = plot_data,
      aes(x = x_seed, y = micro_f1, colour = model_id),
      shape = 124,
      size = 2.4,
      alpha = 0.55,
      show.legend = FALSE
    ) +
    geom_errorbar(
      data = means,
      aes(x = x, ymin = mean - sample_sd, ymax = mean + sample_sd, colour = model_id),
      width = 0.055,
      linewidth = 0.4
    ) +
    geom_point(
      data = means,
      aes(x = x, y = mean, fill = model_id, shape = model_id),
      colour = "black",
      size = 2.45,
      stroke = 0.45
    ) +
    scale_x_continuous(
      name = "Held-out test subject",
      breaks = 1:3,
      labels = c("Subject 05", "Subject 10", "Subject 12"),
      expand = expansion(mult = c(0.08, 0.08))
    ) +
    scale_y_continuous(
      name = "Micro-F1",
      labels = scales::label_percent(accuracy = 1),
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.06, 0.08))
    ) +
    scale_colour_manual(values = model_colors, labels = model_labels, guide = "none") +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
    publication_theme() +
    theme(legend.position = "bottom")
}

build_validation_operating_point <- function(path) {
  source <- read.csv(path, check.names = FALSE)
  source$selected_primary <- tolower(as.character(source$selected_primary)) == "true"
  metrics <- c(
    micro_f1 = "Micro-F1 (%)",
    macro_f1 = "Macro-F1 (%)",
    false_detections_per_100_negative_frames = "False detections / 100 negative frames"
  )
  long <- do.call(rbind, lapply(names(metrics), function(metric) {
    scale <- if (metric %in% c("micro_f1", "macro_f1")) 100 else 1
    data.frame(
      model_id = source$model_id,
      training_seed = source$training_seed,
      threshold = source$threshold,
      selected_primary = source$selected_primary,
      metric = unname(metrics[metric]),
      value = source[[metric]] * scale,
      stringsAsFactors = FALSE
    )
  }))
  long$model_id <- factor(long$model_id, levels = expected_models)
  long$metric <- factor(long$metric, levels = unname(metrics))
  groups <- split(
    long,
    interaction(long$model_id, long$metric, long$threshold, drop = TRUE)
  )
  summary_data <- do.call(rbind, lapply(groups, function(group) {
    data.frame(
      model_id = group$model_id[1],
      metric = group$metric[1],
      threshold = group$threshold[1],
      mean = mean(group$value),
      sample_sd = stats::sd(group$value)
    )
  }))
  summary_data$model_id <- factor(summary_data$model_id, levels = expected_models)
  summary_data$metric <- factor(summary_data$metric, levels = unname(metrics))
  selected <- long[long$selected_primary, ]
  selected_groups <- split(
    selected,
    interaction(selected$model_id, selected$metric, drop = TRUE)
  )
  selected_summary <- do.call(rbind, lapply(selected_groups, function(group) {
    data.frame(
      model_id = group$model_id[1],
      metric = group$metric[1],
      threshold_mean = mean(group$threshold),
      threshold_sd = stats::sd(group$threshold),
      value_mean = mean(group$value),
      value_sd = stats::sd(group$value)
    )
  }))
  selected_summary$model_id <- factor(
    selected_summary$model_id,
    levels = expected_models
  )
  selected_summary$metric <- factor(
    selected_summary$metric,
    levels = unname(metrics)
  )
  legend_models <- intersect(
    expected_models,
    unique(as.character(selected_summary$model_id))
  )

  ggplot(
    summary_data,
    aes(
      x = threshold,
      y = mean,
      colour = model_id,
      fill = model_id,
      linetype = model_id
    )
  ) +
    geom_ribbon(
      aes(ymin = pmax(0, mean - sample_sd), ymax = mean + sample_sd),
      alpha = 0.14,
      colour = NA,
      linetype = 0
    ) +
    geom_line(linewidth = 0.62) +
    geom_point(
      data = selected_summary,
      aes(
        x = threshold_mean,
        y = value_mean,
        shape = model_id,
        fill = model_id
      ),
      colour = "#1A1A1A",
      size = 2.55,
      stroke = 0.65,
      inherit.aes = FALSE
    ) +
    facet_wrap(~metric, nrow = 1, scales = "free_y") +
    scale_x_continuous(
      name = "Confidence threshold",
      breaks = seq(0, 1, by = 0.2),
      limits = c(0.01, 0.99),
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    scale_y_continuous(name = NULL, breaks = scales::breaks_pretty(n = 4)) +
    scale_colour_manual(values = model_colors, labels = model_labels) +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
    scale_linetype_manual(values = model_linetypes, labels = model_labels) +
    guides(
      colour = guide_legend(override.aes = list(
        linetype = unname(model_linetypes[legend_models]),
        shape = unname(model_shapes[legend_models]),
        fill = unname(model_colors[legend_models])
      )),
      fill = "none",
      shape = "none",
      linetype = "none"
    ) +
    publication_theme() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(size = 7, face = "bold"),
      panel.spacing = grid::unit(7, "pt")
    )
}

build_nir_training_negative_exposure <- function(path) {
  payload <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(payload$artifact, "nir_training_negative_exposure_figure_source")) {
    stop("Unexpected NIR figure source type in ", path)
  }
  metric_labels <- c(
    map_50_95 = "mAP@0.5:0.95 (%)",
    macro_f1 = "Macro-F1 (%)",
    false_detections_per_100_negative_frames =
      "False detections /\n100 negative frames"
  )
  if (length(payload$rows) > 0L) {
    plot_data <- do.call(rbind, lapply(payload$rows, function(row) {
      do.call(rbind, lapply(names(metric_labels), function(metric) {
        scale <- if (metric %in% c("map_50_95", "macro_f1")) 100 else 1
        data.frame(
          model_id = as.character(row$model_id),
          ratio = as.character(row$ratio),
          metric = unname(metric_labels[metric]),
          value = as.numeric(row[[metric]]) * scale,
          stringsAsFactors = FALSE
        )
      }))
    }))
    plot_data$model_id <- factor(
      plot_data$model_id, levels = nir_expected_models
    )
    plot_data$metric <- factor(plot_data$metric, levels = unname(metric_labels))
    plot_data$ratio <- factor(plot_data$ratio, levels = c("1:2", "1:6"))
    ratio_means <- aggregate(
      value ~ ratio + metric,
      data = plot_data,
      FUN = mean
    )
    ratio_means$x_start <- ifelse(ratio_means$ratio == "1:2", 0.5, 1.5)
    ratio_means$x_end <- ifelse(ratio_means$ratio == "1:2", 1.5, 2.5)

    return(
      ggplot(plot_data, aes(x = ratio, y = value, fill = model_id)) +
        geom_vline(
          xintercept = 1.5,
          colour = "grey35",
          linewidth = 0.55
        ) +
        geom_col(
          colour = "black",
          linewidth = 0.35,
          width = 0.68,
          position = position_dodge(width = 0.76)
        ) +
        geom_segment(
          data = ratio_means,
          aes(x = x_start, xend = x_end, y = value, yend = value),
          inherit.aes = FALSE,
          colour = "black",
          linewidth = 1.1,
          lineend = "butt"
        ) +
        facet_wrap(~metric, nrow = 1, scales = "free_y") +
        scale_x_discrete(
          name = "Training positive:negative ratio",
          expand = expansion(add = 0.5)
        ) +
        scale_y_continuous(
          name = NULL,
          breaks = scales::breaks_pretty(n = 4),
          limits = c(0, NA),
          expand = expansion(mult = c(0, 0.10))
        ) +
        scale_fill_manual(values = model_colors, labels = model_labels) +
        publication_theme() +
        theme(
          legend.position = "bottom",
          axis.title.x = element_text(size = 9.5, colour = "black"),
          axis.text = element_text(size = 8.5, colour = "black"),
          strip.text = element_text(size = 10, face = "bold"),
          panel.spacing = grid::unit(7, "pt")
        )
    )
  }

  metric_labels <- unname(metric_labels)
  legend_data <- expand.grid(
    model_id = nir_expected_models,
    metric = metric_labels,
    ratio = c("1:2", "1:6"),
    stringsAsFactors = FALSE
  )
  legend_data$value <- 0.5
  legend_data$model_id <- factor(
    legend_data$model_id, levels = nir_expected_models
  )
  legend_data$metric <- factor(legend_data$metric, levels = metric_labels)
  legend_data$ratio <- factor(legend_data$ratio, levels = c("1:2", "1:6"))

  ggplot(legend_data, aes(x = ratio, y = value, fill = model_id, shape = model_id)) +
    geom_point(alpha = 0, size = 2.5, show.legend = TRUE) +
    annotate(
      "text", x = 1.5, y = 0.5,
      label = "Protected-test results pending",
      family = "Times New Roman", fontface = "italic", size = 2.65,
      colour = "grey35"
    ) +
    facet_wrap(~metric, nrow = 1) +
    scale_x_discrete(name = "Training positive:negative ratio") +
    scale_y_continuous(name = NULL, breaks = NULL, limits = c(0, 1)) +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
    guides(
      fill = guide_legend(override.aes = list(
        alpha = 1,
        shape = unname(model_shapes),
        fill = unname(model_colors),
        colour = "black"
      )),
      shape = "none"
    ) +
    publication_theme() +
    theme(
      legend.position = "bottom",
      axis.title.x = element_text(size = 9.5, colour = "black"),
      axis.text = element_text(size = 8.5, colour = "black"),
      strip.text = element_text(size = 10, face = "bold"),
      panel.spacing = grid::unit(7, "pt")
    )
}

if (!only_nir) {
data <- load_rgb_aggregate(source_path)
comparison_data <- load_rgb_table_comparison(
  source_path,
  secondary_source_path,
  dfine_source_path
)
completed_models <- as.character(data$overall$model_id)
pending_models <- expected_models[!expected_models %in% completed_models]
comparison_completed <- as.character(unique(comparison_data$model_id))
comparison_pending <- expected_models[!expected_models %in% comparison_completed]

workflow_outputs <- export_figure(
  build_protocol_workflow(),
  "protocol_workflow",
  width = 7.16,
  height = 2.20,
  title = "Benchmark workflow and protected-test gate",
  figure_id = "protocol_workflow",
  metadata_source = workflow_source_path,
  target_output_dir = shared_output_dir
)
nir_protocol_path <- file.path(repo_root, "configs", "NIR", "protocol.yaml")
workflow_manifest <- write_manifest(
  "protocol_workflow",
  "protocol_workflow",
  workflow_outputs,
  expected_models,
  character(0),
  list(
    inputs = c(
      list(
        list(
          track = "RGB",
          path = relative_path(workflow_source_path),
          sha256 = sha256_file(workflow_source_path)
        ),
        list(
          track = "NIR",
          path = relative_path(nir_protocol_path),
          sha256 = sha256_file(nir_protocol_path)
        )
      ),
      lapply(workflow_image_paths, function(path) {
        list(
          role = "dataset_example",
          path = relative_path(path),
          sha256 = sha256_file(path)
        )
      })
    )
  ),
  manifest_source = workflow_source_path,
  target_output_dir = shared_output_dir
)

if ("--only-protocol-workflow" %in% trailing_arguments) {
  message("Built ggplot2 publication figure: protocol_workflow")
  quit(save = "no", status = 0L)
}

comparison_outputs <- export_figure(
  build_normalized_model_comparison(comparison_data),
  "normalized_model_comparison",
  width = 11.2,
  height = 5.35,
  title = "Normalized RGB accuracy, speed, and complexity comparison",
  figure_id = "rgb_normalized_model_comparison"
)
comparison_inputs <- list(
  list(
    role = "macro_f1_source",
    path = relative_path(secondary_source_path),
    sha256 = sha256_file(secondary_source_path)
  )
)
if (file.exists(dfine_source_path)) {
  comparison_inputs[[length(comparison_inputs) + 1L]] <- list(
    role = "dfine_n_source",
    path = relative_path(dfine_source_path),
    sha256 = sha256_file(dfine_source_path)
  )
}
comparison_manifest <- write_manifest(
  "normalized_model_comparison",
  "rgb_normalized_model_comparison",
  comparison_outputs,
  comparison_completed,
  comparison_pending,
  list(
    metrics = list(
      "map_50",
      "map_50_95",
      "micro_f1",
      "macro_f1",
      "far_per_100_negative_frames",
      "tensor_to_final_detections_p50_ms",
      "tensor_to_final_detections_sustained_fps",
      "parameters",
      "flop_estimates.thop"
    ),
    normalization = list(
      range = list(0, 100),
      display_range = list(0, 105),
      right_is_better = TRUE,
      anchors = "best_directional_sample_sd_bound",
      higher_is_better_domain = list(
        left = "(1 / 2) * max(mean + sample_sd)",
        right = "max(mean + sample_sd)"
      ),
      lower_is_better_domain = list(
        left = "min(mean - sample_sd) / (1 / 2)",
        right = "min(mean - sample_sd)"
      ),
      out_of_domain_values = "clipped_to_zero_or_100",
      lower_is_better = list(
        "far_per_100_negative_frames",
        "tensor_to_final_detections_p50_ms",
        "parameters",
        "flop_estimates.thop"
      )
    ),
    inputs = comparison_inputs
  )
)

accuracy_outputs <- export_figure(
  build_accuracy_speed(data),
  "accuracy_vs_speed",
  width = 3.5,
  height = 2.15,
  title = "RGB accuracy-efficiency trade-off",
  figure_id = "rgb_accuracy_vs_speed"
)
accuracy_manifest <- write_manifest(
  "accuracy_vs_speed",
  "rgb_accuracy_vs_speed",
  accuracy_outputs,
  completed_models,
  pending_models,
  list(metrics = list("map_50_95", "tensor_to_final_detections_sustained_fps"))
)

class_outputs <- export_figure(
  build_per_class_ap(data),
  "per_class_ap",
  width = 3.5,
  height = 2.15,
  title = "RGB per-class average precision",
  figure_id = "rgb_per_class_ap"
)
class_manifest <- write_manifest(
  "per_class_ap",
  "rgb_per_class_ap",
  class_outputs,
  completed_models,
  pending_models,
  list(metric = "per_class_ap_50_95")
)

qualitative_examples <- load_qualitative_examples(qualitative_source_path)
qualitative_plot <- build_qualitative_grid(qualitative_examples)
qualitative_outputs <- export_figure(
  qualitative_plot,
  "qualitative_examples",
  width = 7.16,
  height = 7.35,
  title = "RGB qualitative successes and errors",
  figure_id = "rgb_qualitative_examples",
  metadata_source = qualitative_source_path
)
qualitative_inputs <- lapply(qualitative_examples, function(example) {
  image_path <- file.path(repo_root, as.character(example$path))
  list(
    model_id = example$model_id,
    case = example$case,
    image_id = example$image_id,
    path = relative_path(image_path),
    sha256 = sha256_file(image_path)
  )
})
qualitative_model_ids <- unique(vapply(
  qualitative_examples,
  function(example) as.character(example$model_id),
  character(1)
))
qualitative_completed <- expected_models[expected_models %in% qualitative_model_ids]
qualitative_pending <- expected_models[!expected_models %in% qualitative_model_ids]
qualitative_manifest <- write_manifest(
  "qualitative_examples",
  "rgb_qualitative_examples",
  qualitative_outputs,
  qualitative_completed,
  qualitative_pending,
  list(inputs = qualitative_inputs, training_seed = 13L),
  manifest_source = qualitative_source_path
)

subject_outputs <- export_figure(
  build_subject_sensitivity(subject_source_path),
  "subject_sensitivity",
  width = 7.16,
  height = 2.15,
  title = "RGB held-out subject sensitivity",
  figure_id = "rgb_subject_sensitivity",
  metadata_source = subject_source_path
)
subject_manifest <- write_manifest(
  "subject_sensitivity",
  "rgb_subject_sensitivity",
  subject_outputs,
  completed_models,
  pending_models,
  list(
    metric = "micro_f1",
    points = "individual_training_seeds",
    error_bars = "sample_standard_deviation_across_seeds"
  ),
  manifest_source = subject_source_path
)

validation_outputs <- export_figure(
  build_validation_operating_point(validation_source_path),
  "validation_operating_point",
  width = 7.16,
  height = 2.35,
  title = "RGB validation confidence sweep",
  figure_id = "rgb_validation_operating_point",
  metadata_source = validation_source_path
)
validation_manifest <- write_manifest(
  "validation_operating_point",
  "rgb_validation_operating_point",
  validation_outputs,
  completed_models,
  pending_models,
  list(
    split = "validation",
    thresholds = list(start = 0.01, stop = 0.99, step = 0.01),
    primary_selection_metric = "micro_f1"
  ),
  manifest_source = validation_source_path
)
}

nir_payload <- jsonlite::fromJSON(nir_source_path, simplifyVector = FALSE)
nir_completed_models <- if (length(nir_payload$rows) > 0L) {
  intersect(
    nir_expected_models,
    unique(vapply(nir_payload$rows, function(row) {
      as.character(row$model_id)
    }, character(1)))
  )
} else {
  character(0)
}
nir_pending_models <- nir_expected_models[
  !nir_expected_models %in% nir_completed_models
]
nir_status <- if (length(nir_pending_models) == 0L) {
  "complete"
} else if (length(nir_completed_models) == 0L) {
  "pending"
} else {
  "partial"
}

nir_outputs <- export_figure(
  build_nir_training_negative_exposure(nir_source_path),
  "training_negative_exposure",
  width = 7.16,
  height = 2.35,
  title = paste(tools::toTitleCase(nir_status), "NIR training-negative exposure results"),
  figure_id = "nir_training_negative_exposure",
  metadata_source = nir_source_path,
  target_output_dir = nir_output_dir
)
nir_manifest <- write_manifest(
  "training_negative_exposure",
  "nir_training_negative_exposure",
  nir_outputs,
  nir_completed_models,
  nir_pending_models,
  list(
    status = nir_status,
    seed = 13L,
    ratios = list("1:2", "1:6"),
    encoding = list(
      type = "model_color",
      model_colors = as.list(model_colors[nir_expected_models]),
      ratio_mean = "solid_black_segment"
    ),
    sampling = list(
      fps = 1L,
      sample_time_seconds = 0.5,
      source_frame_offset = 14L
    )
  ),
  manifest_source = nir_source_path,
  target_output_dir = nir_output_dir
)

if (only_nir) {
  cat("Built ggplot2 publication figure: training_negative_exposure\n")
  for (path in c(unlist(nir_outputs), nir_manifest)) {
    cat(normalizePath(path, winslash = "/", mustWork = TRUE), "\n")
  }
  quit(save = "no", status = 0L)
}

cat(
  "Built ggplot2 publication figures for:",
  paste(completed_models, collapse = ", "),
  "\n"
)
if (length(pending_models) > 0L) {
  cat(
    "Pending completed aggregate rows:",
    paste(pending_models, collapse = ", "),
    "\n"
  )
}
output_paths <- c(
  unlist(workflow_outputs),
  workflow_manifest,
  unlist(comparison_outputs),
  comparison_manifest,
  unlist(accuracy_outputs),
  accuracy_manifest,
  unlist(class_outputs),
  class_manifest,
  unlist(qualitative_outputs),
  qualitative_manifest,
  unlist(subject_outputs),
  subject_manifest,
  unlist(validation_outputs),
  validation_manifest,
  unlist(nir_outputs),
  nir_manifest
)
for (path in output_paths) {
  cat(normalizePath(path, winslash = "/", mustWork = TRUE), "\n")
}
