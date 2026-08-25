#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE, warn = 1)

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
model_labels <- c(
  yolo11n = "YOLO11n",
  yolo26n = "YOLO26n",
  dfine_n = "D-FINE-N"
)
model_colors <- c(
  yolo11n = "#0072B2",
  yolo26n = "#D55E00",
  dfine_n = "#009E73"
)
model_shapes <- c(yolo11n = 21, yolo26n = 22, dfine_n = 24)
source_path <- file.path(
  repo_root,
  "results",
  "RGB",
  "summary",
  "final_benchmark_aggregate.json"
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

publication_theme <- function() {
  theme_bw(base_size = 8, base_family = "Times New Roman") +
    theme(
      axis.title = element_text(size = 8, colour = "black"),
      axis.text = element_text(size = 7, colour = "black"),
      axis.ticks = element_line(linewidth = 0.3, colour = "black"),
      panel.border = element_rect(linewidth = 0.45, colour = "black"),
      panel.grid.major = element_line(linewidth = 0.25, colour = "grey85"),
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
  grDevices::svg(
    filename = path,
    width = width,
    height = height,
    pointsize = 8,
    onefile = TRUE,
    family = "Times New Roman",
    bg = "white"
  )
  draw_figure(plot)
  invisible(grDevices::dev.off())
  lines <- sub(
    "[[:blank:]]+$",
    "",
    readLines(path, warn = FALSE, encoding = "UTF-8")
  )
  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeChar(
    paste0(paste(lines, collapse = "\n"), "\n"),
    connection,
    eos = NULL
  )
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
  grDevices::cairo_pdf(
    filename = path,
    width = width,
    height = height,
    onefile = TRUE,
    family = "Times New Roman",
    bg = "white",
    antialias = "subpixel"
  )
  draw_figure(plot)
  invisible(grDevices::dev.off())
  normalize_pdf_metadata(path, title, figure_id, metadata_source)
}

write_png <- function(plot, path, width, height) {
  grDevices::png(
    filename = path,
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
  connection <- file(manifest_path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeChar(paste0(json_text, "\n"), connection, eos = NULL)
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
      linewidth = 0.45
    ) +
    geom_segment(
      aes(
        x = fps_low,
        xend = fps_low,
        y = map_mean - y_cap,
        yend = map_mean + y_cap
      ),
      linewidth = 0.45
    ) +
    geom_segment(
      aes(
        x = fps_high,
        xend = fps_high,
        y = map_mean - y_cap,
        yend = map_mean + y_cap
      ),
      linewidth = 0.45
    ) +
    geom_segment(
      aes(x = fps_mean, xend = fps_mean, y = map_low, yend = map_high),
      linewidth = 0.45
    ) +
    geom_segment(
      aes(
        x = fps_mean - x_cap,
        xend = fps_mean + x_cap,
        y = map_low,
        yend = map_low
      ),
      linewidth = 0.45
    ) +
    geom_segment(
      aes(
        x = fps_mean - x_cap,
        xend = fps_mean + x_cap,
        y = map_high,
        yend = map_high
      ),
      linewidth = 0.45
    ) +
    geom_point(
      aes(fill = model_id, shape = model_id),
      colour = "black",
      size = 3.3,
      stroke = 0.45
    ) +
    scale_colour_manual(values = model_colors, labels = model_labels, guide = "none") +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
    scale_x_continuous(
      name = "Sustained throughput (FPS)",
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.12, 0.12))
    ) +
    scale_y_continuous(
      name = "mAP@0.5:0.95",
      labels = scales::label_percent(accuracy = 0.1),
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.14, 0.16))
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
  dodge <- position_dodge(width = 0.55)

  ggplot(plot_data, aes(x = ap_mean, y = class_name, colour = model_id)) +
    geom_errorbar(
      aes(xmin = ap_low, xmax = ap_high),
      orientation = "y",
      width = 0.24,
      linewidth = 0.45,
      position = dodge
    ) +
    geom_point(
      aes(fill = model_id, shape = model_id),
      colour = "black",
      size = 2.3,
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
      expand = expansion(mult = c(0.08, 0.12))
    ) +
    scale_y_discrete(name = NULL, labels = class_labels) +
    publication_theme() +
    theme(
      legend.position = "bottom",
      legend.box.spacing = grid::unit(2, "pt")
    )
}

load_qualitative_examples <- function(path) {
  payload <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  if (!identical(payload$artifact, "rgb_qualitative_figure_source")) {
    stop("Unexpected qualitative source type in ", path)
  }
  if (length(payload$examples) != 6L) {
    stop("The qualitative grid requires exactly six examples")
  }
  payload$examples
}

build_qualitative_grid <- function(examples) {
  plot <- ggplot()
  for (index in seq_along(examples)) {
    example <- examples[[index]]
    image_path <- file.path(repo_root, as.character(example$path))
    if (!file.exists(image_path)) {
      stop("Qualitative image is missing: ", image_path)
    }
    column <- (index - 1L) %% 3L
    row <- (index - 1L) %/% 3L
    image_ymin <- if (row == 0L) 1.08 else 0.02
    image_ymax <- if (row == 0L) 1.93 else 0.87
    title_y <- if (row == 0L) 2.01 else 0.95
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
        size = 2.45
      )
  }
  plot +
    coord_fixed(
      ratio = 1,
      xlim = c(0, 3),
      ylim = c(0, 2.06),
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
      "Licensed\nsource media",
      "Subject-disjoint\nsplit",
      "Model\ntraining",
      "Validation:\ncheckpoint +\nthreshold",
      "Checksum +\nconfirmation\ngate",
      "Protected\ntest",
      "Tables,\nfigures, public\nartifacts"
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
    label = c(
      "Primary RGB | DMD | 1 FPS | four cues\nSeeds 13, 37, 73; 15,723 frames retained",
      "Exploratory NIR | Drive&Act | 10 FPS | two cues\nSeed 13; training negatives 1:2 versus 1:6"
    ),
    stringsAsFactors = FALSE
  )

  ggplot() +
    geom_rect(
      data = lanes,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
      colour = "grey35",
      linewidth = 0.45,
      show.legend = FALSE
    ) +
    geom_text(
      data = lanes,
      aes(x = (xmin + xmax) / 2, y = (ymin + ymax) / 2, label = label),
      family = "Times New Roman",
      size = 2.65,
      lineheight = 1.05
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
  seed_offset <- c(`13` = -0.035, `37` = 0, `73` = 0.035)
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
      aes(x = x_seed, y = micro_f1, colour = model_id, shape = model_id),
      size = 1.45,
      alpha = 0.45,
      stroke = 0.35
    ) +
    geom_errorbar(
      data = means,
      aes(x = x, ymin = mean - sample_sd, ymax = mean + sample_sd, colour = model_id),
      width = 0.065,
      linewidth = 0.55
    ) +
    geom_point(
      data = means,
      aes(x = x, y = mean, fill = model_id, shape = model_id),
      colour = "black",
      size = 2.6,
      stroke = 0.45
    ) +
    scale_x_continuous(
      name = "Held-out test subject",
      breaks = 1:3,
      labels = c("Subject 05", "Subject 10", "Subject 12"),
      expand = expansion(mult = c(0.12, 0.12))
    ) +
    scale_y_continuous(
      name = "Micro-F1",
      labels = scales::label_percent(accuracy = 1),
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.08, 0.10))
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

  ggplot(
    summary_data,
    aes(x = threshold, y = mean, colour = model_id, fill = model_id)
  ) +
    geom_ribbon(
      aes(ymin = pmax(0, mean - sample_sd), ymax = mean + sample_sd),
      alpha = 0.13,
      colour = NA
    ) +
    geom_line(linewidth = 0.65) +
    geom_point(
      data = selected,
      aes(x = threshold, y = value, shape = model_id, fill = model_id),
      colour = "black",
      size = 1.8,
      stroke = 0.35,
      inherit.aes = FALSE
    ) +
    facet_wrap(~metric, nrow = 1, scales = "free_y") +
    scale_x_continuous(
      name = "Confidence threshold",
      breaks = seq(0, 1, by = 0.2),
      limits = c(0, 1),
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    scale_y_continuous(name = NULL, breaks = scales::breaks_pretty(n = 4)) +
    scale_colour_manual(values = model_colors, labels = model_labels) +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
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
    "mAP@0.5:0.95",
    "Snippet macro-F1",
    "FP snippets / 100 negatives"
  )
  legend_data <- expand.grid(
    model_id = expected_models,
    metric = metric_labels,
    ratio = c("1:2", "1:6"),
    stringsAsFactors = FALSE
  )
  legend_data$value <- 0.5
  legend_data$model_id <- factor(legend_data$model_id, levels = expected_models)
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
      strip.text = element_text(size = 7, face = "bold"),
      panel.spacing = grid::unit(7, "pt")
    )
}

data <- load_rgb_aggregate(source_path)
completed_models <- as.character(data$overall$model_id)
pending_models <- expected_models[!expected_models %in% completed_models]

workflow_outputs <- export_figure(
  build_protocol_workflow(),
  "protocol_workflow",
  width = 7.16,
  height = 2.05,
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
    inputs = list(
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
    )
  ),
  manifest_source = workflow_source_path,
  target_output_dir = shared_output_dir
)

accuracy_outputs <- export_figure(
  build_accuracy_speed(data),
  "accuracy_vs_speed",
  width = 3.5,
  height = 2.58,
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
  height = 2.55,
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
  height = 4.78,
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
qualitative_manifest <- write_manifest(
  "qualitative_examples",
  "rgb_qualitative_examples",
  qualitative_outputs,
  completed_models,
  pending_models,
  list(inputs = qualitative_inputs, training_seed = 13L),
  manifest_source = qualitative_source_path
)

subject_outputs <- export_figure(
  build_subject_sensitivity(subject_source_path),
  "subject_sensitivity",
  width = 7.16,
  height = 2.55,
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
  height = 2.55,
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

nir_outputs <- export_figure(
  build_nir_training_negative_exposure(nir_source_path),
  "training_negative_exposure",
  width = 7.16,
  height = 2.35,
  title = "Pending NIR training-negative exposure results",
  figure_id = "nir_training_negative_exposure",
  metadata_source = nir_source_path,
  target_output_dir = nir_output_dir
)
nir_manifest <- write_manifest(
  "training_negative_exposure",
  "nir_training_negative_exposure",
  nir_outputs,
  character(0),
  expected_models,
  list(status = "pending", seed = 13L, ratios = list("1:2", "1:6")),
  manifest_source = nir_source_path,
  target_output_dir = nir_output_dir
)

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
