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
output_dir <- file.path(repo_root, "results", "RGB", "summary", "figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

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
      plot.margin = margin(3, 4, 3, 3, unit = "pt")
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
  metadata_source = source_path
) {
  outputs <- list(
    pdf = file.path(output_dir, paste0(stem, ".pdf")),
    svg = file.path(output_dir, paste0(stem, ".svg")),
    png = file.path(output_dir, paste0(stem, ".png"))
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
  manifest_source = source_path
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
  manifest_path <- file.path(output_dir, paste0(stem, ".manifest.json"))
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
  plot_data$label <- sprintf(
    "%s\n%.2f MB",
    plot_data$model_label,
    plot_data$artifact_mb
  )
  plot_data$label_x <- plot_data$fps_mean + 0.8
  plot_data$label_y <- plot_data$map_mean

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
      aes(fill = model_id, shape = model_id, size = artifact_mb),
      colour = "black",
      stroke = 0.45
    ) +
    geom_text(
      aes(x = label_x, y = label_y, label = label),
      hjust = 0,
      vjust = 0.5,
      lineheight = 0.9,
      family = "Times New Roman",
      size = 2.45,
      colour = "black",
      show.legend = FALSE
    ) +
    annotate(
      "text",
      x = Inf,
      y = -Inf,
      label = "Marker area scales with model file size",
      hjust = 1.03,
      vjust = -0.55,
      family = "Times New Roman",
      size = 2.1,
      colour = "grey25"
    ) +
    scale_colour_manual(values = model_colors, guide = "none") +
    scale_fill_manual(values = model_colors, guide = "none") +
    scale_shape_manual(values = model_shapes, guide = "none") +
    scale_size_area(max_size = 7.2, guide = "none") +
    scale_x_continuous(
      name = "Sustained throughput (FPS)",
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.12, 0.22))
    ) +
    scale_y_continuous(
      name = "mAP@0.5:0.95",
      labels = scales::label_percent(accuracy = 0.1),
      breaks = scales::breaks_pretty(n = 5),
      expand = expansion(mult = c(0.14, 0.16))
    ) +
    coord_cartesian(clip = "off") +
    publication_theme() +
    theme(legend.position = "none")
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
      legend.box.spacing = grid::unit(0, "pt"),
      legend.margin = margin(0, 0, 0, 0, unit = "pt")
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

data <- load_rgb_aggregate(source_path)
completed_models <- as.character(data$overall$model_id)
pending_models <- expected_models[!expected_models %in% completed_models]

accuracy_outputs <- export_figure(
  build_accuracy_speed(data),
  "accuracy_vs_speed",
  width = 3.5,
  height = 2.45,
  title = "RGB accuracy-efficiency trade-off",
  figure_id = "rgb_accuracy_vs_speed"
)
accuracy_manifest <- write_manifest(
  "accuracy_vs_speed",
  "rgb_accuracy_vs_speed",
  accuracy_outputs,
  completed_models,
  pending_models,
  list(bubble_area = "inference_artifact_bytes")
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
  unlist(accuracy_outputs),
  accuracy_manifest,
  unlist(class_outputs),
  class_manifest,
  unlist(qualitative_outputs),
  qualitative_manifest
)
for (path in output_paths) {
  cat(normalizePath(path, winslash = "/", mustWork = TRUE), "\n")
}
