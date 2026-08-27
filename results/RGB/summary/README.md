# RGB Summary

CSV and JSON tables contain the historical six-run YOLO aggregate and
reproducible secondary analyses. The separately frozen D-FINE-N result is under
[`../dfine_n`](../dfine_n/). [`figures`](figures/) contains ggplot2-generated PDF, SVG, and 600-DPI
PNG publication plots plus source and hash manifests, including the seed-13
qualitative grid.

- Rebuild all plots with `scripts/publication/build_figures.bat`.
- Recompute analysis tables with `scripts/publication/analyze_rgb.bat`.
- [`docs/results.md`](../../../docs/results.md) remains the authoritative
  human-readable aggregate report.
