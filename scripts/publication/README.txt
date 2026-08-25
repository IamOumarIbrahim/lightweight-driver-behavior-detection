PUBLICATION FIGURES
Run analyze_rgb_validation.bat to derive a path-safe validation threshold sweep from the six frozen YOLO validation outputs; it does not load a model or perform inference. Then run build_figures.bat. The pinned project-local R and ggplot2 runtime creates IEEE-sized PDF and SVG figures plus 600-DPI PNG previews for the shared workflow, RGB results, and pending NIR study. Missing R dependencies are installed automatically under ignored third_party.

RGB PREDICTIONS AND SECONDARY ANALYSIS
Run export_rgb_predictions.bat once to sanitize the six frozen local RGB YOLO result files into public, path-safe prediction envelopes. Run analyze_rgb.bat to reproduce the structural annotation audit, macro/per-class/per-subject operating metrics, and paired seed differences without training or inference. These scripts never load checkpoints.

PUBLICATION MANUSCRIPT
Run build_manuscript.bat to compile docs/manuscript/main.tex with MiKTeX. The script runs pdflatex and BibTeX directly, so Perl and latexmk are not required. The source remains authoritative; main.pdf is the review copy.
