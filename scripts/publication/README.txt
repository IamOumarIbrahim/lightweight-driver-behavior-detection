PUBLICATION FIGURES
Run build_figures.bat after aggregate result JSON is frozen. It uses the pinned project-local R and ggplot2 runtime to create IEEE-sized PDF and SVG figures plus 600-DPI PNG previews. Missing R dependencies are installed automatically under ignored third_party. The builder never reads checkpoints or active run directories.

PUBLICATION MANUSCRIPT
Run build_manuscript.bat to compile docs/manuscript/main.tex with MiKTeX. The script runs pdflatex and BibTeX directly, so Perl and latexmk are not required. The source remains authoritative; main.pdf is the review copy.
