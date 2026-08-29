# IEEE Manuscript

[← Back to documentation](../README.md)

`main.tex` and `references.bib` are the authoritative Draft 2 sources;
`main.pdf` is the only supported compiled manuscript and the PI-review copy.
Do not keep alternate compiled PDFs in this directory. Run
`scripts/publication/build_manuscript.bat` from Windows. Generated LaTeX
auxiliaries are ignored.

The manuscript uses the IEEE conference template dated June 27, 2024.
The tracked `IEEEtran.cls` is byte-identical to the class distributed in
`conference-latex-template.zip` (SHA-256
`c972aca108fda004c3514d63658e02816da2e54d9a1451e870b9bd970e003f55`).
`main.tex` retains the official template preamble and uses only manuscript-
required packages beyond it; IEEE font, margin, column, and bibliography
settings are not overridden.

> [!IMPORTANT]
> Preserve the draft's wording, structure, citations, and formatting. The
> editing guardrail at the top of `main.tex` applies to human and automated
> updates: make only surgical changes backed by checksum-frozen results or an
> explicit author request. Rows for unfinished NIR runs must remain absent and
> must never be inferred. Rebuild after every applicable update and
> verify that the paper remains at most six pages, including references.
