NIR ANNOTATIONS
annotations.json is the sanitized source pool. ratio_1to2.json and ratio_1to6.json freeze the two training conditions. Validation and test task identities are identical, and the 1:2 negatives are nested inside the 1:6 training pool.
Run scripts/data/05_build_nir_review_snippets.bat after frame preparation to create portable 10-FPS local videos for Label Studio review.
