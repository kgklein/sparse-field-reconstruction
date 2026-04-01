##Contributing Guide

Thank you for contributing to the Sparse Field Reconstruction project!

This repository is designed to support a collaborative research effort. The goal is to maintain a clean, reproducible, and comparable workflow across all contributions.

#Core principles
-Keep main stable and runnable at all times
-Prefer small, focused contributions
-All methods must be comparable using shared datasets and metrics
-Reproducibility matters as much as performance

Workflow overview

1 Create a feature branch

Do not work directly on main.

git checkout -b feature/your-feature-name

Examples:
-feature/rbf-baseline
-feature/divergence-metric
-feature/simulation-loader

2 Make your changes

Keep changes focused:
-one method
-one metric
-one dataset
-one experiment

Avoid mixing unrelated changes in a single branch.

3 Test your work

Before submitting a pull request:

Ensure the baseline script still runs:

python scripts/run_baseline.py

Verify your feature works on at least one shared dataset

4 Open a pull request

When ready:
-Push your branch
-Open a pull request into main

Include:
-a short description of what you added
-how you tested it
-any relevant results or figures
-Review and merge
-Pull requests will be reviewed before merging
-Do not merge your own PR unless explicitly instructed
-main is protected to ensure stability

##Contribution guidelines

#Methods

All new reconstruction methods should:
-follow the shared interface in methods/base.py
-be runnable within the existing pipeline
-be tested against at least one baseline method

#Metrics

All evaluation metrics should:
-operate on shared data structures
-be numerically stable
-handle missing or invalid values (e.g., NaNs)

Datasets
-Do not commit large datasets to the repository
-Use external storage for large files
-Ensure loaders are configurable via paths or environment variables

Experiments
-Use shared datasets and configurations when possible
-Save outputs (metrics, figures) in a reproducible format
-Avoid one-off scripts that cannot be reused

Coding style:
-Follow standard Python conventions
-Use clear variable names
-Prefer readable code over clever code

Optional tools:

black .
ruff check .

##Commit guidelines

#Use clear commit messages

Examples:
-Add RBF reconstruction baseline
-Implement divergence metric for 2D fields
-Add synthetic dataset generator

Avoid:
-fix stuff
-updates

#What success looks like

A strong contribution:
-integrates cleanly into the existing pipeline
-runs on shared datasets
-compares against a baseline
-produces interpretable results

#Questions?

If you're unsure about direction or scope:

-open an issue
-ask early rather than late

#Final note

This is a collaborative research project. Clear, simple, and reproducible contributions will have the most impact.