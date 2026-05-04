## Contributing Guide

Thank you for contributing to the Sparse Field Reconstruction project.

This repository supports a collaborative research effort. The goal is to maintain a clean,
reproducible, and comparable workflow across reconstruction experiments, simulation-backed
HelioSwarm runs, moving-observatory time-series analyses, and downstream turbulence diagnostics.

## Core Principles

- Keep `main` stable and runnable at all times
- Prefer small, focused contributions
- Use shared datasets and comparable workflows whenever possible
- Treat reproducibility as just as important as performance
- Document and test coordinate transforms carefully, especially when moving between HelioSwarm `km`, `rho_p`, and simulation-box coordinates

## Workflow Overview

### 1. Create a feature branch

Do not work directly on `main`.

```bash
git checkout -b feature/your-feature-name
```

Examples:

- `feature/rbf-baseline`
- `feature/divergence-metric`
- `feature/simulation-loader`
- `feature/hs-timeseries-interpolation`

### 2. Make your changes

Keep changes focused:

- one method
- one metric
- one dataset path
- one experiment workflow
- one documentation or visualization improvement

Avoid mixing unrelated changes in a single branch.

### 3. Test your work

Before submitting a pull request:

- ensure the relevant runnable scripts still work
- verify your feature works on at least one shared dataset or example input
- add or update tests when you change sampling behavior, interpolation, output schema, transform logic, or downstream analysis payloads

Examples of useful checks:

- `python scripts/run_baseline.py ...`
- `python scripts/run_hs_timeseries.py ...`
- `python scripts/run_structure_functions.py ...`
- `python scripts/run_space_time_correlation.py ...`
- `python scripts/run_lag_tetrahedra.py ...`
- `pytest`

### 4. Open a pull request

When ready:

- push your branch
- open a pull request into `main`

Include:

- a short description of what you added
- how you tested it
- any relevant results or figures
- any important coordinate-system, transform, or metadata changes

Pull requests will be reviewed before merging. Do not merge your own PR unless explicitly instructed.

## Contribution Guidelines

### Reconstruction methods

All new reconstruction methods should:

- follow the shared interface in `methods/base.py`
- be runnable within the existing pipeline
- be tested against at least one baseline method

### Time-series and interpolation workflows

Contributions may also target moving-observatory analyses. These should:

- fit the existing `scripts/run_hs_timeseries.py` workflow when appropriate
- preserve or clearly document output schema changes
- test structured-grid sampling behavior carefully
- explicitly document any change to interpolation behavior such as `nearest` versus `trilinear`
- preserve compatibility with downstream consumers of `helioswarm_timeseries.csv` and `helioswarm_timeseries_metadata.json`, or document and test the intended migration

### Elsasser pair products and lag-tetrahedra workflows

Changes to velocity/Elsasser workflows should:

- document when `helioswarm_timeseries_elsasser_pairs.npz` and `.json` are expected to be produced
- keep pair-label ordering, shape metadata, and time alignment consistent unless a schema change is intentional
- test downstream compatibility with `scripts/run_lag_tetrahedra.py` when modifying pair-product content or metadata
- document any change to plotting controls or diagnostic fields written by the lag-tetrahedra analysis

### Structure-function and space-time-correlation workflows

Changes to downstream analysis scripts should:

- preserve clearly documented input requirements for existing time-series products
- note any change to JSON structure, diagnostic sections, or plot outputs
- add or update tests when changing binning rules, undersampled-bin handling, decorrelation summaries, or metadata fields

### Coordinate transforms and rescaling

Changes to simulation-backed HelioSwarm workflows should:

- preserve clear semantics between physical `km`, `rho_p`, and simulation-box coordinates
- document any change to `rho_p_km`, `sim_box_rho_p`, or placement logic
- include tests for transform metadata and edge cases when modifying rescaling behavior

### Metrics

All evaluation metrics should:

- operate on shared data structures
- be numerically stable
- handle missing or invalid values such as `NaN`s

### Datasets

- Do not commit large datasets to the repository
- Use external storage for large files
- Ensure loaders are configurable via paths or environment variables
- Keep simulation and HelioSwarm inputs documented clearly when adding new workflows

### Experiments and outputs

- Use shared datasets and configurations when possible
- Save outputs such as metrics, figures, CSV tables, and metadata in reproducible formats
- Avoid one-off scripts that cannot be reused
- If you change metadata or CSV outputs, add tests that verify the new schema
- If you change NPZ or JSON analysis products, add tests that verify the new schema and downstream expectations

### Coding style

- Follow standard Python conventions
- Use clear variable names
- Prefer readable code over clever code

Optional tools:

```bash
black .
ruff check .
```

## Commit Guidelines

Use clear commit messages.

Examples:

- `Add RBF reconstruction baseline`
- `Implement divergence metric for 2D fields`
- `Add synthetic dataset generator`
- `Add trilinear interpolation for HS time-series sampling`

Avoid:

- `fix stuff`
- `updates`

## What Success Looks Like

A strong contribution:

- integrates cleanly into the existing pipeline
- runs on shared datasets or documented example inputs
- compares against a baseline when applicable
- produces interpretable results
- preserves or improves reproducibility
- documents any transform or rescaling assumptions clearly
- keeps upstream and downstream workflow contracts aligned across time-series, Elsasser, and analysis outputs

## Questions?

If you're unsure about direction or scope:

- open an issue
- ask early rather than late

## Final Note

This is a collaborative research project. Clear, simple, and reproducible contributions will have the most impact.
