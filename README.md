# Sparse Field Reconstruction

A benchmarking framework for reconstructing turbulent vector fields from sparse spatial measurements.

---

## Project overview

This project studies how well we can reconstruct a vector field from a limited number of spatial samples.

The core workflow is:

1. start from a known ground-truth field  
2. sample that field at sparse locations  
3. reconstruct the field over a larger domain  
4. compare reconstruction against truth using common metrics  

The motivating application is space plasma physics, where spacecraft provide sparse measurements of a large volume.

---

## Quick Start

```bash
git clone https://github.com/kgklein/sparse-field-reconstruction.git
cd sparse-field-reconstruction

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev,notebooks]"

python scripts/run_baseline.py

This *should* create a results/ directory with output metrics and figures.

## Repository Structure

sparse-field-reconstruction/
├── configs/
├── data/
├── scripts/
├── src/
└── tests/

## Data

A small synthetic datasets for testing is included.

Large simulation snapshots (~GB scale) are not stored in this repository.
To use external data:

export SPARSE_RECON_DATA=/path/to/your/data

Then point your config or script to that location.

##Development goals
-baseline methods: nearest, linear, RBF
-shared sampling geometries
-consistent evaluation metrics
-reproducible numerical experiments

##Contribution guidelines

For any new method:
-test on a shared dataset
-compare against a baseline
-report common metrics
-save results reproducibly

##License

MIT.