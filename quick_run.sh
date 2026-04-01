PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --methods rbf \
  --sample-counts 24 \
  --geometries random \
  --noise-levels 0.0 \
  --nx 24 \
  --ny 24 \
  --output-dir /tmp/sparse_recon_smoke
