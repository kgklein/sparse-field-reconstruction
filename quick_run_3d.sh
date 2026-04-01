PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --field-kind smooth_3d \
  --methods rbf \
  --sample-counts 48 \
  --geometries random,multi_probe_like \
  --noise-levels 0.0 \
  --nx 10 \
  --ny 9 \
  --nz 8 \
  --output-dir /tmp/sparse_recon_3d_smoke