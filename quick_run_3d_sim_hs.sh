PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --data-source simulation \
  --simulation-path /tmp/sparse_recon_test_field.npy \
  --methods rbf \
  --sample-counts 24 \
  --geometries random \
  --noise-levels 0.0 \
  --output-dir /tmp/sparse_recon_sim_smoke

