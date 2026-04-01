PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --data-source simulation \
  --simulation-path /home/kgklein/Codes/gkeyll/data/ot3D_field_75.npy \
  --methods rbf \
  --noise-levels 0.0 \
  --hs-path /home/kgklein/Codes/HelioSwarm/DRM \
  --hs-time "2029-09-26 00:00:00" \
  --include-hub \
  --output-dir /tmp/sparse_recon_sim_hs
