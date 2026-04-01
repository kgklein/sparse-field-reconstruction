PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --field-kind smooth_3d \
  --methods rbf \
  --noise-levels 0.0 \
  --nx 10 \
  --ny 9 \
  --nz 8 \
  --hs-path ../HelioSwarm/HS-RT/PhB_SRD5B_0x75b \
  --hs-time '2029-06-26 00:00:00' \
  --output-dir /tmp/sparse_recon_hs_smoke
