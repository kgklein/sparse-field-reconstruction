PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_lag_tetrahedra.py \
  --timeseries-metadata figs_velocity/helioswarm_timeseries_metadata.json \
  --elsasser-pairs-npz figs_velocity/helioswarm_timeseries_elsasser_pairs.npz \
  --elsasser-pairs-json figs_velocity/helioswarm_timeseries_elsasser_pairs.json \
  --time-index 0 \
  --dep-max 0.85 \
  --max-arrows 400 \
  --highlight-tetrahedron-index 120 \
  --plot \
  --output-dir figs_lag_tetrahedra
