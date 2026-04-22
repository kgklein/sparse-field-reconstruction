PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_structure_functions.py \
  --input-mode interpolated_timeseries \
  --timeseries-csv /tmp/sparse_recon_hs_timeseries/helioswarm_timeseries.csv \
  --timeseries-metadata /tmp/sparse_recon_hs_timeseries/helioswarm_timeseries_metadata.json \
  --max-order 6 \
  --n-lambda-bins 20 \
  --plot \
  --output-dir /tmp/sparse_recon_structure_functions
