PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_structure_functions.py \
  --input-mode interpolated_timeseries \
  --timeseries-csv figs/helioswarm_timeseries.csv \
  --timeseries-metadata figs/helioswarm_timeseries_metadata.json \
  --max-order 6 \
  --n-lambda-bins 20 \
  --undersampled-fraction 0.01 \
  --plot \
  --output-dir figs

# Add `--plot-include-undersampled` before `--output-dir` to show bins below the
# count-threshold mask in the standard structure-function figure.
