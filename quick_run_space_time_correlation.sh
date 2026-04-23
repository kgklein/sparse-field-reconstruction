PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_space_time_correlation.py \
  --timeseries-csv figs/helioswarm_timeseries.csv \
  --timeseries-metadata figs/helioswarm_timeseries_metadata.json \
  --spacecraft-labels H,N1,N2,N3,N4,N5,N6,N7,N8 \
  --n-r-bins 24 \
  --max-tau-seconds 60 \
  --plot \
  --plot-contour \
  --output-dir figs
