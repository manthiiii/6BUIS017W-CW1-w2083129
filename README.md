# 6BUIS017W CW1: Market Analysis and Stock Segmentation Using RFM

Student: Manthi Gunaratne (w2083129)

## Contents

- `notebook/6BUIS017W_CW1_w2083129_Gunaratne.ipynb`: the unified analysis notebook (executed, outputs saved)
- `notebook/make_unified_notebook.py`: script that generates the notebook
- `data/`: data snapshot used for the analysis (retrieved 2026-11-20)
  - `prices_raw_2021-01-01_to_2024-12-31.csv`: daily prices and volume for the 503 S&P 500 tickers and ^GSPC
  - `sp500_constituents.csv`: ticker list from Wikipedia
  - `splits_events.csv`: split events found in the snapshot
  - `exclusion_log.csv`: shares excluded and why
  - `retrieval_date.txt`
- `outputs/`: results used in the report (feature table, cluster assignments, profiles, k diagnostics, figures)

## Key settings

- Study period: 1 Jan 2021 to 1 Jan 2025 (end exclusive)
- Observation date: 31 Dec 2024
- Shares analysed: 486 (503 requested, 17 excluded)
- Weekly return window: 207 complete weeks
- RFM: 20-session baseline, 60-session window, busy day = volume > 2x baseline mean
- Clustering: Ward and K-Means, k = 3, random seed 42

## Reproducing

1. Install Python 3.13 with pandas, numpy, scikit-learn, scipy, matplotlib and yfinance (versions in `outputs/package_versions.txt`).
2. Open the notebook and run all cells. It reads the local snapshot in `data/` when present, so results match the report without re-downloading.
