# 6BUIS017W CW1: Market Analysis and Stock Segmentation Using RFM

Student: Manthi Gunaratne (w2083129)

## Contents

- `01_chunk1_prices/`: data snapshot used for the analysis (retrieved 2026-11-20)
  - `prices_raw_2021-01-01_to_2024-12-31.csv`: daily prices and volume for the 503 S&P 500 tickers and ^GSPC
  - `sp500_constituents.csv`: ticker list from Wikipedia
  - `splits_events.csv`: split events found in the snapshot
  - `exclusion_log.csv`: shares excluded and why
  - `retrieval_date.txt`
- `05_unified_notebook/6BUIS017W_CW1_w2083129_Gunaratne.ipynb`: the analysis notebook (executed, outputs saved)
- `05_unified_notebook/outputs/`: results used in the report (feature table, cluster assignments, profiles, k diagnostics, figures)

## Key settings

- Study period: 1 Jan 2021 to 1 Jan 2025 (end exclusive)
- Observation date: 31 Dec 2024
- Shares analysed: 486 (503 requested, 17 excluded)
- Weekly return window: 207 complete weeks
- RFM: 20-session baseline, 60-session window, busy day = volume > 2x baseline mean
- Clustering: Ward and K-Means, k = 3, random seed 42

## Reproducing

1. Install the packages listed in `05_unified_notebook/outputs/package_versions.txt`.
2. Open `05_unified_notebook/6BUIS017W_CW1_w2083129_Gunaratne.ipynb` and run all cells from that folder. The notebook reads the snapshot from `../01_chunk1_prices/`, so results match the report without re-downloading from Yahoo Finance.
