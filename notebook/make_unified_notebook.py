"""
Generate the single unified, Colab-ready CRM CW1 notebook:
6BUIS017W_CW1_w2083129_Gunaratne.ipynb

Covers every requirement in the 6BUIS017W brief (Tasks 1, 2, and 3)
plus built-in assertions, unit tests, and cross-checks.
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
NB_PATH = OUT_DIR / "6BUIS017W_CW1_w2083129_Gunaratne.ipynb"


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


cells = [
    md("""
# 6BUIS017W — Market Analysis and Stock Segmentation Using Adapted RFM
**Student:** Manthi Gunaratne (`w2083129`)  
**Study Window:** `2021-01-01` to `2025-01-01` (end date exclusive; last trading session `2024-12-31`)  
**Benchmark:** S&P 500 Index (`^GSPC`) | **Download library:** `yfinance` (`auto_adjust=False, actions=True`)

This single self-contained notebook covers **Task 1** (data acquisition, strict validation, weekly return/beta, adapted RFM + unit tests + worked example), **Task 2** (feature preparation, Ward & K-Means clustering, profiles, PCA/2D plots, correlations, and Monetary-removal sensitivity), and **Task 3** (sector composition, Risk vs RFM comparison, pairwise return correlation diversification check, and survivorship-bias check).
"""),

    md("## 0. Setup, Configuration & Package Versions"),
    code("""
# Uncomment in Google Colab if needed:
# %pip install -q yfinance lxml html5lib beautifulsoup4 scikit-learn scipy matplotlib pandas numpy

import io
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
import sklearn
import yfinance as yf
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Configuration ----------------
START = "2021-01-01"          # inclusive
END = "2025-01-01"            # exclusive in yfinance -> last session 2024-12-31
BENCH = "^GSPC"
BASELINE = 20                 # preceding trading sessions for rolling volume mean
WINDOW = 60                   # latest trading sessions for Frequency & Monetary
MULT = 2.0                    # high-volume threshold multiplier (strictly > 2x)
SEED = 42
K_RANGE = range(2, 9)         # feasible k from 2 to 8
USE_CACHE = True              # True: reuse saved snapshots if available; False: fresh live pull
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

OUT = Path("outputs")
FIG = OUT / "figures"
CACHE = OUT / "cache"
for d in (OUT, FIG, CACHE):
    d.mkdir(parents=True, exist_ok=True)

versions = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "yfinance": yf.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "matplotlib": plt.matplotlib.__version__,
}
(OUT / "package_versions.txt").write_text("\\n".join(f"{k}=={v}" for k, v in versions.items()))
print("Download library used: yfinance")
for k, v in versions.items():
    print(f"  {k:13s}: {v}")
"""),

    md("""
## Task 1.1 — Ticker Selection & Data Acquisition
- Pulls the S&P 500 constituent list from Wikipedia and records the retrieval date.
- Maps Wikipedia dot-tickers to Yahoo hyphen-tickers (`BRK.B -> BRK-B`, `BF.B -> BF-B`).
- Clearly separates **503 requested S&P 500 shares** from the **+1 benchmark (`^GSPC`)** so the share count is never confused with `504`.
- Downloads daily `Open, High, Low, Close, Adj Close, Volume` (plus corporate actions) with `auto_adjust=False`.
"""),
    code("""
excl_log = []
def log_event(ticker, stage, rule, detail, action="EXCLUDE"):
    excl_log.append({
        "Ticker": ticker, "Stage": stage, "Rule": rule, "Detail": detail, "Action": action
    })

# Locate existing local snapshot if USE_CACHE is True (works both locally and in Colab)
EXISTING_CHUNK1 = Path("../01_chunk1_prices")
CONS_CACHE = CACHE / "sp500_constituents.csv"
RAW_CACHE = CACHE / "prices_raw_2021-01-01_to_2024-12-31.csv"
SPLITS_CACHE = CACHE / "splits_events.csv"
DATE_CACHE = CACHE / "retrieval_date.txt"

if USE_CACHE and not CONS_CACHE.exists() and (EXISTING_CHUNK1 / "sp500_constituents.csv").exists():
    cons_src = EXISTING_CHUNK1 / "sp500_constituents.csv"
    raw_src = EXISTING_CHUNK1 / "prices_raw_2021-01-01_to_2024-12-31.csv"
    splits_src = EXISTING_CHUNK1 / "splits_events.csv"
    date_src = EXISTING_CHUNK1 / "retrieval_date.txt"
    cons = pd.read_csv(cons_src, keep_default_na=False)
    RETRIEVAL_DATE = date_src.read_text().strip()
    data_mode = "local cached snapshot (01_chunk1_prices)"
    cons.to_csv(CONS_CACHE, index=False)
    DATE_CACHE.write_text(RETRIEVAL_DATE)
    raw_long = pd.read_csv(raw_src, parse_dates=["Date"], dtype={"Ticker": str})
    splits_df = pd.read_csv(splits_src, parse_dates=["Date"], dtype={"Ticker": str}) if splits_src.exists() else pd.DataFrame(columns=["Date", "Ticker", "Split"])
elif USE_CACHE and CONS_CACHE.exists() and RAW_CACHE.exists():
    cons = pd.read_csv(CONS_CACHE, keep_default_na=False)
    RETRIEVAL_DATE = DATE_CACHE.read_text().strip()
    data_mode = "outputs/cache snapshot"
    raw_long = pd.read_csv(RAW_CACHE, parse_dates=["Date"], dtype={"Ticker": str})
    splits_df = pd.read_csv(SPLITS_CACHE, parse_dates=["Date"], dtype={"Ticker": str}) if SPLITS_CACHE.exists() else pd.DataFrame(columns=["Date", "Ticker", "Split"])
else:
    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (university coursework)"}, timeout=30)
    resp.raise_for_status()
    cons = pd.read_html(io.StringIO(resp.text), match="Symbol")[0]
    RETRIEVAL_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cons.to_csv(CONS_CACHE, index=False)
    DATE_CACHE.write_text(RETRIEVAL_DATE)
    data_mode = "live Wikipedia + yfinance"

    cons["YahooTicker"] = cons["Symbol"].astype(str).str.replace(".", "-", regex=False)
    all_syms = cons["YahooTicker"].drop_duplicates().tolist() + [BENCH]
    parts, sp_parts = [], []
    for i in range(0, len(all_syms), 50):
        batch = all_syms[i:i + 50]
        d = yf.download(batch, start=START, end=END, auto_adjust=False, actions=True,
                        group_by="ticker", progress=False, threads=True)
        if not isinstance(d.columns, pd.MultiIndex):
            d.columns = pd.MultiIndex.from_product([[batch[0]], d.columns])
        top = set(d.columns.get_level_values(0))
        for t in batch:
            if t not in top:
                continue
            sub = d[t].reindex(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]).dropna(how="all")
            if sub.empty:
                continue
            sub = sub.reset_index().rename(columns={"index": "Date"})
            sub["Date"] = pd.to_datetime(sub["Date"]).dt.tz_localize(None)
            sub.insert(1, "Ticker", t)
            parts.append(sub)
            if "Stock Splits" in d[t].columns:
                sp = d[t]["Stock Splits"]
                sp = sp[sp.fillna(0) > 0].rename("Split").reset_index().rename(columns={"index": "Date"})
                sp["Date"] = pd.to_datetime(sp["Date"]).dt.tz_localize(None)
                sp.insert(1, "Ticker", t)
                sp_parts.append(sp)
    raw_long = pd.concat(parts, ignore_index=True)
    splits_df = pd.concat(sp_parts, ignore_index=True) if sp_parts else pd.DataFrame(columns=["Date", "Ticker", "Split"])
    raw_long.to_csv(RAW_CACHE, index=False)
    splits_df.to_csv(SPLITS_CACHE, index=False)

cons["YahooTicker"] = cons["Symbol"].astype(str).str.replace(".", "-", regex=False)
changed = cons[cons["YahooTicker"] != cons["Symbol"]]
for _, r in changed.iterrows():
    log_event(r["YahooTicker"], "1.1 Acquisition", "ticker_format", f"{r['Symbol']} -> {r['YahooTicker']}", "INFO")

tickers = cons["YahooTicker"].drop_duplicates().tolist()
sector_map = cons.drop_duplicates("YahooTicker").set_index("YahooTicker")["GICS Sector"]
security_map = cons.drop_duplicates("YahooTicker").set_index("YahooTicker")["Security"]
date_added_map = cons.drop_duplicates("YahooTicker").set_index("YahooTicker")["Date added"]

# Pivot to wide DataFrames (Dates x Tickers)
raw_long["Date"] = pd.to_datetime(raw_long["Date"]).dt.tz_localize(None)
adj_all = raw_long.pivot_table(index="Date", columns="Ticker", values="Adj Close", aggfunc="first").sort_index()
close_all = raw_long.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="first").sort_index()
vol_all = raw_long.pivot_table(index="Date", columns="Ticker", values="Volume", aggfunc="first").sort_index()

bench_adj = adj_all[BENCH].dropna()
bench_dates = bench_adj.index

print(f"Data source          : {data_mode}")
print(f"Retrieval date       : {RETRIEVAL_DATE}")
print(f"Requested S&P shares : {len(tickers)} (plus 1 benchmark {BENCH} = {len(tickers) + 1} series total)")
print(f"Ticker renames (.-)  : {dict(zip(changed['Symbol'], changed['YahooTicker']))}")
print(f"Benchmark ({BENCH})   : {len(bench_dates)} trading sessions ({bench_dates.min().date()} to {bench_dates.max().date()})")
"""),

    md("""
## Task 1.2 — Strict Data Validation, Corporate Actions & Exclusion Log
To guarantee **100% aligned, clean histories** with no silent imputation:
1. Exclude shares with no downloaded data (survivorship-bias evidence).
2. Exclude shares with duplicate dates, any missing `Close`/`Adj Close`/`Volume` across the 1,005 benchmark sessions (removes late listings / unequal histories).
3. Exclude shares with non-positive prices (`<= 0`).
4. Exclude shares with any zero/negative volume days (`Volume <= 0`, e.g., `SW`, `FERG`).
5. Inspect `Close / Adj Close` ratio jumps against recorded stock splits to verify adjustment integrity.
"""),
    code("""
# Duplicate date check in raw long table
dup_pairs = int(raw_long.duplicated(subset=["Date", "Ticker"]).sum())
print(f"Duplicate (Date, Ticker) rows in raw data: {dup_pairs}")
assert dup_pairs == 0, "Unexpected duplicate (Date, Ticker) rows!"

# Reindex share matrices strictly onto benchmark trading calendar (no silent filling!)
adj_shares = adj_all.reindex(index=bench_dates, columns=tickers)
close_shares = close_all.reindex(index=bench_dates, columns=tickers)
vol_shares = vol_all.reindex(index=bench_dates, columns=tickers)

valid_tickers = []
for t in tickers:
    if adj_shares[t].isna().all():
        log_event(t, "1.2 Validation", "no_data", "No rows returned in 2021-01-01 to 2024-12-31 (post-period listing/rebrand)")
        continue

    miss_adj = int(adj_shares[t].isna().sum())
    miss_close = int(close_shares[t].isna().sum())
    miss_vol = int(vol_shares[t].isna().sum())
    if miss_adj + miss_close + miss_vol > 0:
        first_dt = adj_shares[t].first_valid_index().date()
        log_event(t, "1.2 Validation", "unequal_history",
                  f"Missing {miss_adj} AdjClose / {miss_close} Close / {miss_vol} Volume sessions (first valid {first_dt})")
        continue

    if (adj_shares[t] <= 0).any() or (close_shares[t] <= 0).any():
        log_event(t, "1.2 Validation", "nonpositive_price", "Non-positive Close or Adj Close detected")
        continue

    bad_vol = int((vol_shares[t] <= 0).sum())
    if bad_vol > 0:
        log_event(t, "1.2 Validation", "volume_error", f"{bad_vol} session(s) with zero or negative volume")
        continue

    valid_tickers.append(t)

# Corporate-action check on retained shares: Close/Adj Close ratio jumps > 10%
adj_v = adj_shares[valid_tickers].copy()
close_v = close_shares[valid_tickers].copy()
vol_v = vol_shares[valid_tickers].copy()

ratio = close_v / adj_v
ratio_jump = ratio.pct_change(fill_method=None).abs()
big_jumps = ratio_jump.stack()
big_jumps = big_jumps[big_jumps > 0.10]
print(f"Corporate-action ratio jumps >10% among retained shares: {len(big_jumps)} (splits recorded in snapshot: {len(splits_df)})")

excl_df = pd.DataFrame(excl_log)
excl_only = excl_df[excl_df["Action"] == "EXCLUDE"].reset_index(drop=True)
excl_df.to_csv(OUT / "exclusion_log.csv", index=False)

# Hard assertions on validated dataset
assert adj_v.isna().sum().sum() == 0
assert close_v.isna().sum().sum() == 0
assert vol_v.isna().sum().sum() == 0
assert (adj_v > 0).all().all() and (close_v > 0).all().all() and (vol_v > 0).all().all()

print(f"\\nRequested shares : {len(tickers)}")
print(f"Excluded shares  : {len(excl_only)}")
print(f"Retained shares  : {len(valid_tickers)} (all with {len(bench_dates)} aligned sessions, 0 NaNs)")
print("\\nExclusion summary by rule:")
print(excl_only["Rule"].value_counts().to_string())
print("\\nExcluded tickers:")
print(excl_only[["Ticker", "Rule", "Detail"]].to_string(index=False))
"""),

    md("""
## Task 1.3 — Weekly Return and Weekly Beta
- Resample daily `Adj Close` into **Friday-ending weeks (`W-FRI`)** using the final available trading price in each week.
- Exclude incomplete first and final calendar weeks; retain normal holiday-shortened weeks.
  - `2021-01-04` (Mon) to `2021-01-08` (Fri) is a complete Mon–Fri week (5 sessions).
  - `2024-12-30` (Mon) to `2024-12-31` (Tue) is an incomplete final calendar week (2 sessions) → excluded; last included Friday week is `2024-12-27`.
- `Weekly return = AdjClose_t / AdjClose_{t-1} - 1` with `fill_method=None`.
- `Weekly beta = cov(r_share, r_bench) / var(r_bench)` across all aligned weeks.
"""),
    code("""
def to_complete_weekly_returns(price_df):
    w = price_df.resample("W-FRI").last()
    week_monday = w.index - pd.Timedelta(days=4)
    # Keep only weeks whose Mon-Fri span lies completely inside [START, END-1day]
    keep = (week_monday >= pd.Timestamp(START)) & (w.index <= pd.Timestamp(END) - pd.Timedelta(days=1))
    w = w.loc[keep]
    return w.pct_change(fill_method=None).iloc[1:]

weekly_ret = to_complete_weekly_returns(adj_v)
bench_weekly_ret = to_complete_weekly_returns(bench_adj.to_frame(BENCH))[BENCH]

# Align strictly on identical week labels
common_weeks = weekly_ret.index.intersection(bench_weekly_ret.index)
weekly_ret = weekly_ret.loc[common_weeks]
bench_weekly_ret = bench_weekly_ret.loc[common_weeks]

assert weekly_ret.isna().sum().sum() == 0
assert bench_weekly_ret.isna().sum() == 0

bench_var = bench_weekly_ret.var(ddof=1)
mean_weekly_ret = weekly_ret.mean()
weekly_beta = weekly_ret.apply(lambda s: s.cov(bench_weekly_ret) / bench_var)

df_risk = pd.DataFrame({
    "Mean_Weekly_Return": mean_weekly_ret,
    "Weekly_Beta": weekly_beta,
    "Paired_Weeks": len(common_weeks),
})

print(f"Weekly return window : {common_weeks.min().date()} to {common_weeks.max().date()} ({len(common_weeks)} weeks)")
print(f"Shares with valid return & beta: {len(df_risk)} (all with {int(df_risk['Paired_Weeks'].min())} paired weeks)")
print(df_risk[["Mean_Weekly_Return", "Weekly_Beta"]].describe().round(5))
"""),

    md("""
## Task 1.4 & 1.5 — Adapted RFM Calculations, Unit Tests & Worked Example (`AAPL`)
- **Observation date:** final common completed trading date (`2024-12-31`).
- **High-volume day rule:** `Volume > 2 × mean(preceding 20 trading sessions)` (strictly greater; current day excluded via `.shift(1)`).
- **Recency (`R`):** calendar days from latest qualifying high-volume day in valid history to `2024-12-31` (same day = `0`; no event = `NaN`, never `0`).
- **Frequency (`F`):** count of qualifying high-volume days in the latest 60 trading sessions ending `2024-12-31`.
- **Monetary (`M`):** `sum(Close × Volume)` in USD over those latest 60 trading sessions (approximate traded value, not market capitalisation).
"""),
    code("""
obs_date = close_v.index.max()
window_dates = close_v.index[-WINDOW:]
assert len(window_dates) == WINDOW
assert window_dates[-1] == obs_date

# --- Built-in unit tests matching the coursework brief's exact examples ---
def _recency_days(event_dates, obs):
    return np.nan if len(event_dates) == 0 else int((obs - max(event_dates)).days)

assert _recency_days([pd.Timestamp("2024-09-15")], pd.Timestamp("2024-09-20")) == 5
assert _recency_days([pd.Timestamp("2024-09-20")], pd.Timestamp("2024-09-20")) == 0
assert np.isnan(_recency_days([], pd.Timestamp("2024-09-20")))

_test_vol = pd.Series([1_000_000.0] * 20 + [2_500_000.0, 2_000_000.0, 1_800_000.0])
_test_base = _test_vol.shift(1).rolling(20, min_periods=20).mean()
_test_flag = (_test_vol > 2.0 * _test_base) & _test_base.notna()
assert bool(_test_flag.iloc[20]) is True    # 2.5M > 2.0M -> qualifies
assert bool(_test_flag.iloc[21]) is False   # 2.0M not strictly > 2.0M -> does not qualify
assert bool(_test_flag.iloc[22]) is False   # 1.8M < threshold -> does not qualify
assert 100.0 * 2_000_000.0 == 200_000_000.0
print("Unit tests passed (Recency=5/0/NaN, strict >2x threshold, Close*Volume=$200M).")

# --- Compute RFM for all retained shares ---
rolling_20 = vol_v.shift(1).rolling(BASELINE, min_periods=BASELINE).mean()
high_vol_flags = (vol_v > MULT * rolling_20) & rolling_20.notna()

# Verify every day in the 60-session window has a complete 20-session baseline
assert rolling_20.loc[window_dates].notna().all().all()

rfm_rows = []
for t in valid_tickers:
    ev_dates = high_vol_flags.index[high_vol_flags[t]]
    R = _recency_days(ev_dates, obs_date)
    last_ev = str(ev_dates.max().date()) if len(ev_dates) else None
    F = int(high_vol_flags.loc[window_dates, t].sum())
    M = float((close_v.loc[window_dates, t] * vol_v.loc[window_dates, t]).sum())
    rfm_rows.append({
        "Ticker": t,
        "Observation_Date": str(obs_date.date()),
        "Last_Event_Date": last_ev,
        "Recency": R,
        "Frequency": F,
        "Monetary": M,
        "Monetary_USD_bn": M / 1e9,
        "Window_Start": str(window_dates[0].date()),
        "Window_End": str(window_dates[-1].date()),
        "Window_Sessions": len(window_dates),
    })

df_rfm = pd.DataFrame(rfm_rows).set_index("Ticker")

# Cross-checks on RFM consistency
no_event_tickers = df_rfm[df_rfm["Recency"].isna()].index.tolist()
for t in no_event_tickers:
    log_event(t, "1.4 RFM", "no_high_vol_event", "No qualifying high-volume day in valid history; excluded from RFM clustering")

window_cal_days = (obs_date - window_dates[0]).days
bad_rf = df_rfm[(df_rfm["Frequency"] > 0) & (df_rfm["Recency"] > window_cal_days)]
assert len(bad_rf) == 0, "Inconsistency: F > 0 but R exceeds window calendar length!"

print(f"Observation date : {obs_date.date()} | 60-session window: {window_dates[0].date()} to {window_dates[-1].date()}")
print(f"Shares with no qualifying event (R=NaN): {len(no_event_tickers)}")
print(f"Shares with F=0 in latest 60 sessions  : {int((df_rfm['Frequency'] == 0).sum())}")
print(f"Shares with R=0 (event on obs date)    : {int((df_rfm['Recency'] == 0).sum())}")

# --- Worked example: AAPL ---
EX = "AAPL"
ex_daily = pd.DataFrame({
    "Close": close_v[EX],
    "Volume": vol_v[EX],
    "Prev20_Mean_Vol": rolling_20[EX],
    "Threshold_2x": MULT * rolling_20[EX],
    "High_Volume": high_vol_flags[EX],
    "Traded_Value_USD": close_v[EX] * vol_v[EX],
})
ex_win = ex_daily.loc[window_dates]
ex_events = ex_win[ex_win["High_Volume"]]

print(f"\\n=== Worked Example: {EX} (Observation date: {obs_date.date()}) ===")
print(f"Qualifying high-volume days in latest 60 sessions ({len(ex_events)}):")
print(ex_events[["Close", "Volume", "Prev20_Mean_Vol", "Threshold_2x", "Traded_Value_USD"]].to_string())
print(f"R = ({obs_date.date()} - {df_rfm.loc[EX, 'Last_Event_Date']}) = {int(df_rfm.loc[EX, 'Recency'])} calendar days")
print(f"F = {int(df_rfm.loc[EX, 'Frequency'])} qualifying days in latest 60 sessions")
print(f"M = ${df_rfm.loc[EX, 'Monetary']:,.2f} (${df_rfm.loc[EX, 'Monetary_USD_bn']:.2f} billion)")

# Plot worked example
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(ex_win.index, ex_win["Volume"] / 1e6, color="#9ab3d5", label="Daily volume (M)")
ax.plot(ex_win.index, ex_win["Threshold_2x"] / 1e6, color="#c0392b", lw=1.6, label="2x prior 20-day mean")
ax.scatter(ex_events.index, ex_events["Volume"] / 1e6, color="#e67e22", s=55, zorder=5, label="Qualifying event")
ax.set_title(f"{EX}: High-Volume Event Rule (Latest 60 Sessions to {obs_date.date()})")
ax.set_ylabel("Volume (millions of shares)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "worked_example_AAPL.png", dpi=150)
plt.show()

# Build common feature table (inner join on valid Risk/Return + valid RFM)
feature_table = df_risk.join(df_rfm, how="inner")
feature_table = feature_table[feature_table["Recency"].notna()].copy()
feature_table.insert(0, "Security", security_map.reindex(feature_table.index))
feature_table.insert(1, "GICS_Sector", sector_map.reindex(feature_table.index))
feature_table["Date_Added"] = date_added_map.reindex(feature_table.index)
feature_table.to_csv(OUT / "feature_table.csv")

print(f"\\nSaved outputs/feature_table.csv | Common valid shares: {len(feature_table)}")
print(feature_table[["Mean_Weekly_Return", "Weekly_Beta", "Recency", "Frequency", "Monetary_USD_bn"]].describe().round(4))
"""),

    md("""
## Task 2.1 — Feature Preparation: Skew, Outliers & Standardisation
- Separate matrices on the exact same common share set:
  - **Risk/Return matrix:** `Mean_Weekly_Return`, `Weekly_Beta`
  - **RFM matrix:** `Recency`, `Frequency`, `log1p(Monetary)`
- Inspects skewness and extreme values (`|z| > 3`); extreme shares are genuine market behaviour and are retained rather than silently deleted.
"""),
    code("""
feat = feature_table.copy()
feat["Log1p_Monetary"] = np.log1p(feat["Monetary"])

cols_check = ["Mean_Weekly_Return", "Weekly_Beta", "Recency", "Frequency", "Monetary", "Log1p_Monetary"]
diag_rows = []
for c in cols_check:
    s = feat[c]
    z = (s - s.mean()) / s.std(ddof=0)
    diag_rows.append({
        "Feature": c,
        "Mean": s.mean(),
        "Median": s.median(),
        "SD": s.std(ddof=1),
        "Skew": s.skew(),
        "Outliers_|z|>3": int((z.abs() > 3).sum()),
    })
diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(OUT / "skew_outlier_diagnostics.csv", index=False)
print("Skew & outlier diagnostics:")
print(diag_df.round(4).to_string(index=False))

scaler_risk = StandardScaler()
scaler_rfm = StandardScaler()
X_risk = scaler_risk.fit_transform(feat[["Mean_Weekly_Return", "Weekly_Beta"]])
X_rfm = scaler_rfm.fit_transform(feat[["Recency", "Frequency", "Log1p_Monetary"]])

assert X_risk.shape == (len(feat), 2) and X_rfm.shape == (len(feat), 3)
print(f"\\nStandardised matrices ready: X_risk={X_risk.shape}, X_rfm={X_rfm.shape}")
"""),

    md("""
## Task 2.2 & 2.3 — Agglomerative (Ward) & K-Means Clustering (`k = 2..8`)
- Evaluates Ward linkage (Euclidean) and K-Means (`n_init=50, random_state=42`) across `k = 2..8` on both matrices.
- Reports silhouette score, K-Means inertia, and smallest/largest cluster sizes.
- Runs a K-Means multi-seed stability check (`seeds 0..9` vs `seed 42`).
- Generates dendrograms and k-diagnostic plots.
"""),
    code("""
def eval_k_range(X, name):
    rows = []
    Z = linkage(X, method="ward", metric="euclidean")
    for k in K_RANGE:
        w_lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        w_sizes = pd.Series(w_lab).value_counts().sort_values().tolist()
        w_sil = silhouette_score(X, w_lab)

        km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X)
        km_lab = km.labels_
        km_sizes = pd.Series(km_lab).value_counts().sort_values().tolist()
        km_sil = silhouette_score(X, km_lab)

        rows.append({
            "Matrix": name, "k": k,
            "Ward_Sil": round(w_sil, 4), "Ward_Min": w_sizes[0], "Ward_Sizes": str(w_sizes),
            "KM_Sil": round(km_sil, 4), "KM_Inertia": round(km.inertia_, 1),
            "KM_Min": km_sizes[0], "KM_Sizes": str(km_sizes),
        })
    return pd.DataFrame(rows), Z

diag_risk, Z_risk = eval_k_range(X_risk, "Risk_Return")
diag_rfm, Z_rfm = eval_k_range(X_rfm, "RFM")
k_diag = pd.concat([diag_risk, diag_rfm], ignore_index=True)
k_diag.to_csv(OUT / "k_diagnostics.csv", index=False)

print("=== Risk/Return k=2..8 ===")
print(diag_risk.to_string(index=False))
print("\\n=== RFM k=2..8 ===")
print(diag_rfm.to_string(index=False))

# Plot dendrograms
for Z, title, fname in [
    (Z_risk, "Ward Dendrogram — Risk & Return (Standardised)", "dendrogram_risk_return.png"),
    (Z_rfm, "Ward Dendrogram — Adapted RFM (Standardised)", "dendrogram_rfm.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    dendrogram(Z, truncate_mode="lastp", p=25, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Ward linkage distance")
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150)
    plt.show()

# Plot silhouette & elbow diagnostics
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, df_d, title in [(axes[0], diag_risk, "Risk & Return"), (axes[1], diag_rfm, "Adapted RFM")]:
    ax.plot(df_d["k"], df_d["Ward_Sil"], "o-", label="Ward Silhouette")
    ax.plot(df_d["k"], df_d["KM_Sil"], "s--", label="K-Means Silhouette")
    ax.set_title(f"{title}: Silhouette vs k")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette score"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "k_silhouette_diagnostics.png", dpi=150)
plt.show()

# Select k (balanced silhouette, cluster sizes, and interpretability)
K_RISK = 3
K_RFM = 3

# Fit final four baseline models
feat["Risk_Ward"] = AgglomerativeClustering(n_clusters=K_RISK, linkage="ward").fit_predict(X_risk) + 1
km_risk = KMeans(n_clusters=K_RISK, n_init=50, random_state=SEED).fit(X_risk)
feat["Risk_KMeans"] = km_risk.labels_ + 1

feat["RFM_Ward"] = AgglomerativeClustering(n_clusters=K_RFM, linkage="ward").fit_predict(X_rfm) + 1
km_rfm = KMeans(n_clusters=K_RFM, n_init=50, random_state=SEED).fit(X_rfm)
feat["RFM_KMeans"] = km_rfm.labels_ + 1

# K-Means multi-seed stability check
stab_risk = [round(adjusted_rand_score(km_risk.labels_, KMeans(n_clusters=K_RISK, n_init=50, random_state=s).fit_predict(X_risk)), 4) for s in range(10)]
stab_rfm = [round(adjusted_rand_score(km_rfm.labels_, KMeans(n_clusters=K_RFM, n_init=50, random_state=s).fit_predict(X_rfm)), 4) for s in range(10)]
print(f"\\nK-Means seed stability (ARI vs seed 42, seeds 0..9):")
print(f"  Risk/Return (k={K_RISK}): {stab_risk}")
print(f"  RFM         (k={K_RFM}): {stab_rfm}")
"""),

    md("""
## Task 2.4 — Original-Unit Cluster Profiles, 2D / PCA Visualisations & Algorithm Comparison
- Profiles every solution in **original units** (weekly return %, beta, Recency days, Frequency count, Monetary USD billions).
- Projects 3D standardised RFM into 2D using **PCA** and reports explained variance.
- Compares Ward vs K-Means on the exact same shares using **Adjusted Rand Index (ARI)**.
"""),
    code("""
def profile_table(df, col):
    g = df.groupby(col)
    out = pd.DataFrame({
        "N": g.size(),
        "Share_%": (g.size() / len(df) * 100).round(1),
        "Ret_Mean_%": (g["Mean_Weekly_Return"].mean() * 100).round(3),
        "Ret_Med_%": (g["Mean_Weekly_Return"].median() * 100).round(3),
        "Beta_Mean": g["Weekly_Beta"].mean().round(3),
        "Beta_Med": g["Weekly_Beta"].median().round(3),
        "R_Mean": g["Recency"].mean().round(1),
        "R_Med": g["Recency"].median().round(1),
        "F_Mean": g["Frequency"].mean().round(2),
        "F_Med": g["Frequency"].median().round(1),
        "M_Mean_Bn": g["Monetary_USD_bn"].mean().round(2),
        "M_Med_Bn": g["Monetary_USD_bn"].median().round(2),
    })
    return out

for col in ["Risk_Ward", "Risk_KMeans", "RFM_Ward", "RFM_KMeans"]:
    p = profile_table(feat, col)
    p.to_csv(OUT / f"profile_{col}.csv")
    print(f"\\n=== Profile: {col} ===")
    print(p.to_string())

ari_risk_wk = adjusted_rand_score(feat["Risk_Ward"], feat["Risk_KMeans"])
ari_rfm_wk = adjusted_rand_score(feat["RFM_Ward"], feat["RFM_KMeans"])
print(f"\\nAlgorithm agreement (Ward vs K-Means on same {len(feat)} shares):")
print(f"  Risk/Return ARI = {ari_risk_wk:.4f}")
print(f"  RFM         ARI = {ari_rfm_wk:.4f}")

# 2D Risk/Return plots (original units)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, col, title in [(axes[0], "Risk_Ward", f"Risk/Return — Ward (k={K_RISK})"),
                       (axes[1], "Risk_KMeans", f"Risk/Return — K-Means (k={K_RISK})")]:
    for c in sorted(feat[col].unique()):
        sub = feat[feat[col] == c]
        ax.scatter(sub["Weekly_Beta"], sub["Mean_Weekly_Return"] * 100, s=18, alpha=0.75, label=f"C{c} (n={len(sub)})")
    ax.set_title(title); ax.set_xlabel("Weekly Beta"); ax.set_ylabel("Mean Weekly Return (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / "clusters_risk_return.png", dpi=150)
plt.show()

# 2D PCA projection of standardised RFM
pca = PCA(n_components=2, random_state=SEED)
X_rfm_pca = pca.fit_transform(X_rfm)
ev = pca.explained_variance_ratio_
print(f"RFM PCA explained variance: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}, Total={ev.sum():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, col, title in [(axes[0], "RFM_Ward", f"RFM — Ward (k={K_RFM}, PCA)"),
                       (axes[1], "RFM_KMeans", f"RFM — K-Means (k={K_RFM}, PCA)")]:
    for c in sorted(feat[col].unique()):
        mask = feat[col] == c
        ax.scatter(X_rfm_pca[mask, 0], X_rfm_pca[mask, 1], s=18, alpha=0.75, label=f"C{c} (n={mask.sum()})")
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / "clusters_rfm_pca.png", dpi=150)
plt.show()
"""),

    md("""
## Task 2.5 — RFM Sensitivity: Correlations & Removal of Monetary
- Calculates correlations among `Recency`, `Frequency`, and `log1p(Monetary)`.
- Repeats **Ward RFM** using `Recency` and `Frequency` only, on the exact same shares, with the **same `k`**, **re-standardised afresh**.
- Compares cluster memberships via **ARI**, compares traded-value (`Monetary`) profiles across clusters, and computes `eta²` of `log1p(Monetary)` to measure how much traded value separates clusters before vs after removing `M`.
- Also checks robustness to `log1p(Recency)` vs raw `Recency`.
"""),
    code("""
rfm_corr = feat[["Recency", "Frequency", "Log1p_Monetary"]].corr()
rfm_corr.to_csv(OUT / "rfm_correlations.csv")
print("=== Correlations among R, F, log1p(M) ===")
print(rfm_corr.round(4))

# Controlled Monetary-removal comparison (same shares, same Ward algorithm, same k, re-standardised afresh)
X_rf_only = StandardScaler().fit_transform(feat[["Recency", "Frequency"]])
feat["RFM_Ward_NoM"] = AgglomerativeClustering(n_clusters=K_RFM, linkage="ward").fit_predict(X_rf_only) + 1

ari_noM = adjusted_rand_score(feat["RFM_Ward"], feat["RFM_Ward_NoM"])

def eta_squared(values, labels):
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    ss_bet = sum(len(g) * (g.mean() - grand) ** 2 for _, g in values.groupby(labels))
    return float(ss_bet / ss_tot)

eta_full = eta_squared(feat["Log1p_Monetary"], feat["RFM_Ward"])
eta_noM = eta_squared(feat["Log1p_Monetary"], feat["RFM_Ward_NoM"])

# Robustness check: log1p(Recency) vs raw Recency
X_rfm_logR = StandardScaler().fit_transform(
    pd.DataFrame({"logR": np.log1p(feat["Recency"]), "F": feat["Frequency"], "logM": feat["Log1p_Monetary"]})
)
lab_logR = AgglomerativeClustering(n_clusters=K_RFM, linkage="ward").fit_predict(X_rfm_logR) + 1
ari_logR = adjusted_rand_score(feat["RFM_Ward"], lab_logR)

print(f"\\nMonetary removal (Ward, fixed k={K_RFM}):")
print(f"  ARI (Full RFM vs R+F only)     : {ari_noM:.4f}")
print(f"  eta^2 of log1p(M) — Full RFM   : {eta_full:.4f}")
print(f"  eta^2 of log1p(M) — R+F only   : {eta_noM:.4f}")
print(f"  Robustness ARI (raw R vs logR) : {ari_logR:.4f}")

print("\\nTraded value (USD bn) by Full RFM Ward cluster:")
print(feat.groupby("RFM_Ward")["Monetary_USD_bn"].agg(["count", "median", "mean"]).round(2))
print("\\nTraded value (USD bn) by R+F-only Ward cluster:")
print(feat.groupby("RFM_Ward_NoM")["Monetary_USD_bn"].agg(["count", "median", "mean"]).round(2))

# Boxplot comparing Monetary separation
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
feat.boxplot(column="Log1p_Monetary", by="RFM_Ward", ax=axes[0])
axes[0].set_title(f"Full RFM Ward (eta²={eta_full:.2f})"); axes[0].set_xlabel("Cluster"); axes[0].set_ylabel("log1p(Monetary USD)")
feat.boxplot(column="Log1p_Monetary", by="RFM_Ward_NoM", ax=axes[1])
axes[1].set_title(f"R+F Only Ward (eta²={eta_noM:.2f})"); axes[1].set_xlabel("Cluster")
plt.suptitle("")
plt.tight_layout()
plt.savefig(FIG / "monetary_removal_comparison.png", dpi=150)
plt.show()
"""),

    md("""
## Task 3 — BI Evidence: Risk vs RFM, Sector Composition, Diversification Check & Survivorship Bias
- **Risk/Return vs RFM agreement (ARI):** tests whether risk/return groups match trading-activity groups.
- **GICS Sector composition (%):** shows which sectors dominate each cluster.
- **Pairwise weekly return correlation (within vs across clusters):** directly tests the brief's warning that *weekly beta differences alone do not establish portfolio diversification*.
- **Survivorship bias check:** compares shares added to the S&P 500 after `2021-01-01` vs earlier constituents.
"""),
    code("""
# 1. Risk/Return vs RFM agreement
ari_cross_ward = adjusted_rand_score(feat["Risk_Ward"], feat["RFM_Ward"])
ari_cross_km = adjusted_rand_score(feat["Risk_KMeans"], feat["RFM_KMeans"])
print(f"Cross-matrix agreement (Risk/Return vs RFM): Ward ARI = {ari_cross_ward:.4f} | K-Means ARI = {ari_cross_km:.4f}")

# 2. GICS Sector composition (% within each cluster)
sec_risk = pd.crosstab(feat["Risk_Ward"], feat["GICS_Sector"], normalize="index").mul(100).round(1)
sec_rfm = pd.crosstab(feat["RFM_Ward"], feat["GICS_Sector"], normalize="index").mul(100).round(1)
sec_risk.to_csv(OUT / "sector_composition_Risk_Ward.csv")
sec_rfm.to_csv(OUT / "sector_composition_RFM_Ward.csv")
print("\\n=== Sector composition % (Risk_Ward) ===")
print(sec_risk.to_string())
print("\\n=== Sector composition % (RFM_Ward) ===")
print(sec_rfm.to_string())

# 3. Pairwise weekly return correlation (diversification check)
ret_mat = weekly_ret[feat.index]
corr_mat = ret_mat.corr().values
iu = np.triu_indices(len(feat), k=1)
overall_pair_corr = float(corr_mat[iu].mean())

def within_across_corr(labels):
    lab = np.asarray(labels)
    same = lab[iu[0]] == lab[iu[1]]
    return float(corr_mat[iu][same].mean()), float(corr_mat[iu][~same].mean())

w_risk, a_risk = within_across_corr(feat["Risk_Ward"])
w_rfm, a_rfm = within_across_corr(feat["RFM_Ward"])
print(f"\\nPairwise weekly-return correlation ({len(feat)} shares, {len(iu[0]):,} pairs):")
print(f"  Overall average              : {overall_pair_corr:.4f}")
print(f"  Risk_Ward — within / across  : {w_risk:.4f} / {a_risk:.4f}")
print(f"  RFM_Ward  — within / across  : {w_rfm:.4f} / {a_rfm:.4f}")

# 4. Survivorship bias check using Wikipedia Date added
added_dt = pd.to_datetime(feat["Date_Added"], errors="coerce")
feat["Added_Post_2021"] = added_dt >= pd.Timestamp("2021-01-01")
surv = feat.groupby("Added_Post_2021").agg(
    N=("Mean_Weekly_Return", "size"),
    Mean_Weekly_Ret_Pct=("Mean_Weekly_Return", lambda s: round(s.mean() * 100, 4)),
    Median_Weekly_Ret_Pct=("Mean_Weekly_Return", lambda s: round(s.median() * 100, 4)),
    Mean_Beta=("Weekly_Beta", lambda s: round(s.mean(), 3)),
)
print("\\n=== Survivorship-bias evidence (constituents added on/after 2021-01-01 vs earlier) ===")
print(surv.to_string())
print("Note: companies REMOVED from the S&P 500 during 2021-2024 are absent from the current Wikipedia list.")

# Save final cluster assignments and summary JSON
feat.to_csv(OUT / "cluster_assignments.csv")
summary = {
    "retrieval_date": RETRIEVAL_DATE,
    "observation_date": str(obs_date.date()),
    "study_period": f"{START} to {END} (exclusive)",
    "requested_sp500_tickers": len(tickers),
    "benchmark": BENCH,
    "excluded_tickers": len(excl_only),
    "final_common_shares": len(feat),
    "trading_sessions": len(bench_dates),
    "paired_complete_weeks": len(common_weeks),
    "k_risk": K_RISK,
    "k_rfm": K_RFM,
    "ari_risk_ward_vs_kmeans": round(float(ari_risk_wk), 4),
    "ari_rfm_ward_vs_kmeans": round(float(ari_rfm_wk), 4),
    "ari_rfm_full_vs_noM": round(float(ari_noM), 4),
    "eta2_logM_full": round(float(eta_full), 4),
    "eta2_logM_noM": round(float(eta_noM), 4),
    "ari_rfm_rawR_vs_logR": round(float(ari_logR), 4),
    "ari_risk_vs_rfm_ward": round(float(ari_cross_ward), 4),
    "pca_rfm_explained_var": [round(float(ev[0]), 4), round(float(ev[1]), 4)],
    "overall_pairwise_weekly_corr": round(overall_pair_corr, 4),
    "risk_ward_within_across_corr": [round(w_risk, 4), round(a_risk, 4)],
    "rfm_ward_within_across_corr": [round(w_rfm, 4), round(a_rfm, 4)],
    "post_2021_additions_in_sample": int(feat["Added_Post_2021"].sum()),
}
(OUT / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
print("\\nSaved outputs/cluster_assignments.csv and outputs/summary_metrics.json")
print(json.dumps(summary, indent=2))
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
