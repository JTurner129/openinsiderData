# Insider-buying signal backtest

This fork adds a full backtest of OpenInsider-style insider-buying screens on top of the
[sd3v/openinsiderData](https://github.com/sd3v/openinsiderData) scraper.

**Open `report.html` in a browser** for the results: no setup needed, charts are interactive.
All chart images are in `results/charts/`, every summary table is in `results/` (CSV) and
`results/results.xlsx` (one sheet per table).

## Why SEC data instead of scraping openinsider.com

OpenInsider builds its tables from SEC Form 4 filings. The scraper could not reach
openinsider.com from the cloud environment this was built in (plain HTTP is blocked and the
site drops HTTPS connections), so Phase 1 downloads the same filings straight from the SEC's
[Insider Transactions Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets)
with `sec_insider_data.py`. It reads the same `config.yaml` keys as `openinsider_scraper.py`
and writes the same output columns (`transaction_date` = filing date, `trade_date`, `ticker`,
`company_name`, `owner_name`, `Title`, `transaction_type`, `last_price`, `Qty`, `shares_held`,
`Owned`, `Value`) plus SEC identifiers (`issuer_cik`, `owner_cik`, `accession_number`, ...).
The scraper itself is kept and fixed (pagination, sale filtering, purchases-only request) if
you want to run it from a machine that can reach openinsider.com.

The SEC bulk data starts in **2006 Q1**; that is the earliest month this pipeline returns rows.

## Re-running

```bash
pip install -r requirements.txt -r requirements-backtest.txt
export SEC_USER_AGENT="your-name your-email@example.com"   # SEC requires a contact
python run_backtest.py          # all phases
python run_backtest.py 3 4 5 6  # just some phases, re-using earlier outputs
python -m pytest tests          # title regex and fund-filter tests
node tests/check_report.mjs     # opens report.html in headless Chromium; fails on empty charts/tables
```

The Yahoo price panel (`data/prices.parquet`, `data/prices.csv.gz`) is **not committed**: it is
~190 MB and Yahoo's terms do not allow redistributing it. `python run_backtest.py 3` rebuilds it
(about 30-60 minutes; Yahoo throttles, and the downloader backs off automatically). Everything
else in `data/` and `results/` is committed. The largest CSVs are gzipped (`.csv.gz`) to stay
under GitHub's file-size limit; pandas and Excel's Power Query read them directly.

| Phase | Command | What it does | Output |
|---|---|---|---|
| 1 Data | `python run_backtest.py 1` | Downloads every SEC quarterly Form 4 data set, keeps code `P` purchases, one row per filing | `data/insider_trades.parquet` (+ `.csv.gz`), `data/scrape_summary.json` |
| 2 Signals | `python run_backtest.py 2` | Cleans values, looks up SIC codes, builds the five screens and de-duplicates | `data/signals.*`, `data/signals_all_variants.*`, `data/excluded_funds_trusts.*` |
| 3 Prices | `python run_backtest.py 3` | Adjusted daily prices from Yahoo for every signal ticker, each company's current SEC ticker, SPY and IWM | `data/prices.parquet`, `data/prices.csv.gz` (not committed), `data/price_cleaning.json`, `data/alternate_tickers.json` |
| 4 Returns | `python run_backtest.py 4` | Per-signal returns at 1w/1m/3m/6m/1y, excess vs SPY and IWM, after costs | `data/trade_returns.*`, `data/trade_returns_all_variants.*` |
| 5 Analysis | `python run_backtest.py 5` | Event study, calendar-time portfolios, CAPM and FF5+momentum alphas, robustness | `results/*.csv`, `data/portfolio_daily_returns.*` |
| 6 Report | `python run_backtest.py 6` | Builds `report.html`, PNG charts and `results/results.xlsx` | `report.html`, `results/` |

Every download is cached in `.cache/` (SEC quarters, SEC company info, Yahoo prices, French
factors), so an interrupted run resumes where it stopped. Delete a cache folder to force a refresh.
PNG export needs Chrome/Chromium; set `BROWSER_PATH` if it is not found automatically.

All parameters are in `config.yaml` under `backtest:` (window, thresholds, de-dupe days,
horizons, holding periods, costs, liquidity filter, price-check tolerance).

## Screens

| Key | Screen | Rule |
|---|---|---|
| `cluster3` | Cluster buys | ≥3 distinct insiders and ≥$200K total within a rolling 30-day filing-date window; funds/trusts excluded |
| `exec200k` | Top executives | One purchase filing ≥$200K by a CEO, CFO or Chairman of the Board |
| `exec500k` | Top executives, tighter | Same, ≥$500K |
| `cluster5` | Tighter clusters | Cluster rule with ≥5 insiders |
| `cluster10` | Tighter clusters | Cluster rule with ≥10 insiders |

Each company can signal at most once per screen every 90 days (`dedupe_days`).

## Assumptions

**Data**
- Source: SEC Form 4 originals (`DOCUMENT_TYPE` = 4). Amendments (4/A) are excluded so a corrected filing is not counted twice. Form 5 annual reports are excluded, as on OpenInsider's screens.
- Purchases = transaction code `P` with acquired/disposed flag `A`, from the non-derivative table.
- One row per filing (OpenInsider's convention): quantities and values summed across the filing's lines, price = value-weighted average, trade date = earliest line, shares held = after the last line, `Owned` = quantity ÷ holdings before (blank = new position).
- The insider is the first listed reporting owner. Joint filings (e.g. a person and their LLC) count as one insider. Distinct insiders are counted by SEC owner CIK.
- Tickers come from the filing as reported, cleaned of exchange prefixes and OTC suffixes. Companies are tracked by SEC issuer CIK, so ticker changes do not split a cluster.
- Value cleaning: a purchase price more than 10× away from the median price of the same company's other purchases within ±30 days is treated as a typo and the value is recomputed at that median. Values above $1B after that are excluded from dollar totals (almost all are typos; a handful of genuine $1B+ purchases are lost, which does not change any signal because one such buyer already clears the $200K bar).
- `min_transaction_value` and `min_shares_traded` are 0, so small buys still count toward cluster totals.

**Signals**
- Signal date = SEC filing date, never the trade date. The SEC dates filings received after 5:30pm ET as the next business day, so the next day's open is tradeable.
- Cluster signals fire on the first filing date the 30-day window meets both thresholds. The same cluster cannot fire again until the condition has lapsed for a full window; then the 90-day de-dupe applies on top.
- Fund/trust filter (cluster screens): SIC code from `data.sec.gov/submissions/CIK##########.json`; drop SIC 6720–6799 (note: this includes REITs, SIC 6798) or a name containing fund / ETF / trust / closed-end / municipal, except banks (SIC 6000–6199, e.g. "Northern Trust"). The dropped list is in `data/excluded_funds_trusts.csv`.
- Top-executive titles: regex in `backtest/signals.py` (tested in `tests/test_title_regex.py`) matching CEO / chief executive, CFO / chief financial / principal financial officer, and COB / chairman (including executive and co-chairman), excluding vice and deputy chairs and "former" roles. Exec screens are not fund-filtered (the request applied that filter to clusters only).

**Prices and returns**
- Yahoo Finance via `yfinance`, adjusted for splits and dividends. Adjusted open = open × (adjusted close ÷ close).
- Entry = adjusted open of the first trading day after the filing date (close if no open, flagged in `entry_used_close`). Exit = adjusted close N trading days later (N = 5, 21, 63, 126, 252) on the SPY calendar. Benchmarks use the identical entry and exit dates.
- A horizon is reported only once it has fully elapsed; recent signals have partial horizons (`partial_horizons` = True).
- If a stock stops trading before the horizon, its last close is used and `truncated_<h>` is set.
- Ticker check: Yahoo's split-unadjusted price at entry must be within 2× of the insiders' average purchase price. Otherwise the ticker probably belongs to a different company now, and the signal is marked `price_mismatch` (missing), not used.
- Ticker changes: if the ticker on the filing fails, the company's current tickers from the SEC (`company_tickers.json`, by CIK) are tried in turn, with the same entry and price-match checks. `price_ticker` and `ticker_source` (`filing` or `sec_current`) record which was used; the report shows e.g. `OLD → NEW`.
- Bad prints in Yahoo data are cleaned when prices are loaded (`clean_bad_prints` in `backtest/prices.py`): days with a zero price are dropped; a one-day move of ≥5× that reverses by ≥5× the next day is replaced by the prior close; a one-day **rise** of ≥10× that persists is treated as an unadjusted reverse split and earlier history is rescaled; opens more than 5× from the day's close are replaced by the close. Drops are never "corrected", so real crashes always count. Counts are in `data/price_cleaning.json` and the report.
- Signals without prices are never silently dropped: every signal is in `data/trade_returns.*` with a `price_status` (`ok`, `no_ticker`, `ticker_not_found`, `no_price_at_entry`, `price_mismatch`, `not_yet_entered`), and the report shows the % missing per screen per year.
- Costs: 0.5% round trip, applied as (1 + r) × (1 − 0.005) − 1.
- Liquidity filter: off by default (`min_entry_price: 0`); the robustness section shows the effect of requiring a price ≥ $2 at entry.

**Analysis**
- Event-study t-stats: standard errors clustered by signal month.
- Calendar-time portfolios: each trading day, equal weight in every stock still within its holding period (1m, 3m, 6m, 1y), entry-day return = close ÷ open. Days with no positions earn the risk-free rate.
- Sharpe and Sortino use the Fama-French risk-free rate. Alphas regress daily portfolio excess returns on Mkt-RF (CAPM) and on Mkt-RF, SMB, HML, RMW, CMA and UMD from Kenneth French's data library, with Newey-West (5 lag) standard errors, annualized ×252. The French data ends a month or two before today, so the regressions stop there.
- Eras: pre-2010 (2006–2009), 2010–2019, 2020+.
- De-dupe windows of 0 and 30 days give identical cluster signals, because a cluster can only fire again after its 30-day condition has lapsed anyway.

## Known limitations

- **Survivorship bias (the big one).** Yahoo lacks most delisted stocks. Only about 40-55% of signals could be priced overall, and under 30% for 2006-2011. Those signals are reported as missing, not dropped, but the priced sample leans toward survivors, which flatters returns. For a cleaner test, swap Phase 3 to a source with delisted history: Norgate Data (paid), Tiingo or EODHD (API key; free tiers are too small for ~8,000 tickers), or CRSP via WRDS (academic). Only `backtest/prices.py` needs to change: write the same per-ticker Parquet files (`adj_open`, `adj_close`, `raw_close`, `volume`) into `.cache/prices/`.
- **Microcaps.** Many signals are small, illiquid stocks. Requiring a price of at least $2 at entry cuts alpha substantially (see the report's robustness section). Equal-weight daily rebalancing of illiquid stocks also overstates achievable returns (bid-ask bounce), and 0.5% round-trip costs are optimistic for the smallest names.
- SEC bulk data starts in 2006, so 2003–2005 (which OpenInsider partly covers) is not included.
- Yahoo data occasionally has bad prints; no outlier trimming is applied to returns (the report shows means and medians side by side).

---

# Original scraper documentation

# 📈 OpenInsider Data Scraper

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

A robust Python scraper for collecting insider trading data from openinsider.com.

## ✨ Features

- Multi-threaded data collection for high performance
- Intelligent caching system to minimize server load
- Configurable filters for transaction types and values
- Flexible data export in CSV and Parquet formats
- Comprehensive logging and error handling
- Automatic retry mechanism for failed requests
- Progress tracking with progress bar
- Docker support for easy deployment

## 🚀 Installation

1. Clone the repository:
```bash
git clone git@github.com:sd3v/openinsiderData.git
cd openinsiderData
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

All settings are managed through `config.yaml`:

### 📁 Output Settings
```yaml
output:
  directory: data       # Output directory for scraped data
  filename: insider     # Base filename for output files
  format: csv          # Output format (csv or parquet)
```

### 🔄 Scraping Settings
```yaml
scraping:
  start_year: 2024           # Start year
  start_month: 3             # Start month
  max_workers: 10            # Number of parallel downloads
  retry_attempts: 3          # Number of retry attempts
  timeout: 30               # Request timeout in seconds
```

### 🔎 Filter Settings
```yaml
filters:
  min_transaction_value: 50000  # Minimum transaction value in USD
  transaction_types:            # Transaction types to include
    - P - Purchase
    - S - Sale
    - F - Tax
  exclude_companies: []         # Companies to exclude (by ticker)
  min_shares_traded: 100        # Minimum number of shares
```

### 📝 Logging Settings
```yaml
logging:
  level: INFO          # Logging level (DEBUG, INFO, WARNING, ERROR)
  file: scraper.log    # Log file name
  rotate_logs: true    # Enable log rotation
  max_log_size: 10     # Max log size in MB
```

### 💾 Cache Settings
```yaml
cache:
  enabled: true        # Enable caching
  directory: .cache    # Cache directory
  max_age: 24         # Cache max age in hours
```

## 🔧 Usage

Run the scraper:
```bash
python openinsider_scraper.py
```

## 🐳 Docker Support

Build the container:
```bash
docker build -t openinsider-scraper .
```

Run the container:
```bash
docker run -v $(pwd)/data:/app/data openinsider-scraper
```

## 💼 Transaction Types

Available transaction types:
- P - Purchase
- S - Sale
- F - Tax
- D - Disposition
- G - Gift
- X - Exercise
- M - Options Exercise
- C - Conversion
- W - Will/Inheritance
- H - Holdings
- O - Other

## 👥 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 🔍 Troubleshooting

- If you encounter rate limiting, adjust the `max_workers` setting
- For memory issues, try using Parquet format for large datasets
- Check the log file for detailed error messages

## ⚠️ Disclaimer

This tool is for educational purposes only. Ensure you comply with the website's terms of service and local regulations when scraping data.
