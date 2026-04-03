# Wharton-Investment-High-School-Comp-2025-models
# Wharton Investment Competition 2025 — Monte Carlo Portfolio Simulator

A production-grade Monte Carlo simulation engine built for the [Wharton Global High School Investment Competition](https://globalyouth.wharton.upenn.edu/investment-competition/). Simulates 500+ portfolio return paths over a 10-year horizon using historical calibration, real-time macroeconomic adjustments, and efficient frontier analysis.

Built with Python. GUI and CLI modes.

---

## What It Does

Given a portfolio of stocks, bonds, and ETFs, the simulator:

1. **Downloads and sanitizes** historical price data via yfinance — removing outliers, extreme returns, and low-quality quotes
2. **Calibrates return distributions** using a hybrid approach: standardized efficient frontier parameters for known assets, historical log returns for custom portfolios
3. **Adjusts for macroeconomic conditions** — fetches real-time Shiller CAPE ratios and 10-year inflation expectations (FRED API) to scale expected returns
4. **Applies regime weighting** — downweights pandemic-era returns (2020–2021) and upweights recent market conditions to reduce tail risk bias
5. **Runs Monte Carlo simulation** — samples from a multivariate normal distribution preserving cross-asset correlations, generating 500 simulated portfolio paths
6. **Calculates risk metrics** — median CAGR, annualized volatility, Sharpe ratio, probability of hitting a +30% goal, probability of a -10% bust, and probability of exceeding a 12.5% annual return target
7. **Generates visualizations and CSV outputs** — ensemble plots for the portfolio and each individual asset, plus detailed summary statistics

---

## Key Features

**Hybrid Return Estimation** — Uses standardized efficient frontier parameters when available, falls back to historical calibration for custom tickers, and blends both when the portfolio partially overlaps the standard set.

**Real-Time Macro Adjustments** — Scrapes the current Shiller CAPE ratio from [multpl.com](https://www.multpl.com/shiller-pe) and fetches 10-year inflation breakevens from FRED. Equity drift is scaled by a valuation factor; bond returns use forward yield curves.

**Data Quality Pipeline** — Forward-fills missing prices, clips extreme daily log returns (>±1.5), removes quotes below $0.50, and requires a minimum 252-day overlapping history window. Pandemic-period returns are further clipped at the 5th–95th percentile.

**Regime Weighting** — Pre-2018 returns weighted at 0.85×, pandemic period at 0.45×, post-2022 at 1.15× — reflecting the structural shift in rate environments.

---

## Usage

### GUI Mode
```bash
python "montecarlo wharton.py" --gui
```
Launches a Tkinter interface for editing the portfolio, running simulations, and viewing results inline.

### CLI Mode
```bash
python "montecarlo wharton.py" \
  --tickers INSM GILD TEF ESRT KR AMD ADBE VOO IEF \
  --shares 145 38 896 612 511 826 14 82 45 \
  --sims 500 \
  --years 10 \
  --start 2016-01-01 \
  --end 2025-01-01 \
  --run-name "final_portfolio"
```

### Optional Flags
```
--cash 5000              # Starting cash balance
--asset-types stock bond # Override automatic asset classification
```

---

## Output

Each run generates a directory `montecarlo_portfolio_data_{run_name}/` containing:

| File | Description |
|------|-------------|
| `portfolio_montecarlo_{name}.csv` | All 500 simulated portfolio paths (days × sims) |
| `portfolio_summary_{name}.csv` | Median/mean/5th/95th end values, CAGR, volatility, goal/bust probabilities, Sharpe ratio |
| `asset_assumptions_{name}.csv` | Per-asset: weight, drift (historical vs. adjusted), valuation scalar, classification source |
| `top_paths_summary_{name}.csv` | Top 10 performing scenarios with detailed statistics |
| `portfolio_montecarlo_pct_top10_{name}.png` | Portfolio ensemble plot with median, top paths, goal/bust thresholds |
| `stock_plots/{TICKER}_montecarlo_pct_{name}.png` | Individual asset ensemble plots |

---

## Architecture

```
Historical Prices (yfinance)
        │
        ▼
  Data Sanitization ──── Remove outliers, forward-fill, clip extremes
        │
        ▼
  Return Calibration ─── Hybrid: efficient frontier + historical log returns
        │
        ▼
  Macro Adjustment ───── CAPE ratio (equities) + FRED inflation expectations
        │
        ▼
  Regime Weighting ───── Time-period-based return scaling
        │
        ▼
  Monte Carlo Engine ─── Multivariate normal sampling (correlated returns)
        │
        ▼
  Portfolio Aggregation ─ Price paths × share counts + cash
        │
        ▼
  Risk Metrics ───────── CAGR, volatility, Sharpe, goal/bust probabilities
        │
        ▼
  Visualization ──────── Ensemble plots + CSV exports
```

---

## Dependencies

```
numpy
pandas
yfinance
matplotlib
requests
```

GUI mode additionally requires `tkinter` (included with most Python installations).

---

## Competition Context

The [Wharton Investment Competition](https://globalyouth.wharton.upenn.edu/investment-competition/) challenges high school teams to construct and defend a diversified portfolio. This simulator was built to stress-test portfolio allocations under thousands of possible market scenarios — quantifying downside risk and return probability before submission.

The default portfolio includes 20 assets spanning U.S. equities, international stocks, bonds, REITs, and sector ETFs — designed for diversification across asset classes, geographies, and market regimes.

---

## License

MIT
