from __future__ import annotations

import argparse
import io
import re
import os
import subprocess
import sys
from dataclasses import dataclass
import warnings
from threading import Thread
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from urllib3.exceptions import InsecureRequestWarning
from matplotlib import cm
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.figure import Figure

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


DEFAULT_RUN_NAME = "final port"
DEFAULT_TICKERS = [
    "INSM",
    "GILD",
    "TEF",
    "0LQQ.L",
    "ESRT",
    "KR",
    "CHD",
    "BMW.DE",
    "PPG",
    "KGC",
    "FSLR",
    "HASI",
    "NEE",
    "INTC",
    "AMD",
    "ADBE",
    "PRU",
    "AIG",
    "VOO",
    "IEF",
]
DEFAULT_SHARES = [
    145,
    38,
    896,
    377,
    612,
    511,
    52,
    56,
    46,
    4671,
    17,
    163,
    55,
    122,
    826,
    14,
    44,
    57,
    82,
    45,
]
DEFAULT_CASH = 0
DEFAULT_SIMS = 500
DEFAULT_YEARS = 10
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2025-01-01"

MIN_SENSIBLE_PRICE = 0.5
MAX_ABS_DAILY_LOG_RETURN = 1.5
MIN_REQUIRED_HISTORY_DAYS = 252

# Efficient frontier reference inputs (annual arithmetic returns & covariance).
STANDARDIZED_EXPECTED_RETURNS = {
    "INSM": 0.112029,
    "GILD": 0.03187,
    "TEF": -0.008857,
    "0LQQ": -0.030114,
    "ESRT": -0.039601,
    "KR": 0.047721,
    "CHD": 0.036731,
    "BMW": 0.015406,
    "PPG": -0.002983,
    "KGC": 0.103723,
    "FSLR": 0.076962,
    "HASI": 0.033878,
    "NEE": 0.051056,
    "INTC": -0.002474,
    "AMD": 0.186002,
    "ADBE": 0.028919,
    "PRU": 0.009583,
    "AIG": 0.021661,
    "VOO": 0.055211,
    "UST10Y": 0.014106,
}

_COV_COLUMNS = [
    "INSM",
    "GILD",
    "TEF",
    "0LQQ",
    "ESRT",
    "KR",
    "CHD",
    "BMW",
    "PPG",
    "KGC",
    "FSLR",
    "HASI",
    "NEE",
    "INTC",
    "AMD",
    "ADBE",
    "PRU",
    "AIG",
    "VOO",
    "UST10Y",
]

STANDARDIZED_COVARIANCE = pd.DataFrame(
    [
        [0.476837, 0.026130, 0.031504, 0.007136, 0.044821, 0.005141, 0.008921, 0.023848, 0.038858, 0.020600, 0.059792, 0.067719, 0.028483, 0.059410, 0.079600, 0.058077, 0.052750, 0.047978, 0.042771, 0.000000],
        [0.026130, 0.079059, 0.016690, 0.004624, 0.015068, 0.012295, 0.014906, 0.008295, 0.016097, 0.009382, 0.013019, 0.017598, 0.014723, 0.026749, 0.021077, 0.021269, 0.024236, 0.018294, 0.017214, 0.000000],
        [0.031504, 0.016690, 0.087302, 0.022226, 0.036054, 0.009224, 0.009176, 0.021834, 0.023397, 0.019016, 0.022502, 0.029444, 0.018283, 0.027237, 0.023804, 0.018092, 0.038527, 0.034440, 0.020768, 0.000000],
        [0.007136, 0.004624, 0.022226, 0.089807, 0.018495, 0.002402, 0.002095, 0.013018, 0.011343, 0.009471, 0.011764, 0.018238, 0.006031, 0.006426, 0.005699, -0.000160, 0.013792, 0.013957, 0.006042, 0.000000],
        [0.044821, 0.015068, 0.036054, 0.018495, 0.151850, 0.004104, 0.006362, 0.031144, 0.043509, 0.012426, 0.039437, 0.060007, 0.025602, 0.040742, 0.035658, 0.027872, 0.065287, 0.063444, 0.033205, 0.000000],
        [0.005141, 0.012295, 0.009224, 0.002402, 0.004104, 0.092058, 0.016081, -0.001651, 0.006473, 0.009081, 0.006922, 0.008868, 0.011081, 0.011156, 0.006326, 0.008907, 0.009846, 0.007917, 0.008470, 0.000000],
        [0.008921, 0.014906, 0.009176, 0.002095, 0.006362, 0.016081, 0.065683, 0.002748, 0.012513, 0.010073, 0.005753, 0.013075, 0.019220, 0.014264, 0.010574, 0.015022, 0.011464, 0.008525, 0.012216, 0.000000],
        [0.023848, 0.008295, 0.021834, 0.013018, 0.031144, -0.001651, 0.002748, 0.093809, 0.028650, 0.005098, 0.023232, 0.028023, 0.010451, 0.024771, 0.025642, 0.013873, 0.038387, 0.034179, 0.018266, 0.000000],
        [0.038858, 0.016097, 0.023397, 0.011343, 0.043509, 0.006473, 0.012513, 0.028650, 0.091429, 0.014840, 0.035842, 0.041815, 0.022309, 0.042613, 0.044414, 0.032514, 0.051336, 0.048034, 0.032136, 0.000000],
        [0.020600, 0.009382, 0.019016, 0.009471, 0.012426, 0.009081, 0.010073, 0.005098, 0.014840, 0.204265, 0.020852, 0.039764, 0.022764, 0.022121, 0.033936, 0.014093, 0.011460, 0.013406, 0.015676, 0.000000],
        [0.059792, 0.013019, 0.022502, 0.011764, 0.039437, 0.006922, 0.005753, 0.023232, 0.035842, 0.020852, 0.244338, 0.087551, 0.035600, 0.046667, 0.069933, 0.042312, 0.041644, 0.037364, 0.033572, 0.000000],
        [0.067719, 0.017598, 0.029444, 0.018238, 0.060007, 0.008868, 0.013075, 0.028023, 0.041815, 0.039764, 0.087551, 0.200709, 0.046011, 0.048023, 0.062415, 0.045437, 0.051531, 0.049196, 0.037572, 0.000000],
        [0.028483, 0.014723, 0.018283, 0.006031, 0.025602, 0.011081, 0.019220, 0.010451, 0.022309, 0.022764, 0.035600, 0.046011, 0.085255, 0.025465, 0.026862, 0.023564, 0.027381, 0.025472, 0.022201, 0.000000],
        [0.059410, 0.026749, 0.027237, 0.006426, 0.040742, 0.011156, 0.014264, 0.024771, 0.042613, 0.022121, 0.046667, 0.048023, 0.025465, 0.183538, 0.085734, 0.055909, 0.052582, 0.043036, 0.043307, 0.000000],
        [0.079600, 0.021077, 0.023804, 0.005699, 0.035658, 0.006326, 0.010574, 0.025642, 0.044414, 0.033936, 0.069933, 0.062415, 0.026862, 0.085734, 0.280906, 0.085069, 0.050059, 0.043260, 0.055924, 0.000000],
        [0.058077, 0.021269, 0.018092, -0.000160, 0.027872, 0.008907, 0.015022, 0.013873, 0.032514, 0.014093, 0.042312, 0.045437, 0.023564, 0.055909, 0.085069, 0.133195, 0.035911, 0.032066, 0.041589, 0.000000],
        [0.052750, 0.024236, 0.038527, 0.013792, 0.065287, 0.009846, 0.011464, 0.038387, 0.051336, 0.011460, 0.041644, 0.051531, 0.027381, 0.052582, 0.050059, 0.035911, 0.121505, 0.082276, 0.042059, 0.000000],
        [0.047978, 0.018294, 0.034440, 0.013957, 0.063444, 0.007917, 0.008525, 0.034179, 0.048034, 0.013406, 0.037364, 0.049196, 0.025472, 0.043036, 0.043260, 0.032066, 0.082276, 0.127504, 0.037229, 0.000000],
        [0.042771, 0.017214, 0.020768, 0.006042, 0.033205, 0.008470, 0.012216, 0.018266, 0.032136, 0.015676, 0.033572, 0.037572, 0.022201, 0.043307, 0.055924, 0.041589, 0.042059, 0.037229, 0.054809, 0.000000],
        [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    ],
    index=_COV_COLUMNS,
    columns=_COV_COLUMNS,
)

STANDARDIZED_ALIAS_MAP = {
    "0LQQ.L": "0LQQ",
    "BMW.DE": "BMW",
    "IEF": "UST10Y",
    "UST10Y": "UST10Y",
}

STANDARDIZED_RISK_FREE = STANDARDIZED_EXPECTED_RETURNS["UST10Y"]

STANDARDIZED_REFERENCE_PRICES = {
    "INSM": 189.60,
    "GILD": 119.79,
    "TEF": 5.05,
    "0LQQ": 12.010689655172413,
    "ESRT": 7.39,
    "KR": 63.63,
    "CHD": 87.69,
    "BMW": 80.72,
    "PPG": 97.75,
    "KGC": 23.24,
    "FSLR": 266.94,
    "HASI": 27.71,
    "NEE": 81.40,
    "INTC": 39.99,
    "AMD": 256.12,
    "ADBE": 340.31,
    "PRU": 104.00,
    "AIG": 78.96,
    "VOO": 627.04,
    "UST10Y": 100.00,
}

@dataclass
class SimulationResult:
    output_dir: str
    portfolio_plot_path: str
    portfolio_csv_path: str
    summary_path: str
    asset_assumptions_path: str
    stock_csv_paths: List[str]
    stock_plot_paths: List[str] 
    figure: Figure
    summary: pd.DataFrame
    goal_probability: float
    bust_probability: float
    median_cagr: float
    median_annual_volatility: float
    initial_portfolio_value: float
    goal_threshold: float
    bust_threshold: float


def _extract_close_prices(raw_data: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    try:
        close_data = raw_data["Adj Close"]
    except KeyError as exc:
        raise ValueError("Downloaded data does not contain 'Adj Close' prices.") from exc

    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=tickers[0])

    close_data = close_data.dropna(how="all")
    missing = [ticker for ticker in tickers if ticker not in close_data.columns]
    if missing:
        raise ValueError(f"Missing close prices for tickers: {', '.join(missing)}")

    close_data = close_data.loc[:, tickers].sort_index()

    sanitization_notes: List[str] = []

    def _sanitize_prices(series: pd.Series) -> pd.Series:
        ticker = series.name
        cleaned = pd.to_numeric(series, errors="coerce").astype(float)
        modifications: List[str] = []

        non_positive_mask = (cleaned <= 0) & (~cleaned.isna())
        if non_positive_mask.any():
            cleaned.loc[non_positive_mask] = np.nan
            modifications.append(f"removed {int(non_positive_mask.sum())} non-positive quotes")

        cleaned = cleaned.mask(~np.isfinite(cleaned))

        low_mask = (cleaned < MIN_SENSIBLE_PRICE) & (~cleaned.isna())
        if low_mask.any():
            cleaned.loc[low_mask] = np.nan
            modifications.append(
                f"dropped {int(low_mask.sum())} quotes below ${MIN_SENSIBLE_PRICE:.2f}"
            )

        log_prices = np.log(cleaned)
        log_returns = log_prices.diff()
        extreme_mask = (log_returns.abs() > MAX_ABS_DAILY_LOG_RETURN) & (~log_returns.isna())
        if extreme_mask.any():
            extreme_indices = log_returns.index[extreme_mask]
            cleaned.loc[extreme_indices] = np.nan

            positions = cleaned.index.get_indexer(extreme_indices)
            prev_positions = positions[positions > 0] - 1
            if prev_positions.size > 0:
                cleaned.loc[cleaned.index[prev_positions]] = np.nan

            modifications.append(
                f"removed {len(extreme_indices)} days with daily moves beyond ±{MAX_ABS_DAILY_LOG_RETURN:.1f} log-units"
            )

        if modifications:
            sanitization_notes.append(f"{ticker}: " + "; ".join(modifications))

        return cleaned

    close_data = close_data.apply(_sanitize_prices)

    close_data = close_data.dropna(how="all")
    close_data = close_data.ffill()
    close_data = close_data.dropna(how="any")
    if close_data.empty:
        raise ValueError("No overlapping adjusted close price history after cleaning.")

    if len(close_data) < MIN_REQUIRED_HISTORY_DAYS:
        warnings.warn(
            f"Only {len(close_data)} trading days of overlapping history remain after cleaning; "
            "simulation results may be less reliable.",
            RuntimeWarning,
        )

    for note in sanitization_notes:
        warnings.warn(note, RuntimeWarning)

    return close_data


def _ensure_positive_semidefinite(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Clamp tiny negative eigenvalues produced by numerical noise."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, floor, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _canonicalize_ticker(ticker: str) -> str:
    upper = ticker.upper()
    return STANDARDIZED_ALIAS_MAP.get(upper, upper)

def _prepare_standardized_inputs(
    tickers: Sequence[str],
    trading_days: int,
) -> Optional[Dict[str, object]]:
    canonical: List[str] = []
    for ticker in tickers:
        alias = _canonicalize_ticker(ticker)
        if alias not in STANDARDIZED_EXPECTED_RETURNS or alias not in STANDARDIZED_COVARIANCE.index:
            return None
        canonical.append(alias)

    expected_annual = np.array(
        [STANDARDIZED_EXPECTED_RETURNS[label] for label in canonical],
        dtype=float,
    )

    covariance_annual = STANDARDIZED_COVARIANCE.loc[canonical, canonical].to_numpy(dtype=float)
    covariance_daily_arith = covariance_annual / trading_days
    covariance_daily_log = _ensure_positive_semidefinite(covariance_daily_arith)

    daily_arith = np.power(1.0 + expected_annual, 1.0 / trading_days) - 1.0
    diag_daily_var = np.clip(np.diag(covariance_daily_log), 0.0, None)
    daily_log_mean = np.log1p(daily_arith) - 0.5 * diag_daily_var

    return {
        "canonical": canonical,
        "annual_expected": expected_annual,
        "annual_covariance": covariance_annual,
        "daily_log_mean": daily_log_mean,
        "daily_log_covariance": covariance_daily_log,
    }



def _compute_valuation_adjustment() -> float:
    """Return a drift scaling factor based on the latest Shiller CAPE ratio."""
    cape_page = "https://www.multpl.com/shiller-pe"
    fallback_scalar = 0.6
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            response = requests.get(cape_page, timeout=15, verify=False)
        response.raise_for_status()
    except Exception:
        return fallback_scalar

    text = response.text
    current_match = re.search(r"Current Shiller PE Ratio is ([0-9]+\.?[0-9]*)", text)
    mean_match = re.search(r"Mean:\s*</td>\s*<td>([0-9]+\.?[0-9]*)", text)
    if not current_match or not mean_match:
        return fallback_scalar

    current_cape = float(current_match.group(1))
    long_run_mean = float(mean_match.group(1))
    blended_mean = 0.5 * long_run_mean + 0.5 * max(long_run_mean, 24.0)
    if current_cape <= 0 or blended_mean <= 0:
        return fallback_scalar

    adjustment = blended_mean / current_cape
    return float(np.clip(adjustment, 0.4, 1.15))


def _fetch_market_implied_inflation(default: float = 0.0225) -> Tuple[float, str]:
    """Fetch the latest 10y breakeven inflation (T10YIE) from FRED."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception:
        return default, "Fallback constant"

    try:
        data = pd.read_csv(io.StringIO(response.text))
    except Exception:
        return default, "Fallback constant"

    if "T10YIE" not in data.columns:
        return default, "Fallback constant"

    data = data.dropna(subset=["T10YIE"])
    if data.empty:
        return default, "Fallback constant"

    latest = data.iloc[-1]["T10YIE"]
    try:
        latest_value = float(latest)
    except (TypeError, ValueError):
        return default, "Fallback constant"

    if not np.isfinite(latest_value):
        return default, "Fallback constant"

    return float(np.clip(latest_value / 100.0, 0.0, 0.06)), "FRED T10YIE"


def _normalize_asset_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    text = label.strip().lower()
    if not text:
        return None
    text = text.replace("-", " ")
    if "bond" in text or "fixed income" in text or "treasury" in text:
        return "bond"
    if "stock" in text or "equity" in text:
        return "equity"
    if "other" in text:
        return "other"
    if "etf" in text:
        return "etf"
    return text


def _classify_asset_via_metadata(metadata: Dict[str, object], fallback: Optional[str]) -> str:
    if fallback == "etf":
        fallback = None

    text_parts: List[str] = []
    for key in (
        "category",
        "quoteType",
        "longName",
        "shortName",
        "displayName",
        "fundFamily",
        "fundCategory",
        "assetClass",
    ):
        value = metadata.get(key)
        if isinstance(value, str):
            text_parts.append(value.lower())

    combined = " ".join(text_parts)
    bond_keywords = (
        "bond",
        "fixed income",
        "treasury",
        "municipal",
        "muni",
        "tips",
        "agency",
        "mortgage",
    )
    if any(keyword in combined for keyword in bond_keywords):
        return "bond"

    quote_type = metadata.get("quoteType")
    if isinstance(quote_type, str):
        quote_type_lower = quote_type.lower()
        if quote_type_lower == "bond":
            return "bond"
        if quote_type_lower == "equity":
            return "equity"

    if fallback in {"bond", "equity", "other"}:
        return fallback

    return "equity"


def _infer_asset_classes(
    tickers: Sequence[str],
    overrides: Optional[Sequence[Optional[str]]],
) -> Tuple[List[str], Dict[str, Dict[str, object]], List[str]]:
    metadata_cache: Dict[str, Dict[str, object]] = {}
    classes: List[str] = []
    sources: List[str] = []

    def fetch_metadata(ticker: str) -> Dict[str, object]:
        if ticker not in metadata_cache:
            try:
                metadata_cache[ticker] = yf.Ticker(ticker).info
            except Exception:
                metadata_cache[ticker] = {}
        return metadata_cache[ticker]

    for idx, ticker in enumerate(tickers):
        override = overrides[idx] if overrides and idx < len(overrides) else None
        normalized = _normalize_asset_label(override)
        metadata = fetch_metadata(ticker)

        if normalized in {"bond", "equity", "other"}:
            classes.append(normalized)
            sources.append("user")
            continue

        classification = _classify_asset_via_metadata(metadata, normalized)
        if classification not in {"bond", "equity", "other"}:
            classification = "equity"

        classes.append(classification)
        sources.append("metadata" if metadata else "fallback")

    return classes, metadata_cache, sources


def _estimate_bond_forward_nominal_return(metadata: Dict[str, object]) -> Optional[float]:
    candidate_keys = ("yield", "secYield", "trailingAnnualDividendYield", "fiveYearAverageReturn")
    for key in candidate_keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        if numeric_value > 1.5:
            numeric_value /= 100.0
        if -0.05 <= numeric_value <= 0.25:
            return float(numeric_value)
    return None





def run_simulation(
    tickers: Sequence[str],
    shares_owned: Sequence[int],
    asset_types: Optional[Sequence[str]] = None,
    cash: float = DEFAULT_CASH,
    sims: int = DEFAULT_SIMS,
    years: float = DEFAULT_YEARS,
    run_name: str = DEFAULT_RUN_NAME,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
) -> SimulationResult:
    if len(tickers) != len(shares_owned):
        raise ValueError("Tickers and shares counts must match.")
    if asset_types is not None and len(asset_types) != len(tickers):
        raise ValueError("Asset types list must match the length of tickers.")
    if sims <= 0:
        raise ValueError("Number of simulations must be positive.")
    if years <= 0:
        raise ValueError("Years must be greater than zero.")

    tickers = list(tickers)
    shares_owned = list(shares_owned)
    asset_type_overrides: Optional[List[Optional[str]]] = None
    if asset_types is not None:
        asset_type_overrides = [atype for atype in asset_types]

    asset_classes, metadata_cache, classification_sources = _infer_asset_classes(
        tickers, asset_type_overrides
    )

    days = max(int(years * trading_days_per_year), 1)
    rng = np.random.default_rng()

    try:
        historical_start = pd.Timestamp(start_date)
    except Exception as exc:
        raise ValueError("Invalid historical start date.") from exc

    if end_date is None:
        default_end = DEFAULT_END_DATE
        if default_end is None:
            historical_end = pd.Timestamp.today().normalize()
        else:
            historical_end = pd.Timestamp(default_end)
    else:
        try:
            historical_end = pd.Timestamp(end_date)
        except Exception as exc:
            raise ValueError("Invalid historical end date.") from exc

    if historical_end <= historical_start:
        raise ValueError("Historical end date must be after start date.")

    ten_year_cap = (historical_start + pd.DateOffset(years=10)) - pd.Timedelta(days=1)
    if historical_end > ten_year_cap:
        historical_end = ten_year_cap

    calibration_span_years = (historical_end - historical_start).days / 365.25
    if calibration_span_years < 5:
        raise ValueError("Need at least five years of history for calibration.")

    fetch_start = historical_start.strftime("%Y-%m-%d")
    fetch_end = (historical_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        data = yf.download(
            tickers=tickers,
            start=fetch_start,
            end=fetch_end,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        joined_tickers = ", ".join(map(str, tickers)) or "the requested tickers"
        raise ValueError(f"Failed to download data for {joined_tickers}: {exc}") from exc
    if data.empty:
        raise ValueError("No data downloaded. Check tickers, dates, or internet connectivity.")

    close_prices = _extract_close_prices(data, tickers)
    close_prices = close_prices.loc[
        (close_prices.index >= historical_start) & (close_prices.index <= historical_end)
    ]
    if close_prices.empty:
        raise ValueError("No close price data available after cleaning.")

    canonical_aliases: List[str] = []
    ef_indices: List[int] = []
    ef_aliases: List[str] = []
    for idx, ticker in enumerate(tickers):
        alias = _canonicalize_ticker(ticker)
        canonical_aliases.append(alias)
        if alias in STANDARDIZED_EXPECTED_RETURNS and alias in STANDARDIZED_COVARIANCE.index:
            ef_indices.append(idx)
            ef_aliases.append(alias)
    ef_index_lookup = {idx: pos for pos, idx in enumerate(ef_indices)}

    last_prices = close_prices.iloc[-1].to_numpy(dtype=float)
    if not np.all(np.isfinite(last_prices)) or np.any(last_prices <= 0):
        raise ValueError("Invalid closing prices detected. Ensure all tickers have positive, finite prices.")
    for pos, idx in enumerate(ef_indices):
        reference_price = STANDARDIZED_REFERENCE_PRICES.get(ef_aliases[pos])
        if reference_price is not None:
            last_prices[idx] = reference_price
    standardized_inputs = _prepare_standardized_inputs(tickers, trading_days_per_year)
    if standardized_inputs is not None:
        for idx, ticker in enumerate(tickers):
            alias = _canonicalize_ticker(ticker)
            reference_price = STANDARDIZED_REFERENCE_PRICES.get(alias)
            if reference_price is not None:
                last_prices[idx] = reference_price
    shares_array = np.asarray(shares_owned, dtype=float)
    asset_values = shares_array * last_prices
    total_asset_value = float(np.sum(asset_values))
    initial_portfolio_value = float(total_asset_value + cash)
    if initial_portfolio_value > 0:
        asset_weight_vector = asset_values / initial_portfolio_value
    else:
        asset_weight_vector = np.zeros_like(asset_values)

    use_standardized_inputs = standardized_inputs is not None
    target_metrics: Optional[Dict[str, float]] = None
    ef_subset_inputs: Optional[Dict[str, object]] = None
    if not use_standardized_inputs and ef_indices:
        ef_subset_inputs = _prepare_standardized_inputs([
            tickers[idx] for idx in ef_indices
        ], trading_days_per_year)

    if not use_standardized_inputs:
        log_prices = np.log(close_prices)
        log_returns = log_prices.diff().dropna()
        if log_returns.empty:
            raise ValueError("Insufficient historical data to estimate returns.")

        lower_clip = log_returns.quantile(0.005)
        upper_clip = log_returns.quantile(0.995)
        log_returns = log_returns.clip(lower=lower_clip, upper=upper_clip, axis=1)
        if log_returns.empty:
            raise ValueError("Insufficient historical data to estimate returns.")

        pandemic_clip_mask = (log_returns.index >= pd.Timestamp("2020-03-01")) & (
            log_returns.index <= pd.Timestamp("2021-12-31")
        )
        if pandemic_clip_mask.any():
            pandemic_returns = log_returns.loc[pandemic_clip_mask]
            lower = pandemic_returns.quantile(0.05)
            upper = pandemic_returns.quantile(0.95)
            log_returns.loc[pandemic_clip_mask] = pandemic_returns.clip(
                lower=lower, upper=upper, axis=1
            )

        expected_inflation_annual, inflation_source = _fetch_market_implied_inflation()
        daily_inflation_rate = np.log1p(expected_inflation_annual) / trading_days_per_year
        real_log_returns = log_returns - daily_inflation_rate

        dates = real_log_returns.index
        weights = np.ones(len(dates), dtype=float)
        early_cycle_mask = dates < pd.Timestamp("2018-01-01")
        pandemic_weight_mask = (dates >= pd.Timestamp("2020-03-01")) & (dates <= pd.Timestamp("2021-12-31"))
        late_cycle_mask = dates >= pd.Timestamp("2022-01-01")
        weights[early_cycle_mask] = 0.85
        weights[pandemic_weight_mask] = 0.45
        weights[late_cycle_mask] = 1.15
        weights = np.clip(weights, 0.05, None)
        weight_total = weights.sum()
        if weight_total <= 0:
            raise ValueError("Invalid regime weights — unable to construct sampling probabilities.")
        probabilities = weights / weight_total

        real_log_returns_array = real_log_returns.to_numpy(dtype=float)
        hist_days = real_log_returns_array.shape[0]
        if hist_days < 2:
            raise ValueError("Need at least two days of historical returns.")

        mu_real = np.average(real_log_returns_array, axis=0, weights=probabilities)
        cov_real = np.cov(real_log_returns_array, rowvar=False, aweights=probabilities)
        cov_real = _ensure_positive_semidefinite(cov_real)

        if ef_subset_inputs is not None and ef_indices:
            subset_mu_nominal = np.asarray(ef_subset_inputs["daily_log_mean"], dtype=float)
            subset_cov_log = np.asarray(ef_subset_inputs["daily_log_covariance"], dtype=float)
            cov_real[np.ix_(ef_indices, ef_indices)] = subset_cov_log
            for local_pos, idx in enumerate(ef_indices):
                mu_real[idx] = subset_mu_nominal[local_pos] - daily_inflation_rate

        valuation_adjustment = _compute_valuation_adjustment()
        adjusted_mu_real = mu_real.copy()
        forward_nominal_estimates: List[Optional[float]] = [None] * len(tickers)
        forward_real_targets: List[Optional[float]] = [None] * len(tickers)
        valuation_scalars: List[float] = [np.nan] * len(tickers)

        for idx, asset_class in enumerate(asset_classes):
            if ef_subset_inputs is not None and idx in ef_index_lookup:
                local_pos = ef_index_lookup[idx]
                forward_nominal_estimates[idx] = float(ef_subset_inputs["annual_expected"][local_pos])
                forward_real_targets[idx] = float(ef_subset_inputs["annual_expected"][local_pos])
                valuation_scalars[idx] = np.nan
                adjusted_mu_real[idx] = mu_real[idx]
                continue
            metadata = metadata_cache.get(tickers[idx], {})
            if asset_class == "bond":
                valuation_scalars[idx] = np.nan
                forward_nominal = _estimate_bond_forward_nominal_return(metadata)
                forward_nominal_estimates[idx] = forward_nominal
                if forward_nominal is not None:
                    annual_real_forward = (1.0 + forward_nominal) / (1.0 + expected_inflation_annual) - 1.0
                    if np.isfinite(annual_real_forward):
                        forward_real_targets[idx] = annual_real_forward
                        if annual_real_forward <= -0.999:
                            forward_real_daily = mu_real[idx]
                        else:
                            forward_real_daily = np.log1p(annual_real_forward) / trading_days_per_year
                        adjusted_mu_real[idx] = 0.5 * mu_real[idx] + 0.5 * forward_real_daily
            elif asset_class == "equity":
                valuation_scalars[idx] = valuation_adjustment
                adjusted_mu_real[idx] = mu_real[idx] * valuation_adjustment
            else:
                valuation_scalars[idx] = np.nan

            if not np.isfinite(adjusted_mu_real[idx]):
                adjusted_mu_real[idx] = mu_real[idx]
    else:
        expected_inflation_annual = 0.0
        inflation_source = "Standardized inputs"
        daily_inflation_rate = 0.0
        mu_real = standardized_inputs["daily_log_mean"]
        cov_real = standardized_inputs["daily_log_covariance"]
        valuation_adjustment = 1.0
        adjusted_mu_real = mu_real.copy()
        forward_nominal_estimates = [float(v) for v in standardized_inputs["annual_expected"]]
        forward_real_targets = [float(v) for v in standardized_inputs["annual_expected"]]
        valuation_scalars = [np.nan] * len(tickers)

        expected_annual = standardized_inputs["annual_expected"]
        covariance_annual = standardized_inputs["annual_covariance"]
        portfolio_expected = float(asset_weight_vector @ expected_annual)
        portfolio_variance = float(asset_weight_vector @ covariance_annual @ asset_weight_vector)
        portfolio_variance = max(portfolio_variance, 0.0)
        portfolio_volatility = float(np.sqrt(portfolio_variance))
        if portfolio_volatility > 0:
            sharpe_ratio = float((portfolio_expected - STANDARDIZED_RISK_FREE) / portfolio_volatility)
        else:
            sharpe_ratio = float("nan")
        target_metrics = {
            "expected_return": portfolio_expected,
            "volatility": portfolio_volatility,
            "sharpe": sharpe_ratio,
        }

    if target_metrics is None:
        expected_annual_combined = np.empty(len(tickers), dtype=float)
        for idx in range(len(tickers)):
            if use_standardized_inputs:
                expected_annual_combined[idx] = float(standardized_inputs["annual_expected"][idx])
            elif ef_subset_inputs is not None and idx in ef_index_lookup:
                expected_annual_combined[idx] = float(
                    ef_subset_inputs["annual_expected"][ef_index_lookup[idx]]
                )
            else:
                expected_annual_combined[idx] = float(
                    np.expm1((adjusted_mu_real[idx] + daily_inflation_rate) * trading_days_per_year)
                )

        covariance_annual_combined = cov_real * trading_days_per_year
        portfolio_expected = float(asset_weight_vector @ expected_annual_combined)
        portfolio_variance = float(
            asset_weight_vector @ covariance_annual_combined @ asset_weight_vector
        )
        portfolio_variance = max(portfolio_variance, 0.0)
        portfolio_volatility = float(np.sqrt(portfolio_variance))
        if portfolio_volatility > 0:
            sharpe_ratio = float(
                (portfolio_expected - STANDARDIZED_RISK_FREE) / portfolio_volatility
            )
        else:
            sharpe_ratio = float("nan")

        target_metrics = {
            "expected_return": portfolio_expected,
            "volatility": portfolio_volatility,
            "sharpe": sharpe_ratio,
        }

    sampled_real_log_returns = rng.multivariate_normal(
        mean=adjusted_mu_real,
        cov=cov_real,
        size=days * sims,
    ).reshape(days, sims, len(tickers))
    sampled_nominal_log_returns = sampled_real_log_returns + daily_inflation_rate
    cumulative_nominal_log_returns = np.cumsum(sampled_nominal_log_returns, axis=0)
    simulated_prices = np.exp(cumulative_nominal_log_returns) * last_prices[np.newaxis, np.newaxis, :]
    simulated_simple_returns = np.expm1(sampled_nominal_log_returns)

    output_dir = f"montecarlo_portfolio_data_{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    stock_plots_dir = os.path.join(output_dir, "stock_plots")
    os.makedirs(stock_plots_dir, exist_ok=True)

    colors = cm.get_cmap("tab10").colors
    x_years = np.linspace(0, years, days)

    stock_csv_paths: List[str] = []
    stock_plot_paths: List[str] = []

    for idx, ticker in enumerate(tickers):
        stock_prices = simulated_prices[:, :, idx]

        df_stock = pd.DataFrame(stock_prices)
        stock_csv_path = os.path.join(output_dir, f"{ticker}_montecarlo_{run_name}.csv")
        df_stock.to_csv(stock_csv_path, index=False)
        stock_csv_paths.append(stock_csv_path)

        simulated_pct = stock_prices / last_prices[idx] - 1
        median_pct = np.median(simulated_pct, axis=1)

        top_path_count = min(5, simulated_pct.shape[1])
        if top_path_count > 0:
            top_indices = rng.choice(
                simulated_pct.shape[1],
                size=top_path_count,
                replace=False,
            )
        else:
            top_indices = np.empty((0,), dtype=int)

        fig_stock = Figure(figsize=(12, 6))
        ax_stock = fig_stock.add_subplot(111)
        ax_stock.plot(
            x_years,
            simulated_pct * 100,
            color=colors[idx % len(colors)],
            alpha=0.05,
            linewidth=0.8,
        )
        ax_stock.plot(
            x_years,
            median_pct * 100,
            color=colors[idx % len(colors)],
            linewidth=3,
            label=f"{ticker} Median",
        )

        for j, sim_idx in enumerate(top_indices):
            ax_stock.plot(
                x_years,
                simulated_pct[:, sim_idx] * 100,
                linestyle="--",
                linewidth=2.5,
                alpha=0.9,
                color=colors[idx % len(colors)],
                label=f"{ticker} Top 5" if j == 0 else None,
            )

        ymin_stock = np.percentile(simulated_pct * 100, 5)
        ymax_stock = np.percentile(simulated_pct * 100, 95)
        padding = (ymax_stock - ymin_stock) * 0.05
        ax_stock.set_ylim(ymin_stock - padding, ymax_stock + padding)

        ax_stock.set_title(f"Monte Carlo Simulation for {ticker} (% Growth) — {run_name}")
        ax_stock.set_xlabel("Years")
        ax_stock.set_ylabel("Percentage Growth (%)")
        ax_stock.legend(fontsize=10)
        ax_stock.grid(True, linestyle="--", alpha=0.4)
        fig_stock.tight_layout()

        stock_plot_path = os.path.join(stock_plots_dir, f"{ticker}_montecarlo_pct_{run_name}.png")
        fig_stock.savefig(stock_plot_path)
        stock_plot_paths.append(stock_plot_path)

    if use_standardized_inputs:
        portfolio_simple_returns = np.tensordot(
            simulated_simple_returns,
            asset_weight_vector,
            axes=([2], [0]),
        )
        portfolio_growth = np.cumprod(1.0 + portfolio_simple_returns, axis=0)
        portfolio_values_with_cash = initial_portfolio_value * portfolio_growth
    else:
        portfolio_values = np.tensordot(simulated_prices, shares_array, axes=([2], [0]))
        portfolio_values_with_cash = portfolio_values + cash
    portfolio_pct = portfolio_values_with_cash / initial_portfolio_value - 1

    portfolio_values_full = np.vstack(
        [np.full((1, sims), initial_portfolio_value), portfolio_values_with_cash]
    )
    portfolio_daily_returns = portfolio_values_full[1:] / portfolio_values_full[:-1] - 1
    if days > 1:
        annualized_vol = np.sqrt(trading_days_per_year) * np.std(
            portfolio_daily_returns, axis=0, ddof=1
        )
    else:
        annualized_vol = np.zeros(sims, dtype=float)

    median_portfolio = np.median(portfolio_pct, axis=1)
    median_end_pct = float(median_portfolio[-1] * 100)

    top_path_count = min(10, portfolio_pct.shape[1])
    if top_path_count > 0:
        top_indices = rng.choice(portfolio_pct.shape[1], size=top_path_count, replace=False)
    else:
        top_indices = np.empty((0,), dtype=int)

    portfolio_fig = Figure(figsize=(14, 7))
    portfolio_ax = portfolio_fig.add_subplot(111)
    portfolio_ax.plot(x_years, portfolio_pct * 100, color="red", alpha=0.05, linewidth=0.8)
    portfolio_ax.plot(x_years, median_portfolio * 100, color="black", linewidth=4, label="Portfolio Median")

    for i, sim_idx in enumerate(top_indices):
        portfolio_ax.plot(
            x_years,
            portfolio_pct[:, sim_idx] * 100,
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            color="red",
            label="Top 10 Paths" if i == 0 else None,
        )

    goal_factor = 0.30    # goal +30 percent
    bust_factor = -0.1    # bust -10 percent
    portfolio_ax.axhline(
        goal_factor * 100,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Goal ({goal_factor * 100:.0f}%)",
    )
    portfolio_ax.axhline(
        bust_factor * 100,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Bust ({bust_factor * 100:.0f}%)",
    )

    lower_percentile = float(np.percentile(portfolio_pct * 100, 5))
    upper_percentile = float(np.percentile(portfolio_pct * 100, 90))
    ymin_portfolio = min(lower_percentile, bust_factor * 100 * 1.1)
    ymax_portfolio = max(median_end_pct, goal_factor * 100, upper_percentile)
    padding_portfolio = max((ymax_portfolio - ymin_portfolio) * 0.05, 5.0)
    portfolio_ax.set_ylim(ymin_portfolio - padding_portfolio, ymax_portfolio + padding_portfolio)

    portfolio_ax.set_title(f"Portfolio Monte Carlo (% Growth) — {run_name}")
    portfolio_ax.set_xlabel("Years")
    portfolio_ax.set_ylabel("Portfolio Growth (%)")
    portfolio_ax.grid(True, linestyle="--", alpha=0.4)
    portfolio_ax.legend(fontsize=10)
    portfolio_fig.tight_layout()

    portfolio_plot_path = os.path.join(output_dir, f"portfolio_montecarlo_pct_top10_{run_name}.png")
    portfolio_fig.savefig(portfolio_plot_path)

    df_portfolio = pd.DataFrame(portfolio_values_with_cash)
    portfolio_csv_path = os.path.join(output_dir, f"portfolio_montecarlo_{run_name}.csv")
    df_portfolio.to_csv(portfolio_csv_path, index=False)

    final_values = portfolio_values_with_cash[-1]
    goal_threshold = initial_portfolio_value * (1 + goal_factor)
    bust_threshold = initial_portfolio_value * (1 + bust_factor)
    prob_goal = float(np.mean(final_values >= goal_threshold))
    prob_bust = float(np.mean(final_values <= bust_threshold))
    target_return = 0.125
    target_threshold = initial_portfolio_value * (1 + target_return) ** years
    prob_target = float(np.mean(final_values >= target_threshold))

    with np.errstate(invalid="ignore", divide="ignore"):
        cagr = np.where(
            final_values <= 0,
            -1.0,
            np.power(final_values / initial_portfolio_value, 1 / years) - 1,
        )
    valid_mask = cagr > -1.0
    valid_cagr = cagr[valid_mask]
    if valid_cagr.size:
        median_cagr_value = float(np.median(valid_cagr))
        mean_cagr_value = float(np.mean(valid_cagr))
    else:
        median_cagr_value = float("nan")
        mean_cagr_value = float("nan")
    median_vol_value = float(np.median(annualized_vol))

    top_summary_rows: List[Dict[str, object]] = []
    top_count = min(10, sims)
    if top_count > 0:
        top_indices = np.argsort(final_values)[-top_count:][::-1]
        for rank, sim_idx in enumerate(top_indices, start=1):
            end_value = float(final_values[sim_idx])
            cagr_pct = (
                (cagr[sim_idx] * 100.0)
                if cagr[sim_idx] > -1.0
                else float("nan")
            )
            top_summary_rows.append(
                {
                    "Entry": f"Top {rank}",
                    "Simulation Index": int(sim_idx),
                    "End Value": end_value,
                    "Total Return %": (end_value / initial_portfolio_value - 1) * 100.0,
                    "CAGR %": cagr_pct,
                    "Median CAGR % (All Sims)": "",
                    "Prob (CAGR ≥ 12.5%)": "",
                }
            )
        top_final_values = final_values[top_indices]
        if top_final_values.size:
            median_top_value = float(np.median(top_final_values))
            valid_top_cagr = [c for c in cagr[top_indices] if c > -1.0]
            median_top_cagr = float(np.median(valid_top_cagr) * 100.0) if valid_top_cagr else float("nan")
            top_summary_rows.append(
                {
                    "Entry": "Top 10 Median",
                    "Simulation Index": "",
                    "End Value": median_top_value,
                    "Total Return %": (median_top_value / initial_portfolio_value - 1) * 100.0,
                    "CAGR %": median_top_cagr,
                    "Median CAGR % (All Sims)": "",
                    "Prob (CAGR ≥ 12.5%)": "",
                }
            )
    top_summary_rows.append(
        {
            "Entry": "Summary",
            "Simulation Index": "",
            "End Value": "",
            "Total Return %": "",
            "CAGR %": "",
            "Median CAGR % (All Sims)": float(median_cagr_value * 100.0) if np.isfinite(median_cagr_value) else "",
            "Prob (CAGR ≥ 12.5%)": float(prob_target * 100.0),
        }
    )

    if use_standardized_inputs:
        portfolio_real_drift_daily = float(np.dot(adjusted_mu_real, asset_weight_vector))
    else:
        invested_value = float(np.sum(shares_array * last_prices))
        if invested_value > 0:
            portfolio_real_drift_daily = float(np.dot(adjusted_mu_real, shares_array * last_prices) / invested_value)
        else:
            portfolio_real_drift_daily = 0.0
    portfolio_real_drift_annual = portfolio_real_drift_daily * trading_days_per_year

    summary = pd.DataFrame(
        {
            "Median End Value": [float(np.median(final_values))],
            "Mean End Value": [float(np.mean(final_values))],
            "5th Percentile": [float(np.percentile(final_values, 5))],
            "95th Percentile": [float(np.percentile(final_values, 95))],
            "Initial Portfolio Value": [initial_portfolio_value],
            "Median CAGR": [median_cagr_value],
            "Mean CAGR": [mean_cagr_value],
            "Median Annual Volatility": [median_vol_value],
            "Probability Goal": [prob_goal],
            "Probability Bust": [prob_bust],
            "Probability CAGR ≥ 12.5%": [prob_target],
            "Calibrated Real Drift (annual)": [portfolio_real_drift_annual],
            "Equity Valuation Scalar": [valuation_adjustment],
            "Assumed Inflation (annual)": [expected_inflation_annual],
            "Inflation Source": [inflation_source],
            "Target Expected Return": [
                target_metrics["expected_return"] if target_metrics else float("nan")
            ],
            "Target Annual Volatility": [
                target_metrics["volatility"] if target_metrics else float("nan")
            ],
            "Target Sharpe": [
                target_metrics["sharpe"] if target_metrics else float("nan")
            ],
        }
    )

    assumptions_rows: List[Dict[str, object]] = []
    for idx, ticker in enumerate(tickers):
        metadata = metadata_cache.get(ticker, {})
        forward_nominal = forward_nominal_estimates[idx]
        forward_real = forward_real_targets[idx]
        valuation_scalar_value = valuation_scalars[idx]

        if forward_nominal is None or not np.isfinite(forward_nominal):
            forward_nominal_value: float = np.nan
        else:
            forward_nominal_value = float(forward_nominal)

        if forward_real is None or not np.isfinite(forward_real):
            forward_real_value: float = np.nan
        else:
            forward_real_value = float(forward_real)

        if not np.isfinite(valuation_scalar_value):
            valuation_scalar_export: float = np.nan
        else:
            valuation_scalar_export = float(valuation_scalar_value)

        assumptions_rows.append(
            {
                "Ticker": ticker,
                "Asset Type": asset_classes[idx],
                "Classification Source": classification_sources[idx],
                "User Override": (asset_type_overrides[idx] if asset_type_overrides is not None else None),
                "Current Market Value": float(asset_values[idx]),
                "Portfolio Weight": float(asset_weight_vector[idx]),
                "Historical Real Drift (daily)": float(mu_real[idx]),
                "Adjusted Real Drift (daily)": float(adjusted_mu_real[idx]),
                "Historical Real Return (annual)": float(np.expm1(mu_real[idx] * trading_days_per_year)),
                "Adjusted Real Return (annual)": float(np.expm1(adjusted_mu_real[idx] * trading_days_per_year)),
                "Forward Nominal Input (annual)": forward_nominal_value,
                "Forward Real Target (annual)": forward_real_value,
                "Valuation Scalar Applied": valuation_scalar_export,
                "Metadata Available": bool(metadata),
                "Metadata Category": metadata.get("category"),
            }
        )

    assumptions_df = pd.DataFrame(assumptions_rows)
    assumptions_path = os.path.join(output_dir, f"asset_assumptions_{run_name}.csv")
    assumptions_df.to_csv(assumptions_path, index=False)

    summary_path = os.path.join(output_dir, f"portfolio_summary_{run_name}.csv")
    summary.to_csv(summary_path, index=False)

    top_summary_df = pd.DataFrame(top_summary_rows)
    top_summary_path = os.path.join(output_dir, f"top_paths_summary_{run_name}.csv")
    top_summary_df.to_csv(top_summary_path, index=False)

    return SimulationResult(
        output_dir=output_dir,
        portfolio_plot_path=portfolio_plot_path,
        portfolio_csv_path=portfolio_csv_path,
        summary_path=summary_path,
        asset_assumptions_path=assumptions_path,
        stock_csv_paths=stock_csv_paths,
        stock_plot_paths=stock_plot_paths,
        figure=portfolio_fig,
        summary=summary,
        goal_probability=prob_goal,
        bust_probability=prob_bust,
        median_cagr=median_cagr_value,
        median_annual_volatility=median_vol_value,
        initial_portfolio_value=initial_portfolio_value,
        goal_threshold=goal_threshold,
        bust_threshold=bust_threshold,
    )


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError("Tkinter is required for the GUI but is not available in this environment.") from exc

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Unable to launch Tkinter GUI (no available display?).") from exc
    root.title("Monte Carlo Portfolio Simulator")
    root.geometry("1100x800")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    status_var = tk.StringVar(value="Ready")
    last_result: Optional[SimulationResult] = None
    current_canvas: Optional[FigureCanvasTkAgg] = None
    current_figure: Optional[Figure] = None
    last_run_assets: List[str] = []
    ASSET_TYPES = ("Stock", "Bond", "ETF", "Other")
    assets: List[dict] = [
        {"ticker": ticker.upper(), "shares": shares, "asset_type": "Stock"}
        for ticker, shares in zip(DEFAULT_TICKERS, DEFAULT_SHARES)
    ]

    main_frame = ttk.Frame(root, padding=12)
    main_frame.grid(row=0, column=0, sticky="nsew")
    for col in range(4):
        main_frame.columnconfigure(col, weight=1)
    main_frame.rowconfigure(0, weight=2)
    main_frame.rowconfigure(6, weight=3)
    main_frame.rowconfigure(7, weight=1)

    cash_var = tk.StringVar(value=str(DEFAULT_CASH))
    sims_var = tk.StringVar(value=str(DEFAULT_SIMS))
    years_var = tk.StringVar(value=str(DEFAULT_YEARS))
    run_name_var = tk.StringVar(value=DEFAULT_RUN_NAME)
    start_var = tk.StringVar(value=DEFAULT_START_DATE)
    end_var = tk.StringVar(value=DEFAULT_END_DATE or "")

    assets_frame = ttk.LabelFrame(main_frame, text="Portfolio Assets")
    assets_frame.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=(0, 12))
    assets_frame.columnconfigure(0, weight=1)
    assets_frame.rowconfigure(0, weight=1)

    asset_tree = ttk.Treeview(
        assets_frame,
        columns=("Ticker", "Type", "Shares"),
        show="headings",
        height=8,
    )
    asset_tree.heading("Ticker", text="Ticker")
    asset_tree.heading("Type", text="Type")
    asset_tree.heading("Shares", text="Shares")
    asset_tree.column("Ticker", width=120, anchor="w")
    asset_tree.column("Type", width=120, anchor="w")
    asset_tree.column("Shares", width=120, anchor="e")
    asset_tree.grid(row=0, column=0, sticky="nsew")

    asset_scroll = ttk.Scrollbar(assets_frame, orient="vertical", command=asset_tree.yview)
    asset_scroll.grid(row=0, column=1, sticky="ns")
    asset_tree.configure(yscrollcommand=asset_scroll.set)

    asset_button_frame = ttk.Frame(assets_frame)
    asset_button_frame.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="ew")
    asset_button_frame.columnconfigure(0, weight=1)
    asset_button_frame.columnconfigure(1, weight=1)
    asset_button_frame.columnconfigure(2, weight=1)
    asset_button_frame.columnconfigure(3, weight=1)

    add_asset_btn = ttk.Button(asset_button_frame, text="Add Asset")
    edit_asset_btn = ttk.Button(asset_button_frame, text="Edit Selected")
    remove_asset_btn = ttk.Button(asset_button_frame, text="Remove Selected")
    clear_assets_btn = ttk.Button(asset_button_frame, text="Clear All")

    add_asset_btn.grid(row=0, column=0, padx=4, sticky="ew")
    edit_asset_btn.grid(row=0, column=1, padx=4, sticky="ew")
    remove_asset_btn.grid(row=0, column=2, padx=4, sticky="ew")
    clear_assets_btn.grid(row=0, column=3, padx=4, sticky="ew")

    assets_hint = ttk.Label(
        assets_frame,
        text="Double-click an asset to edit it. Use Add Asset to include stocks, bond funds, or ETFs.",
        anchor="w",
    )
    assets_hint.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def update_asset_buttons() -> None:
        has_selection = bool(asset_tree.selection())
        has_assets = bool(assets)
        edit_asset_btn.config(state="normal" if has_selection else "disabled")
        remove_asset_btn.config(state="normal" if has_selection else "disabled")
        clear_assets_btn.config(state="normal" if has_assets else "disabled")

    def refresh_asset_tree() -> None:
        asset_tree.delete(*asset_tree.get_children())
        for idx, asset in enumerate(assets):
            asset_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(asset["ticker"], asset["asset_type"], asset["shares"]),
            )
        update_asset_buttons()

    def open_asset_dialog(existing: Optional[dict] = None, index: Optional[int] = None) -> None:
        dialog = tk.Toplevel(root)
        dialog.title("Edit Asset" if existing else "Add Asset")
        dialog.transient(root)
        dialog.grab_set()
        dialog.resizable(False, False)

        ticker_var = tk.StringVar(value=existing["ticker"] if existing else "")
        shares_var = tk.StringVar(value=str(existing["shares"]) if existing else "")
        type_var = tk.StringVar(value=existing["asset_type"] if existing else ASSET_TYPES[0])

        ttk.Label(dialog, text="Ticker").grid(row=0, column=0, sticky="w", padx=8, pady=(12, 4))
        ticker_entry = ttk.Entry(dialog, textvariable=ticker_var, width=20)
        ticker_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=(12, 4))

        ttk.Label(dialog, text="Shares").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        shares_entry = ttk.Entry(dialog, textvariable=shares_var, width=20)
        shares_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(dialog, text="Asset Type").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        type_combo = ttk.Combobox(dialog, values=ASSET_TYPES, textvariable=type_var, state="readonly", width=18)
        type_combo.grid(row=2, column=1, sticky="ew", padx=8, pady=4)
        if type_var.get() not in ASSET_TYPES:
            type_combo.current(0)

        info_text = (
            "Enter the Yahoo Finance ticker symbol and the number of shares to simulate. "
            "For bonds, use an ETF or mutual fund ticker that provides daily pricing."
        )
        info_label = ttk.Label(dialog, text=info_text, wraplength=360, justify="left")
        info_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 12))

        button_box = ttk.Frame(dialog)
        button_box.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 12))
        button_box.columnconfigure(0, weight=1)
        button_box.columnconfigure(1, weight=1)

        def submit() -> None:
            ticker_value = ticker_var.get().strip().upper()
            if not ticker_value:
                messagebox.showerror("Invalid asset", "Enter a ticker symbol.")
                return

            try:
                shares_value = int(shares_var.get())
            except ValueError:
                messagebox.showerror("Invalid asset", "Shares must be a whole number.")
                return

            if shares_value <= 0:
                messagebox.showerror("Invalid asset", "Shares must be greater than zero.")
                return

            asset_type_value = type_var.get() or ASSET_TYPES[0]
            asset_data = {
                "ticker": ticker_value,
                "shares": shares_value,
                "asset_type": asset_type_value,
            }

            if index is None:
                assets.append(asset_data)
            else:
                assets[index] = asset_data

            refresh_asset_tree()
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        ttk.Button(button_box, text="Cancel", command=cancel).grid(row=0, column=0, padx=4, sticky="ew")
        ttk.Button(button_box, text="Save Asset", command=submit).grid(row=0, column=1, padx=4, sticky="ew")

        dialog.bind("<Return>", lambda _: submit())
        dialog.bind("<Escape>", lambda _: cancel())
        ticker_entry.focus_set()
        dialog.wait_window()

    def add_asset() -> None:
        open_asset_dialog()

    def edit_selected_asset(event: Optional[tk.Event] = None) -> None:
        selection = asset_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        open_asset_dialog(assets[index], index)

    def remove_selected_asset() -> None:
        selection = asset_tree.selection()
        if not selection:
            messagebox.showinfo("Remove asset", "Select an asset to remove.")
            return
        for item in sorted((int(i) for i in selection), reverse=True):
            assets.pop(item)
        refresh_asset_tree()

    def clear_assets() -> None:
        if not assets:
            messagebox.showinfo("Clear assets", "There are no assets to clear.")
            return
        if messagebox.askyesno("Clear assets", "Remove all assets from the list?"):
            assets.clear()
            refresh_asset_tree()

    add_asset_btn.config(command=add_asset)
    edit_asset_btn.config(command=edit_selected_asset)
    remove_asset_btn.config(command=remove_selected_asset)
    clear_assets_btn.config(command=clear_assets)

    asset_tree.bind("<<TreeviewSelect>>", lambda _: update_asset_buttons())
    asset_tree.bind("<Double-1>", edit_selected_asset)
    asset_tree.bind("<Return>", edit_selected_asset)
    asset_tree.bind("<Delete>", lambda _: remove_selected_asset())
    asset_tree.bind("<BackSpace>", lambda _: remove_selected_asset())
    asset_tree.focus_set()

    def add_labeled_entry(row: int, column: int, label: str, text_var: tk.StringVar) -> ttk.Entry:
        ttk.Label(main_frame, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(main_frame, textvariable=text_var)
        entry.grid(row=row, column=column + 1, sticky="ew", pady=4)
        return entry

    add_labeled_entry(1, 0, "Cash", cash_var)
    add_labeled_entry(1, 2, "Simulations", sims_var)
    add_labeled_entry(2, 0, "Years", years_var)
    add_labeled_entry(2, 2, "Run name", run_name_var)
    add_labeled_entry(3, 0, "Start date (YYYY-MM-DD)", start_var)
    add_labeled_entry(3, 2, "End date (YYYY-MM-DD)", end_var)

    asset_info_frame = ttk.LabelFrame(main_frame, text="Asset Guidance")
    asset_info_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(12, 8))
    asset_info_text = (
        "Stocks: use the equity ticker (e.g., VOO, AAPL). Bonds: supply the ticker of a bond ETF or mutual fund "
        "that trades daily. The simulator retrieves historical close prices from Yahoo Finance for every ticker "
        "listed above; ensure the instruments you add have sufficient history covering the selected date range."
    )
    ttk.Label(asset_info_frame, text=asset_info_text, wraplength=1000, justify="left").grid(
        row=0, column=0, sticky="w", padx=8, pady=8
    )

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=5, column=0, columnspan=4, pady=(12, 8), sticky="ew")
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)
    button_frame.columnconfigure(2, weight=1)

    chart_frame = ttk.LabelFrame(main_frame, text="Portfolio Chart")
    chart_frame.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(8, 8))
    chart_frame.rowconfigure(0, weight=1)
    chart_frame.columnconfigure(0, weight=1)

    files_frame = ttk.LabelFrame(main_frame, text="Generated Files")
    files_frame.grid(row=7, column=0, columnspan=4, sticky="nsew")
    files_frame.columnconfigure(0, weight=1)
    files_frame.rowconfigure(0, weight=1)

    files_listbox = tk.Listbox(files_frame, height=8)
    files_listbox.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=files_listbox.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    files_listbox.configure(yscrollcommand=scrollbar.set)

    status_label = ttk.Label(main_frame, textvariable=status_var)
    status_label.grid(row=8, column=0, columnspan=4, sticky="w", pady=(8, 0))

    refresh_asset_tree()

    def reset_chart() -> None:
        nonlocal current_canvas, current_figure
        if current_canvas is not None:
            current_canvas.get_tk_widget().destroy()
            current_canvas = None
        if current_figure is not None:
            current_figure.clf()
            current_figure = None

    def collect_inputs() -> dict:
        if not assets:
            raise ValueError("Add at least one asset before running the simulation.")

        raw_tickers = [asset["ticker"] for asset in assets]
        shares = [asset["shares"] for asset in assets]
        asset_type_values = [asset.get("asset_type") for asset in assets]

        try:
            cash_value = float(cash_var.get() or 0)
            sims_value = int(sims_var.get() or DEFAULT_SIMS)
            years_value = float(years_var.get() or DEFAULT_YEARS)
        except ValueError as exc:
            raise ValueError("Cash, simulations, and years must be numeric.") from exc

        if sims_value <= 0:
            raise ValueError("Simulations must be positive.")
        if years_value <= 0:
            raise ValueError("Years must be positive.")

        run_name_value = run_name_var.get().strip() or DEFAULT_RUN_NAME
        start_value = start_var.get().strip() or DEFAULT_START_DATE
        end_value = end_var.get().strip() or None

        return {
            "tickers": raw_tickers,
            "shares_owned": shares,
            "cash": cash_value,
            "sims": sims_value,
            "years": years_value,
            "run_name": run_name_value,
            "start_date": start_value,
            "end_date": end_value,
            "asset_types": asset_type_values,
        }

    run_button = ttk.Button(button_frame, text="Run Simulation")
    open_button = ttk.Button(button_frame, text="Open Output Folder")
    quit_button = ttk.Button(button_frame, text="Quit", command=root.destroy)

    run_button.grid(row=0, column=0, padx=4, sticky="ew")
    open_button.grid(row=0, column=1, padx=4, sticky="ew")
    quit_button.grid(row=0, column=2, padx=4, sticky="ew")
    open_button.config(state="disabled")

    def handle_success(result: SimulationResult) -> None:
        nonlocal last_result, current_canvas, current_figure
        run_button.config(state="normal")
        open_button.config(state="normal")
        last_result = result
        status_var.set(f"Completed. Results saved in {result.output_dir}")

        files_listbox.delete(0, tk.END)
        files_listbox.insert(tk.END, os.path.basename(result.portfolio_csv_path))
        files_listbox.insert(tk.END, os.path.basename(result.summary_path))
        files_listbox.insert(tk.END, os.path.basename(result.asset_assumptions_path))
        files_listbox.insert(tk.END, os.path.basename(result.portfolio_plot_path))
        for path in result.stock_csv_paths:
            files_listbox.insert(tk.END, os.path.basename(path))
        for path in result.stock_plot_paths:
            files_listbox.insert(tk.END, os.path.join("stock_plots", os.path.basename(path)))
        if last_run_assets:
            files_listbox.insert(tk.END, "Assets: " + ", ".join(last_run_assets))
        files_listbox.insert(tk.END, f"Goal probability: {result.goal_probability:.1%}")
        files_listbox.insert(tk.END, f"Bust probability: {result.bust_probability:.1%}")
        files_listbox.insert(tk.END, f"Median CAGR: {result.median_cagr:.2%}")

        reset_chart()
        current_figure = result.figure
        current_canvas = FigureCanvasTkAgg(current_figure, master=chart_frame)
        current_canvas.draw()
        current_canvas.get_tk_widget().pack(fill="both", expand=True)

    def handle_error(error: Exception) -> None:
        run_button.config(state="normal")
        open_button.config(state="disabled")
        status_var.set(f"Error: {error}")
        messagebox.showerror("Simulation failed", str(error))

    def worker_thread(config: dict) -> None:
        try:
            result = run_simulation(**config)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            root.after(0, lambda err=exc: handle_error(err))
        else:
            root.after(0, lambda: handle_success(result))

    def run_simulation_async() -> None:
        try:
            config = collect_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        run_button.config(state="disabled")
        open_button.config(state="disabled")
        status_var.set("Running simulation...")
        files_listbox.delete(0, tk.END)
        reset_chart()

        nonlocal last_run_assets
        last_run_assets = list(config["tickers"])

        Thread(target=worker_thread, args=(config,), daemon=True).start()

    def open_output_dir() -> None:
        if last_result is None:
            messagebox.showinfo("Open folder", "Run a simulation first.")
            return

        path = last_result.output_dir
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", path])
            elif sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                raise OSError(f"Unsupported platform: {sys.platform}")
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    run_button.config(command=run_simulation_async)
    open_button.config(command=open_output_dir)

    root.bind("<Return>", lambda _: run_simulation_async())
    root.bind("<KP_Enter>", lambda _: run_simulation_async())

    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo portfolio simulator.")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical interface.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run name to append to outputs.")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS, help="Tickers to include.")
    parser.add_argument("--shares", nargs="*", type=int, default=DEFAULT_SHARES, help="Shares held for each ticker.")
    parser.add_argument(
        "--asset-types",
        nargs="*",
        help="Optional asset type hints aligned with tickers (e.g., stock, bond, etf).",
    )
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH, help="Cash portion of the portfolio.")
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS, help="Number of Monte Carlo simulations.")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS, help="Years to simulate.")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Historical start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
        help="Historical end date (YYYY-MM-DD). Use 'none' to fetch through the latest date.",
    )
    args = parser.parse_args()

    if args.gui:
        try:
            launch_gui()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if len(args.tickers) != len(args.shares):
        raise SystemExit("Error: number of tickers must match number of shares provided.")
    asset_types = args.asset_types
    if asset_types and len(asset_types) != len(args.tickers):
        raise SystemExit("Error: number of asset types must match number of tickers.")

    end_date = args.end
    if end_date and end_date.lower() == "none":
        end_date = None

    result = run_simulation(
        tickers=args.tickers,
        shares_owned=args.shares,
        asset_types=asset_types,
        cash=args.cash,
        sims=args.sims,
        years=args.years,
        run_name=args.run_name,
        start_date=args.start,
        end_date=end_date,
    )

    result.figure.clf()

    print("✅ Monte Carlo simulation complete!")
    print(f"📂 Results saved in: {result.output_dir}/")
    print(f"✅ Individual stock plots saved in '{os.path.join(result.output_dir, 'stock_plots')}/'")
    print(f"✅ Portfolio plot saved as '{os.path.basename(result.portfolio_plot_path)}'")
    print(f"✅ CSV files and summary saved with suffix '_{args.run_name}'")
    print(f"✅ Asset assumptions saved as '{os.path.basename(result.asset_assumptions_path)}'")
    goal_pct = result.goal_threshold / result.initial_portfolio_value - 1
    bust_pct = result.bust_threshold / result.initial_portfolio_value - 1
    print(f"ℹ️ Median CAGR: {result.median_cagr:.2%}")
    print(f"ℹ️ Median annual volatility: {result.median_annual_volatility:.2%}")
    print(f"ℹ️ Goal probability (≥ {goal_pct:.0%}): {result.goal_probability:.1%}")
    print(f"ℹ️ Bust probability (≤ {bust_pct:.0%}): {result.bust_probability:.1%}")


if __name__ == "__main__":
    main()
