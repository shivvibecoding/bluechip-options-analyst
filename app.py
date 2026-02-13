import math
import json
import time
import random
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib import error, request

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from fpdf import FPDF

BLUE_CHIP_TECH = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "ORCL",
    "ADBE",
    "CRM",
]

STRATEGY_LABELS = {
    "naked_call": "Buy Naked Call",
    "covered_call": "Sell Covered Call",
    "cash_secured_put": "Sell Cash-Secured Put",
}

CHAT_PERSONAS: Dict[str, str] = {
    "Calm Mentor": "Speak calmly, clearly, and with patience. Reduce anxiety while reinforcing discipline.",
    "Desk Strategist": "Speak like a pragmatic trading-desk coach: concise, direct, and execution-focused.",
    "Tough Coach": "Be blunt but constructive. Challenge weak logic and force clear risk rules.",
}


@dataclass
class CandidateTrade:
    ticker: str
    strategy: str
    risk_level: str
    score: float
    confidence: str
    expiry: str
    strike: float
    premium: float
    spot: float
    rationale: str
    trade_plan: str
    max_profit: str
    max_risk: str
    breakeven: str
    contracts: int = 1
    est_max_loss_dollars: float = 0.0
    capital_required: float = 0.0
    earnings_date: str = "N/A"


def next_business_day(today: Optional[date] = None) -> date:
    today = today or date.today()
    ts = pd.Timestamp(today)
    return (ts + pd.offsets.BDay(1)).date()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_signal_strength(hist: pd.DataFrame) -> Dict[str, float]:
    close = hist["Close"].dropna()
    if len(close) < 60:
        return {"trend": 0.0, "momentum": 0.0, "vol": 0.0, "total": 0.0}

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    spot = close.iloc[-1]
    rsi = compute_rsi(close).iloc[-1]

    returns = close.pct_change().dropna()
    hist_vol = returns.tail(20).std() * np.sqrt(252)

    trend = 1.0 if spot > sma20 and sma20 > sma50 else -1.0
    momentum = 1.0 if 48 <= rsi <= 68 else -0.5 if rsi > 75 else 0.2
    vol = 1.0 if hist_vol <= 0.45 else 0.3

    total = 0.5 * trend + 0.3 * momentum + 0.2 * vol
    return {
        "trend": trend,
        "momentum": momentum,
        "vol": vol,
        "total": total,
        "rsi": float(rsi),
        "hist_vol": float(hist_vol),
        "spot": float(spot),
        "sma20": float(sma20),
        "sma50": float(sma50),
    }


def pick_expiry(options: List[str], risk_level: str, as_of: date) -> Optional[str]:
    if not options:
        return None

    min_dte = 7 if risk_level == "conservative" else 10
    max_dte = 35 if risk_level == "moderate" else 30

    parsed = []
    for expiry in options:
        try:
            exp = datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = (exp - as_of).days
            parsed.append((expiry, dte))
        except ValueError:
            continue

    valid = [item for item in parsed if min_dte <= item[1] <= max_dte]
    if valid:
        return sorted(valid, key=lambda x: x[1])[0][0]

    future = [item for item in parsed if item[1] > 0]
    return sorted(future, key=lambda x: x[1])[0][0] if future else None


def option_mid(bid: float, ask: float, last: float) -> float:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    if last > 0:
        return last
    return 0.0


def confidence_label(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"


def extract_next_earnings_date(tk: yf.Ticker) -> Optional[date]:
    # yfinance calendar shapes vary; this parser handles common variants.
    try:
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for _, row in cal.iterrows():
                for val in row.values:
                    dt = pd.to_datetime(val, errors="coerce")
                    if pd.notna(dt):
                        return dt.date()
        elif isinstance(cal, dict):
            for val in cal.values():
                dt = pd.to_datetime(val, errors="coerce")
                if pd.notna(dt):
                    return dt.date()
    except Exception:
        pass

    try:
        ed = tk.get_earnings_dates(limit=4)
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            idx = ed.index[0]
            dt = pd.to_datetime(idx, errors="coerce")
            if pd.notna(dt):
                return dt.date()
    except Exception:
        pass

    return None


def earnings_too_close(
    earnings_dt: Optional[date],
    as_of: date,
    expiry: str,
    buffer_days: int,
    avoid_if_before_expiry: bool,
) -> bool:
    if not earnings_dt:
        return False

    exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
    delta_days = (earnings_dt - as_of).days

    if abs(delta_days) <= buffer_days:
        return True

    if avoid_if_before_expiry and as_of <= earnings_dt <= exp_dt:
        return True

    return False


def generate_naked_call(
    ticker: str,
    calls: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    max_premium_budget: float,
) -> Optional[CandidateTrade]:
    if risk_level == "conservative":
        return None
    if sig["total"] < 0.35:
        return None

    spot = sig["spot"]
    min_strike = spot * 1.00
    max_strike = spot * 1.05

    pool = calls[(calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)].copy()
    if pool.empty:
        return None

    pool["mid"] = pool.apply(
        lambda r: option_mid(float(r.get("bid", 0)), float(r.get("ask", 0)), float(r.get("lastPrice", 0))),
        axis=1,
    )
    pool = pool[(pool["mid"] > 0.05) & (pool["mid"] * 100 <= max_premium_budget)]
    if pool.empty:
        return None

    pool["moneyness"] = (pool["strike"] - spot).abs()
    best = pool.sort_values(["moneyness", "mid"], ascending=[True, True]).iloc[0]

    premium = float(best["mid"])
    strike = float(best["strike"])
    score = min(0.9, 0.55 + sig["total"] * 0.4)

    return CandidateTrade(
        ticker=ticker,
        strategy="naked_call",
        risk_level=risk_level,
        score=score,
        confidence=confidence_label(score),
        expiry=expiry,
        strike=strike,
        premium=premium,
        spot=spot,
        rationale=f"Bullish trend setup (RSI {sig['rsi']:.1f}, spot above 20/50 SMA) with controlled call premium.",
        trade_plan=f"Buy calls near-the-money for {expiry}. Take profits around 40-60% option gain or cut at 50% premium loss.",
        max_profit="Unlimited upside",
        max_risk=f"Premium paid per contract: ${premium * 100:.2f}",
        breakeven=f"${strike + premium:.2f}",
    )


def generate_covered_call(
    ticker: str,
    calls: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    shares_owned: int,
) -> Optional[CandidateTrade]:
    if shares_owned < 100:
        return None

    spot = sig["spot"]
    otm = 0.05 if risk_level == "conservative" else 0.03

    pool = calls[calls["strike"] >= spot * (1 + otm)].copy()
    if pool.empty:
        return None

    pool["mid"] = pool.apply(
        lambda r: option_mid(float(r.get("bid", 0)), float(r.get("ask", 0)), float(r.get("lastPrice", 0))),
        axis=1,
    )
    pool = pool[pool["mid"] > 0.05]
    if pool.empty:
        return None

    pool["distance"] = (pool["strike"] / spot) - 1
    best = pool.sort_values(["distance", "mid"], ascending=[True, False]).iloc[0]

    premium = float(best["mid"])
    strike = float(best["strike"])
    call_yield = premium / spot

    base = 0.78 if risk_level == "conservative" else 0.68
    score = min(0.95, base + min(0.12, call_yield * 5))

    return CandidateTrade(
        ticker=ticker,
        strategy="covered_call",
        risk_level=risk_level,
        score=score,
        confidence=confidence_label(score),
        expiry=expiry,
        strike=strike,
        premium=premium,
        spot=spot,
        rationale=f"Income overwrite. Strike sits {((strike / spot) - 1) * 100:.1f}% above spot with premium cushion.",
        trade_plan=f"Sell covered calls against owned shares for {expiry}. Monitor if stock approaches strike.",
        max_profit=f"~${((strike - spot + premium) * 100):.2f} per contract plus dividends if held",
        max_risk="Stock downside risk remains (partially cushioned by premium)",
        breakeven=f"${spot - premium:.2f}",
    )


def generate_cash_secured_put(
    ticker: str,
    puts: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    cash_per_trade: float,
) -> Optional[CandidateTrade]:
    spot = sig["spot"]
    discount = 0.06 if risk_level == "conservative" else 0.04

    pool = puts[puts["strike"] <= spot * (1 - discount)].copy()
    if pool.empty:
        return None

    pool["mid"] = pool.apply(
        lambda r: option_mid(float(r.get("bid", 0)), float(r.get("ask", 0)), float(r.get("lastPrice", 0))),
        axis=1,
    )
    pool = pool[(pool["mid"] > 0.05) & (pool["strike"] * 100 <= cash_per_trade)]
    if pool.empty:
        return None

    pool["distance"] = 1 - (pool["strike"] / spot)
    best = pool.sort_values(["distance", "mid"], ascending=[True, False]).iloc[0]

    premium = float(best["mid"])
    strike = float(best["strike"])
    collateral = strike * 100
    roc = (premium * 100) / collateral

    base = 0.82 if risk_level == "conservative" else 0.72
    score = min(0.96, base + min(0.1, roc * 8) + (0.04 if sig["total"] >= 0 else -0.05))

    return CandidateTrade(
        ticker=ticker,
        strategy="cash_secured_put",
        risk_level=risk_level,
        score=score,
        confidence=confidence_label(score),
        expiry=expiry,
        strike=strike,
        premium=premium,
        spot=spot,
        rationale=f"Acquire-at-discount setup with strike {((1 - strike / spot) * 100):.1f}% below spot and paid premium.",
        trade_plan=f"Sell cash-secured puts for {expiry}; maintain full collateral.",
        max_profit=f"Premium received per contract: ${premium * 100:.2f}",
        max_risk=f"Assignment risk down to near-zero stock price (net entry ${strike - premium:.2f})",
        breakeven=f"${strike - premium:.2f}",
    )


def apply_position_sizing(
    idea: CandidateTrade,
    account_size: float,
    max_risk_per_trade_pct: float,
    shares_owned: int,
    cash_per_trade: float,
    max_premium_budget: float,
) -> Optional[CandidateTrade]:
    risk_budget = account_size * (max_risk_per_trade_pct / 100)

    if idea.strategy == "naked_call":
        per_contract_risk = idea.premium * 100
        contracts_by_risk = math.floor(risk_budget / per_contract_risk) if per_contract_risk > 0 else 0
        contracts_by_premium = math.floor(max_premium_budget / per_contract_risk) if per_contract_risk > 0 else 0
        contracts = max(0, min(contracts_by_risk, contracts_by_premium))
        if contracts < 1:
            return None

        idea.contracts = contracts
        idea.est_max_loss_dollars = round(contracts * per_contract_risk, 2)
        idea.capital_required = round(contracts * per_contract_risk, 2)
        idea.trade_plan = (
            f"Buy {contracts} call contract(s) expiring {idea.expiry}. "
            "Target 40-60% gain; cut at 50% premium loss."
        )
        return idea

    if idea.strategy == "covered_call":
        available_contracts = shares_owned // 100
        if available_contracts < 1:
            return None

        # Covered calls still carry stock downside risk; use a conservative proxy for risk budgeting.
        per_contract_est_risk = (idea.spot - idea.premium) * 100 * 0.18
        contracts_by_risk = math.floor(risk_budget / per_contract_est_risk) if per_contract_est_risk > 0 else 0
        contracts = max(0, min(available_contracts, max(1, contracts_by_risk)))
        if contracts < 1:
            return None

        idea.contracts = contracts
        idea.est_max_loss_dollars = round(contracts * per_contract_est_risk, 2)
        idea.capital_required = 0.0
        idea.trade_plan = (
            f"Sell {contracts} covered call contract(s) against {contracts * 100} owned shares, expiry {idea.expiry}."
        )
        return idea

    if idea.strategy == "cash_secured_put":
        per_contract_collateral = idea.strike * 100
        per_contract_est_risk = (idea.strike - idea.premium) * 100

        contracts_by_cash = math.floor(cash_per_trade / per_contract_collateral) if per_contract_collateral > 0 else 0
        contracts_by_risk = math.floor(risk_budget / per_contract_est_risk) if per_contract_est_risk > 0 else 0
        contracts = max(0, min(contracts_by_cash, contracts_by_risk))
        if contracts < 1:
            return None

        idea.contracts = contracts
        idea.capital_required = round(contracts * per_contract_collateral, 2)
        idea.est_max_loss_dollars = round(contracts * per_contract_est_risk, 2)
        idea.trade_plan = (
            f"Sell {contracts} cash-secured put contract(s), expiry {idea.expiry}; hold ${idea.capital_required:,.0f} collateral."
        )
        return idea

    return None


def fetch_history_with_retry(tk: yf.Ticker, ticker: str, retries: int = 3) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            hist = tk.history(period="6mo", interval="1d", auto_adjust=False)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                return hist
            # Fallback path for flaky ticker endpoints.
            fallback = yf.download(
                ticker,
                period="6mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if isinstance(fallback, pd.DataFrame) and not fallback.empty:
                return fallback
        except Exception as exc:
            last_exc = exc
        time.sleep((0.45 * (attempt + 1)) + random.uniform(0.05, 0.25))
    if last_exc:
        raise last_exc
    raise RuntimeError("No recent price history")


def fetch_options_with_retry(tk: yf.Ticker, expiry: str, retries: int = 3):
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return tk.option_chain(expiry)
        except Exception as exc:
            last_exc = exc
            time.sleep((0.55 * (attempt + 1)) + random.uniform(0.1, 0.35))
    if last_exc:
        raise last_exc
    raise RuntimeError("Option chain unavailable")


def is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {str(exc)}".lower()
    return "ratelimit" in text or "rate limit" in text or "too many requests" in text


def fetch_expiries_with_retry(tk: yf.Ticker, retries: int = 4) -> List[str]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            expiries = list(tk.options)
            if expiries:
                return expiries
            raise RuntimeError("No expiries returned")
        except Exception as exc:
            last_exc = exc
            if is_rate_limit_error(exc):
                time.sleep((1.2 * (attempt + 1)) + random.uniform(0.2, 0.6))
            else:
                time.sleep((0.45 * (attempt + 1)) + random.uniform(0.05, 0.2))
    if last_exc:
        raise last_exc
    raise RuntimeError("Expiry fetch failed")


def http_get_json(url: str, headers: Dict[str, str], timeout: int = 25) -> Dict:
    req = request.Request(url, headers=headers, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_tradier_expiries(symbol: str, tradier_token: str) -> List[str]:
    base = "https://api.tradier.com/v1/markets/options/expirations"
    query = urlencode({"symbol": symbol, "includeAllRoots": "true", "strikes": "false"})
    headers = {"Authorization": f"Bearer {tradier_token}", "Accept": "application/json"}
    data = http_get_json(f"{base}?{query}", headers=headers)
    dates = data.get("expirations", {}).get("date", [])
    if isinstance(dates, str):
        return [dates]
    if isinstance(dates, list):
        return [str(d) for d in dates]
    return []


def fetch_tradier_chain(symbol: str, expiry: str, tradier_token: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = "https://api.tradier.com/v1/markets/options/chains"
    query = urlencode({"symbol": symbol, "expiration": expiry, "greeks": "false"})
    headers = {"Authorization": f"Bearer {tradier_token}", "Accept": "application/json"}
    data = http_get_json(f"{base}?{query}", headers=headers)
    options = data.get("options", {}).get("option", [])
    if isinstance(options, dict):
        options = [options]
    if not isinstance(options, list):
        options = []

    rows = []
    for item in options:
        rows.append(
            {
                "strike": float(item.get("strike") or 0.0),
                "bid": float(item.get("bid") or 0.0),
                "ask": float(item.get("ask") or 0.0),
                "lastPrice": float(item.get("last") or 0.0),
                "type": str(item.get("option_type", "")).lower(),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    calls = df[df["type"] == "call"][["strike", "bid", "ask", "lastPrice"]].copy()
    puts = df[df["type"] == "put"][["strike", "bid", "ask", "lastPrice"]].copy()
    return calls, puts


def build_trade_idea(
    ticker: str,
    risk_level: str,
    allowed_strategies: List[str],
    shares_owned: int,
    cash_per_trade: float,
    max_premium_budget: float,
    account_size: float,
    max_risk_per_trade_pct: float,
    as_of: date,
    use_earnings_filter: bool,
    earnings_buffer_days: int,
    avoid_if_before_expiry: bool,
    options_provider: str,
    tradier_token: str,
) -> Tuple[Optional[CandidateTrade], str]:
    tk = yf.Ticker(ticker)

    try:
        hist = fetch_history_with_retry(tk, ticker=ticker, retries=3)
    except Exception as exc:
        return None, f"History fetch failed: {type(exc).__name__}: {str(exc)[:90]}"
    if hist.empty:
        return None, "No recent price history"

    sig = compute_signal_strength(hist)
    if options_provider == "tradier":
        if not tradier_token.strip():
            return None, "Tradier token missing. Add TRADIER_TOKEN secret or paste token in app."
        try:
            expiries = fetch_tradier_expiries(ticker, tradier_token.strip())
        except Exception as exc:
            return None, f"Tradier expiry fetch failed: {type(exc).__name__}: {str(exc)[:90]}"
    else:
        try:
            expiries = fetch_expiries_with_retry(tk, retries=4)
        except Exception as exc:
            if is_rate_limit_error(exc):
                return (
                    None,
                    "Expiry fetch rate-limited by Yahoo Finance. Switch provider to Tradier or retry later.",
                )
            return None, f"Expiry fetch failed: {type(exc).__name__}: {str(exc)[:90]}"

    expiry = pick_expiry(expiries, risk_level, as_of)
    if not expiry:
        return None, "No valid option expiry"

    earnings_dt = extract_next_earnings_date(tk)
    if use_earnings_filter and earnings_too_close(
        earnings_dt=earnings_dt,
        as_of=as_of,
        expiry=expiry,
        buffer_days=earnings_buffer_days,
        avoid_if_before_expiry=avoid_if_before_expiry,
    ):
        if earnings_dt:
            return None, f"Earnings filter: next earnings on {earnings_dt}"
        return None, "Earnings filter: near event window"

    if options_provider == "tradier":
        try:
            calls, puts = fetch_tradier_chain(ticker, expiry, tradier_token.strip())
        except Exception as exc:
            return None, f"Tradier chain failed: {type(exc).__name__}: {str(exc)[:90]}"
    else:
        try:
            chain = fetch_options_with_retry(tk, expiry=expiry, retries=4)
        except Exception as exc:
            if is_rate_limit_error(exc):
                return (
                    None,
                    "Option chain rate-limited by Yahoo Finance. Switch provider to Tradier or retry later.",
                )
            return None, f"Option chain failed: {type(exc).__name__}: {str(exc)[:90]}"
        calls = chain.calls.copy()
        puts = chain.puts.copy()

    if calls.empty and puts.empty:
        return None, "Option chain data empty for selected expiry"

    candidates: List[CandidateTrade] = []

    if "naked_call" in allowed_strategies:
        idea = generate_naked_call(ticker, calls, sig, risk_level, expiry, max_premium_budget=max_premium_budget)
        if idea:
            candidates.append(idea)

    if "covered_call" in allowed_strategies:
        idea = generate_covered_call(ticker, calls, sig, risk_level, expiry, shares_owned=shares_owned)
        if idea:
            candidates.append(idea)

    if "cash_secured_put" in allowed_strategies:
        idea = generate_cash_secured_put(ticker, puts, sig, risk_level, expiry, cash_per_trade=cash_per_trade)
        if idea:
            candidates.append(idea)

    if not candidates:
        return None, "No strategy qualified under current filters"

    best = sorted(candidates, key=lambda c: c.score, reverse=True)[0]
    sized = apply_position_sizing(
        idea=best,
        account_size=account_size,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        shares_owned=shares_owned,
        cash_per_trade=cash_per_trade,
        max_premium_budget=max_premium_budget,
    )
    if not sized:
        return None, "Position sizing rejected (risk/capital constraints)"

    sized.earnings_date = earnings_dt.isoformat() if earnings_dt else "N/A"
    return sized, ""


def ideas_to_df(ideas: List[CandidateTrade]) -> pd.DataFrame:
    rows = []
    for idea in ideas:
        rows.append(
            {
                "Ticker": idea.ticker,
                "Strategy": STRATEGY_LABELS[idea.strategy],
                "Confidence": idea.confidence,
                "Score": round(idea.score, 3),
                "Contracts": idea.contracts,
                "Spot": round(idea.spot, 2),
                "Expiry": idea.expiry,
                "Strike": round(idea.strike, 2),
                "Mid Premium": round(idea.premium, 2),
                "Capital Req ($)": round(idea.capital_required, 2),
                "Est Max Loss ($)": round(idea.est_max_loss_dollars, 2),
                "Breakeven": idea.breakeven,
                "Next Earnings": idea.earnings_date,
            }
        )
    return pd.DataFrame(rows)


def build_trade_tickets_df(ideas: List[CandidateTrade], next_day: date) -> pd.DataFrame:
    rows = []
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx, idea in enumerate(ideas, start=1):
        rows.append(
            {
                "TicketID": f"TKT-{next_day.strftime('%Y%m%d')}-{idx:03d}",
                "GeneratedAt": now_text,
                "TradeDate": str(next_day),
                "Ticker": idea.ticker,
                "Strategy": STRATEGY_LABELS[idea.strategy],
                "Action": "BUY" if idea.strategy == "naked_call" else "SELL",
                "Contracts": idea.contracts,
                "Expiry": idea.expiry,
                "Strike": round(idea.strike, 2),
                "LimitPremium": round(idea.premium, 2),
                "EstimatedCapital": round(idea.capital_required, 2),
                "EstimatedMaxLoss": round(idea.est_max_loss_dollars, 2),
                "Breakeven": idea.breakeven,
                "Rationale": idea.rationale,
                "Plan": idea.trade_plan,
            }
        )
    return pd.DataFrame(rows)


def build_pdf_report_bytes(
    ideas: List[CandidateTrade],
    next_day: date,
    risk_level: str,
    account_size: float,
    max_risk_per_trade_pct: float,
) -> bytes:
    def safe_pdf_text(value: str) -> str:
        # Core PDF fonts are latin-1; normalize dynamic content defensively.
        if value is None:
            return ""
        text = str(value)
        text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
        return text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, safe_pdf_text("Blue-Chip Tech Options Trade Tickets"), ln=1)

    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(
        0,
        6,
        safe_pdf_text(
            f"Trade Date: {next_day} | Risk Mode: {risk_level} | Account Size: ${account_size:,.0f} | Max Risk/Trade: {max_risk_per_trade_pct:.2f}%"
        ),
    )
    pdf.ln(2)

    for idx, idea in enumerate(ideas, start=1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(
            0,
            7,
            safe_pdf_text(f"{idx}. {idea.ticker} - {STRATEGY_LABELS[idea.strategy]}"),
            ln=1,
        )
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(
            0,
            6,
            safe_pdf_text(
                f"Contracts: {idea.contracts} | Expiry: {idea.expiry} | Strike: {idea.strike:.2f} | Mid: {idea.premium:.2f} | "
                f"Capital: ${idea.capital_required:,.0f} | Est Max Loss: ${idea.est_max_loss_dollars:,.0f}"
            ),
        )
        pdf.multi_cell(0, 6, safe_pdf_text(f"Plan: {idea.trade_plan}"))
        pdf.multi_cell(0, 6, safe_pdf_text(f"Rationale: {idea.rationale}"))
        pdf.ln(1)

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1", errors="ignore")
    return bytes(raw)


def fallback_ai_memo(idea: CandidateTrade, risk_level: str, next_day: date) -> str:
    return (
        f"## AI Trade Committee Memo ({idea.ticker})\n\n"
        f"### 1) Executive Summary\n"
        f"- Trade date: {next_day}\n"
        f"- Strategy: {STRATEGY_LABELS[idea.strategy]}\n"
        f"- Risk mode selected: {risk_level}\n"
        f"- Suggested size: {idea.contracts} contract(s)\n"
        f"- Why this fit: {idea.rationale}\n\n"
        f"### 2) Key Risks To Monitor\n"
        f"- Event risk: next earnings date {idea.earnings_date}\n"
        f"- Liquidity/slippage risk around open and close\n"
        f"- Strategy-specific risk: {idea.max_risk}\n\n"
        f"### 3) Pre-Market Checklist\n"
        f"- Confirm spread quality and avoid wide bid/ask markets.\n"
        f"- Validate no major overnight news changes the thesis.\n"
        f"- Place limit order near model premium ({idea.premium:.2f}).\n\n"
        f"### 4) Intraday Management Checklist\n"
        f"- Reassess if spot breaks thesis level by >2% versus entry plan.\n"
        f"- Avoid emotional averaging; follow the sizing limit.\n"
        f"- Track position versus breakeven: {idea.breakeven}\n\n"
        f"### 5) Beginner-Friendly Explanation\n"
        f"This trade is chosen because it balances opportunity and controlled exposure under a {risk_level} profile. "
        f"The plan is to follow the entry and risk rules strictly, not to predict every price move.\n\n"
        f"### 6) Desk-Style Analyst Note\n"
        f"Maintain discipline on execution quality, pre-defined sizing, and event risk gates. "
        f"Only proceed if opening market conditions preserve expected reward/risk.\n"
    )


def call_openai_memo(
    api_key: str,
    model: str,
    idea: CandidateTrade,
    risk_level: str,
    next_day: date,
) -> str:
    prompt = (
        "You are a sell-side options strategist writing an internal trade committee memo.\\n"
        "Write concise markdown with these headings exactly:\\n"
        "1) Executive Summary\\n"
        "2) Key Risks To Monitor\\n"
        "3) Pre-Market Checklist\\n"
        "4) Intraday Management Checklist\\n"
        "5) Beginner-Friendly Explanation\\n"
        "6) Desk-Style Analyst Note\\n\\n"
        f"Trade date: {next_day}\\n"
        f"Ticker: {idea.ticker}\\n"
        f"Strategy: {STRATEGY_LABELS[idea.strategy]}\\n"
        f"Risk level: {risk_level}\\n"
        f"Contracts: {idea.contracts}\\n"
        f"Spot: {idea.spot:.2f}\\n"
        f"Strike: {idea.strike:.2f}\\n"
        f"Mid premium: {idea.premium:.2f}\\n"
        f"Breakeven: {idea.breakeven}\\n"
        f"Estimated capital required: ${idea.capital_required:,.2f}\\n"
        f"Estimated max loss: ${idea.est_max_loss_dollars:,.2f}\\n"
        f"Rationale: {idea.rationale}\\n"
        f"Plan: {idea.trade_plan}\\n"
        f"Next earnings date: {idea.earnings_date}\\n\\n"
        "Constraints: Keep it practical, no hype, no guarantees, and include concrete numeric trigger examples."
    )

    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 900,
    }
    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=35) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(f"OpenAI HTTP error: {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError("OpenAI network error") from exc

    data = json.loads(body)
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    output = data.get("output", [])
    chunks: List[str] = []
    if isinstance(output, list):
        for item in output:
            content = item.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
    text = "\\n".join(chunks).strip()
    if text:
        return text
    raise RuntimeError("OpenAI response contained no text")


def generate_ai_memo(
    idea: CandidateTrade,
    risk_level: str,
    next_day: date,
    api_key: str,
    model: str,
) -> Tuple[str, str]:
    key = api_key.strip()
    if not key:
        try:
            key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        except Exception:
            key = ""

    if not key:
        return fallback_ai_memo(idea, risk_level, next_day), "offline"

    try:
        memo = call_openai_memo(key, model, idea, risk_level, next_day)
        return memo, "openai"
    except Exception:
        return fallback_ai_memo(idea, risk_level, next_day), "fallback"


def fallback_chat_reply(user_text: str, context_tickers: List[str], risk_level: str, persona: str) -> str:
    base = (
        f"[{persona}] Risk mode is {risk_level}. "
        f"Current context tickers: {', '.join(context_tickers) if context_tickers else 'none'}."
    )
    if "explain" in user_text.lower():
        return (
            f"{base} Focus on three checks: thesis quality, event risk, and position sizing. "
            "Use limit orders, pre-define max loss, and avoid forced trades when liquidity is poor."
        )
    if "entry" in user_text.lower() or "when" in user_text.lower():
        return (
            f"{base} Entry framework: confirm spread quality, no thesis-breaking news, and that spot remains within plan levels. "
            "If one condition fails, skip."
        )
    return (
        f"{base} Coaching guidance: keep trade size within your risk budget, avoid earnings-event exposure unless intentional, "
        "and log your decision before placing orders."
    )


def call_openai_chat(
    api_key: str,
    model: str,
    user_text: str,
    risk_level: str,
    context_tickers: List[str],
    chat_history: List[Dict[str, str]],
    persona: str,
) -> str:
    recent = chat_history[-10:]
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    prompt = (
        "You are a professional options coach assisting with conservative/moderate strategy decisions. "
        "Be practical, concise, and risk-aware. Do not give guarantees.\n\n"
        f"Personality mode: {persona}\n"
        f"Personality instructions: {CHAT_PERSONAS.get(persona, CHAT_PERSONAS['Desk Strategist'])}\n"
        f"Risk mode: {risk_level}\n"
        f"Context tickers: {', '.join(context_tickers) if context_tickers else 'none'}\n"
        f"Recent chat:\n{history_text}\n\n"
        f"User message: {user_text}\n\n"
        "Respond with: (1) direct answer, (2) risk check, (3) next action."
    )
    payload = {"model": model, "input": prompt, "max_output_tokens": 500}
    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=35) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(f"OpenAI HTTP error: {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError("OpenAI network error") from exc

    data = json.loads(body)
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    output = data.get("output", [])
    chunks: List[str] = []
    if isinstance(output, list):
        for item in output:
            content = item.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
    text = "\n".join(chunks).strip()
    if text:
        return text
    raise RuntimeError("OpenAI chat response contained no text")


def render_header(next_day: date) -> None:
    st.set_page_config(page_title="Blue-Chip Tech Options Analyst", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1.25rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 1200px;
        }
        div[data-testid="stDataFrame"] {
            overflow-x: auto;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.5rem;
                padding-left: 0.6rem;
                padding-right: 0.6rem;
            }
            h1 {
                font-size: 1.35rem !important;
                line-height: 1.25 !important;
            }
            h2, h3 {
                font-size: 1.05rem !important;
            }
            p, label, div, span {
                font-size: 0.92rem !important;
            }
            .stButton > button {
                width: 100%;
                min-height: 2.6rem;
                font-size: 0.95rem;
            }
            div[data-testid="stSidebar"] .stButton > button {
                width: 100%;
            }
            div[data-testid="stDataFrame"] iframe,
            div[data-testid="stDataFrame"] > div {
                min-width: 680px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Blue-Chip Tech Options Analyst")
    st.caption(
        f"Generates next-trading-day options ideas for {next_day} using blue-chip tech names and risk filters."
    )
    st.warning(
        "Educational use only. This is not investment advice. Options involve significant risk, including loss of principal."
    )


def main() -> None:
    as_of = date.today()
    next_day = next_business_day(as_of)
    render_header(next_day)

    st.subheader("Trade Setup")
    st.caption("All filters are visible here for better mobile usability.")

    risk_col, portfolio_col = st.columns(2)
    with risk_col:
        risk_level = st.selectbox("Risk Level", ["conservative", "moderate"], index=0)
    with portfolio_col:
        account_size = st.number_input(
            "Portfolio Size ($)",
            min_value=5000.0,
            max_value=10000000.0,
            value=150000.0,
            step=1000.0,
        )

    tickers = st.multiselect(
        "Blue-Chip Tech Universe",
        BLUE_CHIP_TECH,
        default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    )

    allowed_strategy_labels = st.multiselect(
        "Allowed Strategies",
        list(STRATEGY_LABELS.values()),
        default=[STRATEGY_LABELS["covered_call"], STRATEGY_LABELS["cash_secured_put"]],
    )

    strategy_reverse = {v: k for k, v in STRATEGY_LABELS.items()}
    allowed_strategies = [strategy_reverse[label] for label in allowed_strategy_labels]

    st.markdown("Shares owned per ticker (for covered calls)")
    shares_default = pd.DataFrame({"Ticker": tickers, "SharesOwned": [0] * len(tickers)})
    shares_df = st.data_editor(
        shares_default,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
    )
    shares_map = {row["Ticker"]: int(row["SharesOwned"]) for _, row in shares_df.iterrows()}

    default_risk = 1.0 if risk_level == "conservative" else 2.0
    risk_size_col, csp_col, call_col = st.columns(3)
    with risk_size_col:
        max_risk_per_trade_pct = st.slider("Max Risk Per Trade (%)", 0.25, 5.0, float(default_risk), 0.25)
    with csp_col:
        cash_per_trade = st.number_input(
            "Cash reserved per CSP trade ($)",
            min_value=1000.0,
            max_value=500000.0,
            value=25000.0,
            step=500.0,
        )
    with call_col:
        max_premium_budget = st.number_input(
            "Max premium budget for naked calls ($)",
            min_value=100.0,
            max_value=25000.0,
            value=750.0,
            step=50.0,
        )

    st.markdown("Earnings Filter")
    earn_col1, earn_col2 = st.columns(2)
    with earn_col1:
        use_earnings_filter = st.checkbox("Avoid trades around earnings", value=True)
        avoid_if_before_expiry = st.checkbox("Avoid if earnings falls before option expiry", value=True)
    with earn_col2:
        earnings_buffer_days = st.slider("Earnings proximity window (days)", 3, 21, 7, 1)

    st.markdown("Options Data Provider")
    provider_col1, provider_col2 = st.columns(2)
    with provider_col1:
        options_provider = st.selectbox(
            "Provider",
            ["yahoo", "tradier"],
            index=0,
            help="Use Tradier if Yahoo rate limits options expiries/chains on Streamlit Cloud.",
        )
    with provider_col2:
        tradier_token_input = st.text_input(
            "Tradier Token (optional)",
            type="password",
            help="Used when Provider is set to tradier. You can also store TRADIER_TOKEN in Streamlit secrets.",
        )

    tradier_token = tradier_token_input.strip()
    if not tradier_token:
        try:
            tradier_token = str(st.secrets.get("TRADIER_TOKEN", "")).strip()
        except Exception:
            tradier_token = ""

    if not tickers:
        st.info("Select at least one ticker.")
        return

    if not allowed_strategies:
        st.info("Select at least one strategy.")
        return

    if "analysis_payload" not in st.session_state:
        st.session_state.analysis_payload = None
    if "ai_memos" not in st.session_state:
        st.session_state.ai_memos = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "I am your options coach bot. Ask about setups, entries, sizing, or risk controls.",
            }
        ]

    if st.button("Analyze and Generate Next-Day Trades", type="primary"):
        ideas: List[CandidateTrade] = []
        skipped: Dict[str, str] = {}
        rate_limited = False

        progress = st.progress(0)
        for idx, ticker in enumerate(tickers):
            try:
                idea, reason = build_trade_idea(
                    ticker=ticker,
                    risk_level=risk_level,
                    allowed_strategies=allowed_strategies,
                    shares_owned=shares_map.get(ticker, 0),
                    cash_per_trade=float(cash_per_trade),
                    max_premium_budget=float(max_premium_budget),
                    account_size=float(account_size),
                    max_risk_per_trade_pct=float(max_risk_per_trade_pct),
                    as_of=as_of,
                    use_earnings_filter=use_earnings_filter,
                    earnings_buffer_days=int(earnings_buffer_days),
                    avoid_if_before_expiry=avoid_if_before_expiry,
                    options_provider=options_provider,
                    tradier_token=tradier_token,
                )
                if idea:
                    ideas.append(idea)
                else:
                    skipped[ticker] = reason
                    if isinstance(reason, str) and "rate-limit" in reason.lower():
                        rate_limited = True
            except Exception as exc:
                skipped[ticker] = f"Unhandled error: {type(exc).__name__}: {str(exc)[:120]}"

            progress.progress((idx + 1) / len(tickers))
            if rate_limited:
                break
            time.sleep(0.15)

        if not ideas:
            st.error("No qualifying trades found with current constraints. Adjust filters or risk limits.")
            if rate_limited:
                st.warning(
                    "Yahoo Finance rate-limited this request. Try again in 1-3 minutes and analyze fewer tickers at once."
                )
            if skipped:
                st.write("Skip reasons:")
                st.table(pd.DataFrame({"Ticker": list(skipped.keys()), "Reason": list(skipped.values())}))
            st.session_state.analysis_payload = None
            return

        ideas = sorted(ideas, key=lambda x: x.score, reverse=True)
        st.session_state.analysis_payload = {
            "ideas": ideas,
            "skipped": skipped,
            "risk_level": risk_level,
            "account_size": float(account_size),
            "max_risk_per_trade_pct": float(max_risk_per_trade_pct),
            "next_day": str(next_day),
        }
        st.session_state.ai_memos = {}

    payload = st.session_state.analysis_payload
    if not payload:
        return

    ideas = payload["ideas"]
    skipped = payload["skipped"]
    result_risk_level = payload["risk_level"]
    result_account_size = payload["account_size"]
    result_max_risk_pct = payload["max_risk_per_trade_pct"]
    result_next_day = datetime.strptime(payload["next_day"], "%Y-%m-%d").date()

    ideas_df = ideas_to_df(ideas)
    tickets_df = build_trade_tickets_df(ideas, result_next_day)

    st.subheader("Next-Day Trade Ideas")
    st.dataframe(ideas_df, use_container_width=True, height=320)

    for idea in ideas:
        with st.expander(f"{idea.ticker}: {STRATEGY_LABELS[idea.strategy]} ({idea.confidence} confidence)"):
            st.write(f"Rationale: {idea.rationale}")
            st.write(f"Plan: {idea.trade_plan}")
            st.write(f"Contracts: {idea.contracts}")
            st.write(f"Estimated Capital Required: ${idea.capital_required:,.2f}")
            st.write(f"Estimated Max Loss: ${idea.est_max_loss_dollars:,.2f}")
            st.write(f"Max Profit: {idea.max_profit}")
            st.write(f"Max Risk: {idea.max_risk}")
            st.write(f"Breakeven: {idea.breakeven}")
            st.write(f"Next Earnings: {idea.earnings_date}")

    st.subheader("AI Memo")
    memo_col1, memo_col2, memo_col3 = st.columns([2, 2, 1])
    with memo_col1:
        selected_ticker = st.selectbox(
            "Memo For Ticker",
            [idea.ticker for idea in ideas],
            key="memo_ticker",
        )
    with memo_col2:
        memo_model = st.text_input("AI Model", value="gpt-4.1-mini", key="memo_model")
    with memo_col3:
        st.write("")
        st.write("")
        generate_memo_clicked = st.button("Generate AI Memo", type="secondary")

    api_key_input = st.text_input(
        "OpenAI API Key (optional, uses offline memo if blank)",
        type="password",
        key="memo_api_key",
    )

    if generate_memo_clicked:
        selected_idea = next((item for item in ideas if item.ticker == selected_ticker), None)
        if selected_idea:
            memo_text, memo_source = generate_ai_memo(
                idea=selected_idea,
                risk_level=result_risk_level,
                next_day=result_next_day,
                api_key=api_key_input,
                model=memo_model.strip() or "gpt-4.1-mini",
            )
            st.session_state.ai_memos[selected_ticker] = {
                "memo": memo_text,
                "source": memo_source,
            }

    if selected_ticker in st.session_state.ai_memos:
        memo_block = st.session_state.ai_memos[selected_ticker]
        source_label = memo_block["source"]
        if source_label == "openai":
            st.caption("Memo source: OpenAI")
        elif source_label == "fallback":
            st.caption("Memo source: OpenAI fallback template (API call failed)")
        else:
            st.caption("Memo source: Offline template")
        st.markdown(memo_block["memo"])
        st.download_button(
            f"Download Memo ({selected_ticker})",
            data=memo_block["memo"].encode("utf-8"),
            file_name=f"ai_memo_{selected_ticker}_{result_next_day}.md",
            mime="text/markdown",
        )

    st.subheader("AI Chatbot")
    chat_settings_col1, chat_settings_col2, chat_settings_col3 = st.columns(3)
    with chat_settings_col1:
        chat_model = st.text_input("Chat Model", value="gpt-4.1-mini", key="chat_model")
    with chat_settings_col2:
        chat_api_key_input = st.text_input(
            "OpenAI API Key for Chat (optional)",
            type="password",
            key="chat_api_key",
        )
    with chat_settings_col3:
        chat_persona = st.selectbox("Chat Personality", list(CHAT_PERSONAS.keys()), index=1, key="chat_persona")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_chat = st.chat_input("Ask the coach bot about today's trade setups...")
    if user_chat:
        st.session_state.chat_history.append({"role": "user", "content": user_chat})
        with st.chat_message("user"):
            st.markdown(user_chat)

        key = chat_api_key_input.strip()
        if not key:
            try:
                key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
            except Exception:
                key = ""

        context_tickers = [idea.ticker for idea in ideas]
        if key:
            try:
                bot_reply = call_openai_chat(
                    api_key=key,
                    model=chat_model.strip() or "gpt-4.1-mini",
                    user_text=user_chat,
                    risk_level=result_risk_level,
                    context_tickers=context_tickers,
                    chat_history=st.session_state.chat_history,
                    persona=chat_persona,
                )
            except Exception:
                bot_reply = fallback_chat_reply(user_chat, context_tickers, result_risk_level, chat_persona)
        else:
            bot_reply = fallback_chat_reply(user_chat, context_tickers, result_risk_level, chat_persona)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    chat_export = "\n\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history]
    )
    st.download_button(
        "Download Chat Transcript",
        data=chat_export.encode("utf-8"),
        file_name=f"coach_chat_{result_next_day}.txt",
        mime="text/plain",
    )

    st.subheader("Trade Ticket Export")
    st.dataframe(tickets_df, use_container_width=True, height=320)

    csv_bytes = tickets_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Trade Tickets (CSV)",
        data=csv_bytes,
        file_name=f"next_day_trade_tickets_{result_next_day}.csv",
        mime="text/csv",
    )

    try:
        pdf_bytes = build_pdf_report_bytes(
            ideas=ideas,
            next_day=result_next_day,
            risk_level=result_risk_level,
            account_size=float(result_account_size),
            max_risk_per_trade_pct=float(result_max_risk_pct),
        )
        st.download_button(
            "Download Trade Tickets (PDF)",
            data=pdf_bytes,
            file_name=f"next_day_trade_tickets_{result_next_day}.pdf",
            mime="application/pdf",
        )
    except Exception as exc:
        st.warning(f"PDF export is temporarily unavailable: {type(exc).__name__}. CSV export still works.")

    if skipped:
        st.caption("Skipped tickers and reasons")
        st.table(pd.DataFrame({"Ticker": list(skipped.keys()), "Reason": list(skipped.values())}))


if __name__ == "__main__":
    main()
