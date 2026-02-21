import math
import json
import time
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
    "naked_put": "Buy Naked Put",
    "covered_call": "Sell Covered Call",
    "cash_secured_put": "Sell Cash-Secured Put",
    "bull_call_spread": "Buy Bull Call Spread",
    "bear_put_spread": "Buy Bear Put Spread",
}

CHAT_PERSONAS: Dict[str, str] = {
    "Calm Mentor": "Speak calmly, clearly, and with patience. Reduce anxiety while reinforcing discipline.",
    "Desk Strategist": "Speak like a pragmatic trading-desk coach: concise, direct, and execution-focused.",
    "Tough Coach": "Be blunt but constructive. Challenge weak logic and force clear risk rules.",
}

DEFAULT_OPENAI_MODEL = "gpt-5.2"
OPENAI_MODEL_FALLBACKS = ["gpt-5-mini", "gpt-4.1-mini"]


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
    option_volume: int = 0
    option_oi: int = 0
    spread_pct: float = 0.0
    tradability_score: float = 0.0
    provider_used: str = "unknown"


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


def with_liquidity_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "openInterest" not in out.columns:
        out["openInterest"] = 0
    if "volume" not in out.columns:
        out["volume"] = 0
    out["openInterest"] = pd.to_numeric(out["openInterest"], errors="coerce").fillna(0).astype(int)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    out["bid"] = pd.to_numeric(out.get("bid", 0), errors="coerce").fillna(0.0)
    out["ask"] = pd.to_numeric(out.get("ask", 0), errors="coerce").fillna(0.0)
    out["mid"] = out.apply(
        lambda r: option_mid(float(r.get("bid", 0)), float(r.get("ask", 0)), float(r.get("lastPrice", 0))),
        axis=1,
    )
    out["spread_pct"] = out.apply(
        lambda r: ((float(r["ask"]) - float(r["bid"])) / float(r["mid"]) * 100.0)
        if float(r["mid"]) > 0
        else 999.0,
        axis=1,
    )
    return out


def apply_liquidity_filter(df: pd.DataFrame, min_oi: int, min_volume: int, max_spread_pct: float) -> pd.DataFrame:
    out = with_liquidity_fields(df)
    return out[
        (out["openInterest"] >= int(min_oi))
        & (out["volume"] >= int(min_volume))
        & (out["spread_pct"] <= float(max_spread_pct))
        & (out["mid"] > 0.05)
    ].copy()


def compute_tradability_score(base_score: float, spread_pct: float, oi: int, volume: int) -> float:
    spread_component = max(0.0, 100.0 - (spread_pct * 5.0))
    oi_component = min(100.0, (oi / 1000.0) * 100.0)
    vol_component = min(100.0, (volume / 500.0) * 100.0)
    strategy_component = max(0.0, min(100.0, base_score * 100.0))
    return round((0.35 * spread_component) + (0.3 * oi_component) + (0.2 * vol_component) + (0.15 * strategy_component), 1)


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
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
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

    pool = apply_liquidity_filter(pool, min_oi=min_oi, min_volume=min_volume, max_spread_pct=max_spread_pct)
    pool = pool[(pool["mid"] * 100 <= max_premium_budget)]
    if pool.empty:
        return None

    pool["moneyness"] = (pool["strike"] - spot).abs()
    best = pool.sort_values(["moneyness", "mid"], ascending=[True, True]).iloc[0]

    premium = float(best["mid"])
    strike = float(best["strike"])
    oi = int(best.get("openInterest", 0))
    vol = int(best.get("volume", 0))
    spread_pct = float(best.get("spread_pct", 999.0))
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
        option_volume=vol,
        option_oi=oi,
        spread_pct=spread_pct,
        tradability_score=compute_tradability_score(score, spread_pct, oi, vol),
    )


def generate_naked_put(
    ticker: str,
    puts: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    max_premium_budget: float,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
) -> Optional[CandidateTrade]:
    if risk_level == "conservative":
        return None
    if sig["total"] > -0.1:
        return None

    spot = sig["spot"]
    min_strike = spot * 0.95
    max_strike = spot * 1.00

    pool = puts[(puts["strike"] >= min_strike) & (puts["strike"] <= max_strike)].copy()
    if pool.empty:
        return None

    pool = apply_liquidity_filter(pool, min_oi=min_oi, min_volume=min_volume, max_spread_pct=max_spread_pct)
    pool = pool[(pool["mid"] * 100 <= max_premium_budget)]
    if pool.empty:
        return None

    pool["moneyness"] = (pool["strike"] - spot).abs()
    best = pool.sort_values(["moneyness", "mid"], ascending=[True, True]).iloc[0]

    premium = float(best["mid"])
    strike = float(best["strike"])
    oi = int(best.get("openInterest", 0))
    vol = int(best.get("volume", 0))
    spread_pct = float(best.get("spread_pct", 999.0))
    score = min(0.9, 0.52 + abs(sig["total"]) * 0.42)

    return CandidateTrade(
        ticker=ticker,
        strategy="naked_put",
        risk_level=risk_level,
        score=score,
        confidence=confidence_label(score),
        expiry=expiry,
        strike=strike,
        premium=premium,
        spot=spot,
        rationale=f"Bearish momentum setup (RSI {sig['rsi']:.1f}) with defined premium-at-risk long put.",
        trade_plan=f"Buy puts near-the-money for {expiry}. Take profits around 40-60% gain or cut at 50% premium loss.",
        max_profit="Substantial downside participation",
        max_risk=f"Premium paid per contract: ${premium * 100:.2f}",
        breakeven=f"${strike - premium:.2f}",
        option_volume=vol,
        option_oi=oi,
        spread_pct=spread_pct,
        tradability_score=compute_tradability_score(score, spread_pct, oi, vol),
    )


def generate_covered_call(
    ticker: str,
    calls: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    shares_owned: int,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
) -> Optional[CandidateTrade]:
    if shares_owned < 100:
        return None

    spot = sig["spot"]
    otm = 0.05 if risk_level == "conservative" else 0.03

    pool = calls[calls["strike"] >= spot * (1 + otm)].copy()
    if pool.empty:
        return None

    pool = apply_liquidity_filter(pool, min_oi=min_oi, min_volume=min_volume, max_spread_pct=max_spread_pct)
    if pool.empty:
        return None

    pool["distance"] = (pool["strike"] / spot) - 1
    best = pool.sort_values(["distance", "mid"], ascending=[True, False]).iloc[0]

    premium = float(best["mid"])
    strike = float(best["strike"])
    oi = int(best.get("openInterest", 0))
    vol = int(best.get("volume", 0))
    spread_pct = float(best.get("spread_pct", 999.0))
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
        option_volume=vol,
        option_oi=oi,
        spread_pct=spread_pct,
        tradability_score=compute_tradability_score(score, spread_pct, oi, vol),
    )


def generate_bull_call_spread(
    ticker: str,
    calls: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    max_premium_budget: float,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
) -> Optional[CandidateTrade]:
    if sig["total"] < 0.15:
        return None

    spot = sig["spot"]
    calls_liq = apply_liquidity_filter(calls, min_oi=min_oi, min_volume=min_volume, max_spread_pct=max_spread_pct)
    if calls_liq.empty:
        return None

    long_leg_pool = calls_liq[(calls_liq["strike"] >= spot * 0.98) & (calls_liq["strike"] <= spot * 1.03)].copy()
    if long_leg_pool.empty:
        return None
    long_leg = long_leg_pool.sort_values(["strike"]).iloc[0]

    short_pool = calls_liq[calls_liq["strike"] > float(long_leg["strike"])].copy()
    short_pool = short_pool[short_pool["strike"] <= float(long_leg["strike"]) * 1.08]
    if short_pool.empty:
        return None
    short_leg = short_pool.sort_values(["strike"]).iloc[0]

    net_debit = float(long_leg["mid"]) - float(short_leg["mid"])
    width = float(short_leg["strike"]) - float(long_leg["strike"])
    if net_debit <= 0.05 or width <= net_debit:
        return None
    if net_debit * 100 > max_premium_budget:
        return None

    oi = min(int(long_leg.get("openInterest", 0)), int(short_leg.get("openInterest", 0)))
    vol = min(int(long_leg.get("volume", 0)), int(short_leg.get("volume", 0)))
    spread_pct = max(float(long_leg.get("spread_pct", 999.0)), float(short_leg.get("spread_pct", 999.0)))
    base_score = min(0.92, 0.58 + sig["total"] * 0.35)

    return CandidateTrade(
        ticker=ticker,
        strategy="bull_call_spread",
        risk_level=risk_level,
        score=base_score,
        confidence=confidence_label(base_score),
        expiry=expiry,
        strike=float(long_leg["strike"]),
        premium=net_debit,
        spot=spot,
        rationale="Bullish defined-risk debit spread to cap premium outlay while targeting upside.",
        trade_plan=(
            f"Buy {float(long_leg['strike']):.2f} call and sell {float(short_leg['strike']):.2f} call for {expiry}."
        ),
        max_profit=f"Approx ${((width - net_debit) * 100):.2f} per contract",
        max_risk=f"Net debit paid: ${net_debit * 100:.2f}",
        breakeven=f"${float(long_leg['strike']) + net_debit:.2f}",
        option_volume=vol,
        option_oi=oi,
        spread_pct=spread_pct,
        tradability_score=compute_tradability_score(base_score, spread_pct, oi, vol),
    )


def generate_cash_secured_put(
    ticker: str,
    puts: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    cash_per_trade: float,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
) -> Optional[CandidateTrade]:
    spot = sig["spot"]
    discount = 0.06 if risk_level == "conservative" else 0.04

    pool = puts[puts["strike"] <= spot * (1 - discount)].copy()
    if pool.empty:
        return None

    pool = apply_liquidity_filter(pool, min_oi=min_oi, min_volume=min_volume, max_spread_pct=max_spread_pct)
    pool = pool[(pool["strike"] * 100 <= cash_per_trade)]
    if pool.empty:
        return None

    pool["distance"] = 1 - (pool["strike"] / spot)
    best = pool.sort_values(["distance", "mid"], ascending=[True, False]).iloc[0]

    premium = float(best["mid"])
    strike = float(best["strike"])
    oi = int(best.get("openInterest", 0))
    vol = int(best.get("volume", 0))
    spread_pct = float(best.get("spread_pct", 999.0))
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
        option_volume=vol,
        option_oi=oi,
        spread_pct=spread_pct,
        tradability_score=compute_tradability_score(score, spread_pct, oi, vol),
    )


def generate_bear_put_spread(
    ticker: str,
    puts: pd.DataFrame,
    sig: Dict[str, float],
    risk_level: str,
    expiry: str,
    max_premium_budget: float,
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
) -> Optional[CandidateTrade]:
    if sig["total"] > -0.1:
        return None

    spot = sig["spot"]
    puts_liq = apply_liquidity_filter(puts, min_oi=min_oi, min_volume=min_volume, max_spread_pct=max_spread_pct)
    if puts_liq.empty:
        return None

    long_leg_pool = puts_liq[(puts_liq["strike"] >= spot * 0.97) & (puts_liq["strike"] <= spot * 1.02)].copy()
    if long_leg_pool.empty:
        return None
    long_leg = long_leg_pool.sort_values(["strike"], ascending=False).iloc[0]

    short_pool = puts_liq[puts_liq["strike"] < float(long_leg["strike"])].copy()
    short_pool = short_pool[short_pool["strike"] >= float(long_leg["strike"]) * 0.92]
    if short_pool.empty:
        return None
    short_leg = short_pool.sort_values(["strike"], ascending=False).iloc[0]

    net_debit = float(long_leg["mid"]) - float(short_leg["mid"])
    width = float(long_leg["strike"]) - float(short_leg["strike"])
    if net_debit <= 0.05 or width <= net_debit:
        return None
    if net_debit * 100 > max_premium_budget:
        return None

    oi = min(int(long_leg.get("openInterest", 0)), int(short_leg.get("openInterest", 0)))
    vol = min(int(long_leg.get("volume", 0)), int(short_leg.get("volume", 0)))
    spread_pct = max(float(long_leg.get("spread_pct", 999.0)), float(short_leg.get("spread_pct", 999.0)))
    base_score = min(0.92, 0.56 + abs(sig["total"]) * 0.35)

    return CandidateTrade(
        ticker=ticker,
        strategy="bear_put_spread",
        risk_level=risk_level,
        score=base_score,
        confidence=confidence_label(base_score),
        expiry=expiry,
        strike=float(long_leg["strike"]),
        premium=net_debit,
        spot=spot,
        rationale="Bearish defined-risk debit spread to control premium while targeting downside.",
        trade_plan=(
            f"Buy {float(long_leg['strike']):.2f} put and sell {float(short_leg['strike']):.2f} put for {expiry}."
        ),
        max_profit=f"Approx ${((width - net_debit) * 100):.2f} per contract",
        max_risk=f"Net debit paid: ${net_debit * 100:.2f}",
        breakeven=f"${float(long_leg['strike']) - net_debit:.2f}",
        option_volume=vol,
        option_oi=oi,
        spread_pct=spread_pct,
        tradability_score=compute_tradability_score(base_score, spread_pct, oi, vol),
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

    if idea.strategy in {"naked_call", "naked_put", "bull_call_spread", "bear_put_spread"}:
        per_contract_risk = idea.premium * 100
        contracts_by_risk = math.floor(risk_budget / per_contract_risk) if per_contract_risk > 0 else 0
        contracts_by_premium = math.floor(max_premium_budget / per_contract_risk) if per_contract_risk > 0 else 0
        contracts = max(0, min(contracts_by_risk, contracts_by_premium))
        if contracts < 1:
            return None

        idea.contracts = contracts
        idea.est_max_loss_dollars = round(contracts * per_contract_risk, 2)
        idea.capital_required = round(contracts * per_contract_risk, 2)
        if idea.strategy == "naked_call":
            idea.trade_plan = (
                f"Buy {contracts} call contract(s) expiring {idea.expiry}. "
                "Target 40-60% gain; cut at 50% premium loss."
            )
        elif idea.strategy == "naked_put":
            idea.trade_plan = (
                f"Buy {contracts} put contract(s) expiring {idea.expiry}. "
                "Target 40-60% gain; cut at 50% premium loss."
            )
        else:
            idea.trade_plan = f"{idea.trade_plan} Position size: {contracts} spread contract(s)."
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


def supabase_config() -> Tuple[str, str]:
    try:
        url = str(st.secrets.get("SUPABASE_URL", "")).strip().rstrip("/")
        anon = str(st.secrets.get("SUPABASE_ANON_KEY", "")).strip()
        return url, anon
    except Exception:
        return "", ""


def supabase_available() -> bool:
    url, anon = supabase_config()
    return bool(url and anon)


def supabase_request(
    endpoint: str,
    method: str = "GET",
    payload: Optional[Dict] = None,
    access_token: str = "",
    auth_api: bool = False,
    query: str = "",
) -> Tuple[Optional[Dict], Optional[str]]:
    base_url, anon_key = supabase_config()
    if not base_url or not anon_key:
        return None, "Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets."

    base_path = "/auth/v1" if auth_api else "/rest/v1"
    url = f"{base_url}{base_path}/{endpoint}"
    if query:
        url = f"{url}?{query}"

    headers = {
        "apikey": anon_key,
        "Content-Type": "application/json",
    }
    headers["Authorization"] = f"Bearer {access_token or anon_key}"
    if not auth_api:
        headers["Prefer"] = "return=representation"

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
            if not text:
                return {}, None
            parsed = json.loads(text)
            return parsed, None
    except error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
            parsed = json.loads(body)
            msg = parsed.get("msg") or parsed.get("message") or body
        except Exception:
            msg = str(exc)
        return None, f"Supabase HTTP {exc.code}: {msg}"
    except Exception as exc:
        return None, f"Supabase request failed: {type(exc).__name__}: {str(exc)}"


def supabase_sign_up(email: str, password: str) -> Optional[str]:
    _, err = supabase_request(
        endpoint="signup",
        method="POST",
        payload={"email": email, "password": password},
        auth_api=True,
    )
    return err


def supabase_sign_in(email: str, password: str) -> Tuple[Optional[Dict], Optional[str]]:
    return supabase_request(
        endpoint="token",
        method="POST",
        payload={"email": email, "password": password},
        auth_api=True,
        query="grant_type=password",
    )


def supabase_get_user(access_token: str) -> Tuple[Optional[Dict], Optional[str]]:
    return supabase_request(endpoint="user", method="GET", access_token=access_token, auth_api=True)


def supabase_get_profile(access_token: str, user_id: str) -> Tuple[Optional[Dict], Optional[str]]:
    data, err = supabase_request(
        endpoint="profiles",
        method="GET",
        access_token=access_token,
        query=f"select=*&id=eq.{user_id}&limit=1",
    )
    if err:
        return None, err
    if isinstance(data, list) and data:
        return data[0], None
    return {}, None


def supabase_upsert_profile(access_token: str, payload: Dict) -> Optional[str]:
    _, err = supabase_request(
        endpoint="profiles",
        method="POST",
        access_token=access_token,
        query="on_conflict=id",
        payload=payload,
    )
    return err


def supabase_list_watchlists(access_token: str) -> Tuple[List[Dict], Optional[str]]:
    data, err = supabase_request(
        endpoint="watchlists",
        method="GET",
        access_token=access_token,
        query="select=id,name,created_at&order=created_at.desc",
    )
    if err:
        return [], err
    return data if isinstance(data, list) else [], None


def supabase_create_watchlist(access_token: str, name: str, user_id: str) -> Optional[str]:
    _, err = supabase_request(
        endpoint="watchlists",
        method="POST",
        access_token=access_token,
        payload={"name": name, "user_id": user_id},
    )
    return err


def supabase_delete_watchlist(access_token: str, watchlist_id: str) -> Optional[str]:
    _, err = supabase_request(
        endpoint="watchlists",
        method="DELETE",
        access_token=access_token,
        query=f"id=eq.{watchlist_id}",
    )
    return err


def supabase_list_watchlist_items(access_token: str, watchlist_id: str) -> Tuple[List[Dict], Optional[str]]:
    data, err = supabase_request(
        endpoint="watchlist_items",
        method="GET",
        access_token=access_token,
        query=f"select=id,symbol,note,added_at&watchlist_id=eq.{watchlist_id}&order=added_at.desc",
    )
    if err:
        return [], err
    return data if isinstance(data, list) else [], None


def supabase_add_watchlist_item(access_token: str, watchlist_id: str, symbol: str, note: str = "") -> Optional[str]:
    _, err = supabase_request(
        endpoint="watchlist_items",
        method="POST",
        access_token=access_token,
        payload={"watchlist_id": watchlist_id, "symbol": symbol, "note": note},
    )
    return err


def supabase_remove_watchlist_item(access_token: str, watchlist_id: str, symbol: str) -> Optional[str]:
    _, err = supabase_request(
        endpoint="watchlist_items",
        method="DELETE",
        access_token=access_token,
        query=f"watchlist_id=eq.{watchlist_id}&symbol=eq.{symbol}",
    )
    return err


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
                "openInterest": int(item.get("open_interest") or 0),
                "volume": int(item.get("volume") or 0),
                "type": str(item.get("option_type", "")).lower(),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    calls = df[df["type"] == "call"][["strike", "bid", "ask", "lastPrice", "openInterest", "volume"]].copy()
    puts = df[df["type"] == "put"][["strike", "bid", "ask", "lastPrice", "openInterest", "volume"]].copy()
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
    min_oi: int,
    min_volume: int,
    max_spread_pct: float,
) -> Tuple[Optional[CandidateTrade], str]:
    tk = yf.Ticker(ticker)

    try:
        hist = fetch_history_with_retry(tk, ticker=ticker, retries=3)
    except Exception as exc:
        return None, f"History fetch failed: {type(exc).__name__}: {str(exc)[:90]}"
    if hist.empty:
        return None, "No recent price history"

    sig = compute_signal_strength(hist)
    provider_used = "yahoo"
    if options_provider == "auto":
        try:
            expiries = fetch_expiries_with_retry(tk, retries=4)
            provider_used = "yahoo"
        except Exception as exc:
            if tradier_token.strip():
                try:
                    expiries = fetch_tradier_expiries(ticker, tradier_token.strip())
                    provider_used = "tradier"
                except Exception as t_exc:
                    return None, f"Auto provider failed (Yahoo+Tradier): {type(t_exc).__name__}: {str(t_exc)[:90]}"
            else:
                if is_rate_limit_error(exc):
                    return (
                        None,
                        "Yahoo rate-limited and no Tradier token configured. Add TRADIER_TOKEN or switch provider.",
                    )
                return None, f"Expiry fetch failed: {type(exc).__name__}: {str(exc)[:90]}"
    elif options_provider == "tradier":
        if not tradier_token.strip():
            return None, "Tradier token missing. Add TRADIER_TOKEN secret or paste token in app."
        try:
            expiries = fetch_tradier_expiries(ticker, tradier_token.strip())
            provider_used = "tradier"
        except Exception as exc:
            return None, f"Tradier expiry fetch failed: {type(exc).__name__}: {str(exc)[:90]}"
    else:
        try:
            expiries = fetch_expiries_with_retry(tk, retries=4)
            provider_used = "yahoo"
        except Exception as exc:
            if is_rate_limit_error(exc):
                return (
                    None,
                    "Expiry fetch rate-limited by Yahoo Finance. Switch provider to Tradier or auto.",
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

    if provider_used == "tradier":
        try:
            calls, puts = fetch_tradier_chain(ticker, expiry, tradier_token.strip())
        except Exception as exc:
            return None, f"Tradier chain failed: {type(exc).__name__}: {str(exc)[:90]}"
    else:
        try:
            chain = fetch_options_with_retry(tk, expiry=expiry, retries=4)
        except Exception as exc:
            if options_provider == "auto" and tradier_token.strip():
                try:
                    tradier_expiries = fetch_tradier_expiries(ticker, tradier_token.strip())
                    tradier_expiry = pick_expiry(tradier_expiries, risk_level, as_of)
                    if tradier_expiry:
                        calls, puts = fetch_tradier_chain(ticker, tradier_expiry, tradier_token.strip())
                        provider_used = "tradier"
                        expiry = tradier_expiry
                    else:
                        return None, "Auto fallback failed: no Tradier expiry in valid DTE window"
                except Exception as t_exc:
                    return None, f"Auto fallback chain failed: {type(t_exc).__name__}: {str(t_exc)[:90]}"
            elif is_rate_limit_error(exc):
                return (
                    None,
                    "Option chain rate-limited by Yahoo Finance. Switch provider to Tradier or auto.",
                )
            else:
                return None, f"Option chain failed: {type(exc).__name__}: {str(exc)[:90]}"
        if provider_used == "yahoo":
            calls = chain.calls.copy()
            puts = chain.puts.copy()

    if calls.empty and puts.empty:
        return None, "Option chain data empty for selected expiry"

    candidates: List[CandidateTrade] = []

    if "naked_call" in allowed_strategies:
        idea = generate_naked_call(
            ticker,
            calls,
            sig,
            risk_level,
            expiry,
            max_premium_budget=max_premium_budget,
            min_oi=min_oi,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
        )
        if idea:
            idea.provider_used = provider_used
            candidates.append(idea)

    if "naked_put" in allowed_strategies:
        idea = generate_naked_put(
            ticker,
            puts,
            sig,
            risk_level,
            expiry,
            max_premium_budget=max_premium_budget,
            min_oi=min_oi,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
        )
        if idea:
            idea.provider_used = provider_used
            candidates.append(idea)

    if "covered_call" in allowed_strategies:
        idea = generate_covered_call(
            ticker,
            calls,
            sig,
            risk_level,
            expiry,
            shares_owned=shares_owned,
            min_oi=min_oi,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
        )
        if idea:
            idea.provider_used = provider_used
            candidates.append(idea)

    if "cash_secured_put" in allowed_strategies:
        idea = generate_cash_secured_put(
            ticker,
            puts,
            sig,
            risk_level,
            expiry,
            cash_per_trade=cash_per_trade,
            min_oi=min_oi,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
        )
        if idea:
            idea.provider_used = provider_used
            candidates.append(idea)

    if "bull_call_spread" in allowed_strategies:
        idea = generate_bull_call_spread(
            ticker,
            calls,
            sig,
            risk_level,
            expiry,
            max_premium_budget=max_premium_budget,
            min_oi=min_oi,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
        )
        if idea:
            idea.provider_used = provider_used
            candidates.append(idea)

    if "bear_put_spread" in allowed_strategies:
        idea = generate_bear_put_spread(
            ticker,
            puts,
            sig,
            risk_level,
            expiry,
            max_premium_budget=max_premium_budget,
            min_oi=min_oi,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
        )
        if idea:
            idea.provider_used = provider_used
            candidates.append(idea)

    if not candidates:
        return None, "No strategy qualified under current filters (including liquidity thresholds)"

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
                "Spread %": round(idea.spread_pct, 2),
                "Volume": int(idea.option_volume),
                "OI": int(idea.option_oi),
                "Tradability": round(idea.tradability_score, 1),
                "Capital Req ($)": round(idea.capital_required, 2),
                "Est Max Loss ($)": round(idea.est_max_loss_dollars, 2),
                "Breakeven": idea.breakeven,
                "Next Earnings": idea.earnings_date,
                "Provider": idea.provider_used,
            }
        )
    return pd.DataFrame(rows)


def render_trade_cards(ideas: List[CandidateTrade]) -> None:
    for idea in ideas:
        st.markdown(
            (
                "<div class='trade-card'>"
                f"<div class='trade-card-head'><strong>{idea.ticker}</strong> · {STRATEGY_LABELS[idea.strategy]}</div>"
                f"<div class='trade-card-row'>Confidence: {idea.confidence} | Score: {idea.score:.3f} | Tradability: {idea.tradability_score:.1f}</div>"
                f"<div class='trade-card-row'>Expiry {idea.expiry} | Strike ${idea.strike:.2f} | Mid ${idea.premium:.2f} | Contracts {idea.contracts}</div>"
                f"<div class='trade-card-row'>Spread {idea.spread_pct:.2f}% | Vol {idea.option_volume} | OI {idea.option_oi} | Provider {idea.provider_used}</div>"
                f"<div class='trade-card-row'>Capital ${idea.capital_required:,.0f} | Est Max Loss ${idea.est_max_loss_dollars:,.0f}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


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
                "Action": (
                    "BUY"
                    if idea.strategy in {"naked_call", "naked_put", "bull_call_spread", "bear_put_spread"}
                    else "SELL"
                ),
                "Contracts": idea.contracts,
                "Expiry": idea.expiry,
                "Strike": round(idea.strike, 2),
                "LimitPremium": round(idea.premium, 2),
                "EstimatedCapital": round(idea.capital_required, 2),
                "EstimatedMaxLoss": round(idea.est_max_loss_dollars, 2),
                "TradabilityScore": round(idea.tradability_score, 1),
                "Provider": idea.provider_used,
                "Breakeven": idea.breakeven,
                "Rationale": idea.rationale,
                "Plan": idea.trade_plan,
            }
        )
    return pd.DataFrame(rows)


def parse_spread_legs_from_plan(trade_plan: str) -> Tuple[Optional[float], Optional[float]]:
    nums = []
    token = ""
    for ch in trade_plan:
        if ch.isdigit() or ch == ".":
            token += ch
        else:
            if token:
                try:
                    nums.append(float(token))
                except ValueError:
                    pass
                token = ""
    if token:
        try:
            nums.append(float(token))
        except ValueError:
            pass
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None


def estimate_contract_pnl(idea: CandidateTrade, spot_move_pct: float) -> float:
    future_spot = idea.spot * (1.0 + (spot_move_pct / 100.0))
    strike = idea.strike
    premium = idea.premium
    pnl = 0.0

    if idea.strategy == "naked_call":
        pnl = (max(future_spot - strike, 0.0) - premium) * 100
    elif idea.strategy == "naked_put":
        pnl = (max(strike - future_spot, 0.0) - premium) * 100
    elif idea.strategy == "covered_call":
        stock_pnl = (future_spot - idea.spot) * 100
        short_call_payoff = premium * 100 - max(future_spot - strike, 0.0) * 100
        pnl = stock_pnl + short_call_payoff
    elif idea.strategy == "cash_secured_put":
        pnl = premium * 100 - max(strike - future_spot, 0.0) * 100
    elif idea.strategy in {"bull_call_spread", "bear_put_spread"}:
        long_leg, short_leg = parse_spread_legs_from_plan(idea.trade_plan)
        if long_leg is None or short_leg is None:
            pnl = -premium * 100
        elif idea.strategy == "bull_call_spread":
            spread_value = max(future_spot - long_leg, 0.0) - max(future_spot - short_leg, 0.0)
            pnl = (spread_value - premium) * 100
        else:
            spread_value = max(long_leg - future_spot, 0.0) - max(short_leg - future_spot, 0.0)
            pnl = (spread_value - premium) * 100
    return pnl * idea.contracts


def scenario_dataframe(ideas: List[CandidateTrade], move_min: int, move_max: int, step: int) -> pd.DataFrame:
    rows = []
    for move in range(move_min, move_max + 1, step):
        total = 0.0
        for idea in ideas:
            total += estimate_contract_pnl(idea, float(move))
        rows.append({"MovePct": move, "PortfolioPnL": round(total, 2)})
    return pd.DataFrame(rows)


def exposure_summary_df(ideas: List[CandidateTrade]) -> pd.DataFrame:
    delta_map = {
        "naked_call": 0.5,
        "naked_put": -0.5,
        "covered_call": 0.35,
        "cash_secured_put": 0.2,
        "bull_call_spread": 0.25,
        "bear_put_spread": -0.25,
    }
    theta_map = {
        "naked_call": -0.05,
        "naked_put": -0.05,
        "covered_call": 0.03,
        "cash_secured_put": 0.03,
        "bull_call_spread": -0.02,
        "bear_put_spread": -0.02,
    }

    rows = []
    for idea in ideas:
        contracts = idea.contracts
        rows.append(
            {
                "Ticker": idea.ticker,
                "Strategy": STRATEGY_LABELS[idea.strategy],
                "DeltaApprox": round(delta_map.get(idea.strategy, 0.0) * contracts * 100, 2),
                "ThetaApprox": round(theta_map.get(idea.strategy, 0.0) * contracts * 100, 2),
                "CapitalReq": round(idea.capital_required, 2),
                "EstMaxLoss": round(idea.est_max_loss_dollars, 2),
            }
        )
    return pd.DataFrame(rows)


def strategy_direction(strategy: str) -> str:
    if strategy in {"naked_call", "bull_call_spread", "covered_call", "cash_secured_put"}:
        return "bullish"
    if strategy in {"naked_put", "bear_put_spread"}:
        return "bearish"
    return "neutral"


def lifecycle_row_from_idea(idea: CandidateTrade) -> Dict[str, object]:
    direction = strategy_direction(idea.strategy)
    if direction == "bullish":
        invalidation = round(idea.spot * 0.97, 2)
    elif direction == "bearish":
        invalidation = round(idea.spot * 1.03, 2)
    else:
        invalidation = round(idea.spot, 2)
    return {
        "TicketID": f"LCM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{idea.ticker}",
        "Ticker": idea.ticker,
        "Strategy": STRATEGY_LABELS[idea.strategy],
        "StrategyKey": idea.strategy,
        "Contracts": int(idea.contracts),
        "EntryMark": round(float(idea.premium), 4),
        "CurrentMark": round(float(idea.premium), 4),
        "CloseMark": None,
        "TargetReturnPct": 40.0,
        "StopLossPct": 50.0,
        "MaxHoldDays": 7,
        "MaxSlippagePct": 5.0,
        "EventExit": True,
        "EventRiskNow": False,
        "NextEarnings": idea.earnings_date,
        "EventNote": "",
        "UnderlyingEntry": round(float(idea.spot), 2),
        "UnderlyingNow": round(float(idea.spot), 2),
        "InvalidationUnderlying": invalidation,
        "Status": "Pending",
        "ExitReason": "",
        "CreatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "OpenedAt": "",
        "ClosedAt": "",
        "RiskLevel": idea.risk_level,
    }


def lifecycle_pnl(row: Dict[str, object], mark: float) -> float:
    strategy = str(row.get("StrategyKey", ""))
    entry = float(row.get("EntryMark", 0.0) or 0.0)
    contracts = int(row.get("Contracts", 0) or 0)
    if contracts <= 0 or entry <= 0:
        return 0.0

    long_strategies = {"naked_call", "naked_put", "bull_call_spread", "bear_put_spread"}
    if strategy in long_strategies:
        per_contract = (mark - entry) * 100.0
    else:
        per_contract = (entry - mark) * 100.0
    return round(per_contract * contracts, 2)


def evaluate_lifecycle_row(row: Dict[str, object]) -> Dict[str, object]:
    updated = dict(row)
    status = str(updated.get("Status", "Pending"))
    if status != "Active":
        return updated

    strategy = str(updated.get("StrategyKey", ""))
    direction = strategy_direction(strategy)

    entry_mark = float(updated.get("EntryMark", 0.0) or 0.0)
    current_mark = float(updated.get("CurrentMark", entry_mark) or entry_mark)
    if entry_mark <= 0:
        return updated

    max_hold_days = int(updated.get("MaxHoldDays", 7) or 7)
    target_return = float(updated.get("TargetReturnPct", 40.0) or 40.0)
    stop_loss_pct = float(updated.get("StopLossPct", 50.0) or 50.0)

    opened_at = str(updated.get("OpenedAt", "")).strip()
    days_held = 0
    if opened_at:
        try:
            open_ts = datetime.strptime(opened_at, "%Y-%m-%d %H:%M:%S")
            days_held = max(0, (datetime.now() - open_ts).days)
        except ValueError:
            days_held = 0

    long_strategies = {"naked_call", "naked_put", "bull_call_spread", "bear_put_spread"}
    if strategy in long_strategies:
        return_pct = ((current_mark - entry_mark) / entry_mark) * 100.0
    else:
        return_pct = ((entry_mark - current_mark) / entry_mark) * 100.0

    trigger_reason = ""
    if bool(updated.get("EventExit", False)) and bool(updated.get("EventRiskNow", False)):
        trigger_reason = "Event exit trigger"
    elif return_pct >= target_return:
        trigger_reason = "Profit target hit"
    elif return_pct <= -abs(stop_loss_pct):
        trigger_reason = "Stop loss hit"
    elif days_held >= max_hold_days:
        trigger_reason = "Time stop hit"
    else:
        underlying_now = float(updated.get("UnderlyingNow", 0.0) or 0.0)
        invalidation = float(updated.get("InvalidationUnderlying", 0.0) or 0.0)
        if direction == "bullish" and underlying_now > 0 and underlying_now <= invalidation:
            trigger_reason = "Underlying invalidation hit"
        elif direction == "bearish" and underlying_now > 0 and underlying_now >= invalidation:
            trigger_reason = "Underlying invalidation hit"

    if trigger_reason:
        updated["Status"] = "Closed"
        updated["ExitReason"] = trigger_reason
        updated["CloseMark"] = round(current_mark, 4)
        updated["ClosedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return updated


def lifecycle_review_df(rows: List[Dict[str, object]]) -> pd.DataFrame:
    closed = [r for r in rows if str(r.get("Status", "")) == "Closed"]
    out = []
    for row in closed:
        close_mark = float(row.get("CloseMark", row.get("CurrentMark", 0.0)) or 0.0)
        pnl = lifecycle_pnl(row, close_mark)
        entry = float(row.get("EntryMark", 0.0) or 0.0)
        contracts = int(row.get("Contracts", 0) or 0)
        notional = max(1.0, entry * 100.0 * max(1, contracts))
        actual_return = round((pnl / notional) * 100.0, 2)
        target = float(row.get("TargetReturnPct", 40.0) or 40.0)
        gap = round(actual_return - target, 2)
        note = "Met/exceeded plan." if gap >= 0 else "Missed plan; tighten entry/exit discipline."
        out.append(
            {
                "TicketID": row.get("TicketID"),
                "Ticker": row.get("Ticker"),
                "Strategy": row.get("Strategy"),
                "ExitReason": row.get("ExitReason"),
                "PnL($)": pnl,
                "ActualReturn(%)": actual_return,
                "TargetReturn(%)": target,
                "PlanVsActual(%)": gap,
                "CoachNote": note,
            }
        )
    return pd.DataFrame(out)


def parse_iso_date_safe(value: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_macro_dates_csv(raw: str) -> List[date]:
    dates: List[date] = []
    for token in str(raw or "").split(","):
        dt = parse_iso_date_safe(token.strip())
        if dt:
            dates.append(dt)
    return dates


def update_event_flags(rows: List[Dict[str, object]], window_days: int, macro_dates: List[date]) -> List[Dict[str, object]]:
    today = date.today()
    horizon = today + timedelta(days=int(window_days))
    updated_rows: List[Dict[str, object]] = []

    for row in rows:
        item = dict(row)
        notes: List[str] = []
        risk_flag = False

        earnings_dt = parse_iso_date_safe(str(item.get("NextEarnings", "")))
        if earnings_dt and today <= earnings_dt <= horizon:
            risk_flag = True
            notes.append(f"Earnings {earnings_dt.isoformat()}")

        for macro_dt in macro_dates:
            if today <= macro_dt <= horizon:
                risk_flag = True
                notes.append(f"Macro {macro_dt.isoformat()}")

        item["EventRiskNow"] = bool(risk_flag)
        item["EventNote"] = "; ".join(notes)
        updated_rows.append(item)

    return updated_rows


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

    ordered_models = [model.strip() or DEFAULT_OPENAI_MODEL]
    for m in OPENAI_MODEL_FALLBACKS:
        if m not in ordered_models:
            ordered_models.append(m)

    last_exc: Optional[Exception] = None
    for m in ordered_models:
        try:
            memo = call_openai_memo(key, m, idea, risk_level, next_day)
            return memo, "openai"
        except Exception as exc:
            last_exc = exc
            continue

    _ = last_exc
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
        .trade-card {
            border: 1px solid rgba(49, 51, 63, 0.25);
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.55rem;
            background: linear-gradient(180deg, rgba(250,250,252,0.94), rgba(244,246,252,0.9));
        }
        .trade-card-head {
            font-size: 0.98rem;
            margin-bottom: 0.25rem;
        }
        .trade-card-row {
            font-size: 0.88rem;
            margin-bottom: 0.1rem;
            color: rgba(49, 51, 63, 0.95);
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
            .trade-card {
                padding: 0.65rem 0.7rem;
                border-radius: 10px;
            }
            .trade-card-row {
                font-size: 0.84rem;
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

    if "auth_access_token" not in st.session_state:
        st.session_state.auth_access_token = ""
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = {}
    if "watchlists_cache" not in st.session_state:
        st.session_state.watchlists_cache = []
    if "watchlist_items_cache" not in st.session_state:
        st.session_state.watchlist_items_cache = []

    if not supabase_available():
        st.error("Supabase is required. Add SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets to continue.")
        return

    is_logged_in = bool(st.session_state.auth_access_token)
    if is_logged_in and not st.session_state.auth_user:
        user, uerr = supabase_get_user(st.session_state.auth_access_token)
        if uerr:
            st.warning(uerr)
        elif user:
            st.session_state.auth_user = user

    if not is_logged_in:
        st.subheader("Sign In")
        st.caption("Log in to access your profile, watchlists, and trade workspace.")
        sign_in_tab, sign_up_tab = st.tabs(["Log In", "Create Account"])

        with sign_in_tab:
            a1, a2 = st.columns(2)
            with a1:
                auth_email = st.text_input("Email", key="auth_email")
            with a2:
                auth_password = st.text_input("Password", type="password", key="auth_password")
            if st.button("Log In", key="login_btn"):
                data, err = supabase_sign_in(auth_email.strip(), auth_password)
                if err:
                    st.error(err)
                else:
                    token = str(data.get("access_token", ""))
                    st.session_state.auth_access_token = token
                    user, uerr = supabase_get_user(token)
                    if uerr:
                        st.error(uerr)
                    else:
                        st.session_state.auth_user = user or {}
                        st.success("Logged in.")
                        st.rerun()

        with sign_up_tab:
            s1, s2 = st.columns(2)
            with s1:
                signup_email = st.text_input("Email", key="signup_email")
            with s2:
                signup_password = st.text_input("Password", type="password", key="signup_password")
            if st.button("Create Account", key="signup_btn"):
                err = supabase_sign_up(signup_email.strip(), signup_password)
                if err:
                    st.error(err)
                else:
                    st.success("Sign-up submitted. Verify email if required, then log in.")
        return

    st.subheader("Account & Watchlists")
    user_email = st.session_state.auth_user.get("email", "Unknown")
    user_id = st.session_state.auth_user.get("id", "")
    st.caption(f"Logged in as: {user_email}")
    if st.button("Log Out", key="logout_btn"):
        st.session_state.auth_access_token = ""
        st.session_state.auth_user = {}
        st.session_state.watchlists_cache = []
        st.session_state.watchlist_items_cache = []
        st.rerun()

    profile_tab, watch_tab = st.tabs(["Profile", "Watchlists"])

    with profile_tab:
        profile_data, perr = supabase_get_profile(st.session_state.auth_access_token, user_id)
        if perr:
            st.warning(perr)
        p1, p2, p3 = st.columns(3)
        with p1:
            full_name = st.text_input("Full Name", value=str(profile_data.get("full_name", "")), key="profile_full_name")
        with p2:
            experience = st.selectbox(
                "Options Experience",
                ["beginner", "intermediate", "advanced"],
                index=["beginner", "intermediate", "advanced"].index(str(profile_data.get("experience_level", "beginner")))
                if str(profile_data.get("experience_level", "beginner")) in ["beginner", "intermediate", "advanced"]
                else 0,
                key="profile_experience",
            )
        with p3:
            default_risk_pref = st.selectbox(
                "Default Risk Preference",
                ["conservative", "moderate"],
                index=0 if str(profile_data.get("default_risk", "conservative")) == "conservative" else 1,
                key="profile_risk_pref",
            )
        if st.button("Save Profile", key="save_profile_btn"):
            err = supabase_upsert_profile(
                st.session_state.auth_access_token,
                {
                    "id": user_id,
                    "email": user_email,
                    "full_name": full_name.strip(),
                    "experience_level": experience,
                    "default_risk": default_risk_pref,
                    "updated_at": datetime.now().isoformat(),
                },
            )
            if err:
                st.error(err)
            else:
                st.success("Profile saved")

    with watch_tab:
        token = st.session_state.auth_access_token
        if st.button("Refresh Watchlists", key="refresh_watchlists_btn"):
            watchlists, err = supabase_list_watchlists(token)
            if err:
                st.error(err)
            else:
                st.session_state.watchlists_cache = watchlists
        if not st.session_state.watchlists_cache:
            watchlists, err = supabase_list_watchlists(token)
            if not err:
                st.session_state.watchlists_cache = watchlists

        w1, w2 = st.columns(2)
        with w1:
            new_watchlist_name = st.text_input("New Watchlist Name", key="new_watchlist_name")
        with w2:
            st.write("")
            st.write("")
            if st.button("Create Watchlist", key="create_watchlist_btn"):
                err = supabase_create_watchlist(token, new_watchlist_name.strip(), st.session_state.auth_user.get("id", ""))
                if err:
                    st.error(err)
                else:
                    st.success("Watchlist created")
                    st.session_state.watchlists_cache, _ = supabase_list_watchlists(token)
                    st.rerun()

        wl_options = st.session_state.watchlists_cache
        wl_labels = [f"{w.get('name')} ({w.get('id')})" for w in wl_options]
        selected_label = st.selectbox(
            "Select Watchlist",
            wl_labels if wl_labels else [""],
            index=0,
            key="selected_watchlist_label",
        )
        selected_watchlist_id = ""
        if selected_label:
            selected_watchlist_id = selected_label.split("(")[-1].replace(")", "").strip()

        if selected_watchlist_id:
            items, err = supabase_list_watchlist_items(token, selected_watchlist_id)
            if err:
                st.error(err)
            else:
                st.session_state.watchlist_items_cache = items

            item_symbols = [str(x.get("symbol", "")).upper() for x in st.session_state.watchlist_items_cache]
            st.caption("Symbols: " + (", ".join(item_symbols) if item_symbols else "No symbols yet"))

            i1, i2 = st.columns(2)
            with i1:
                add_symbol = st.text_input("Add Symbol", key="add_watchlist_symbol")
            with i2:
                add_note = st.text_input("Symbol Note (optional)", key="add_watchlist_note")
            if st.button("Add Symbol To Watchlist", key="add_symbol_btn"):
                symbol = add_symbol.strip().upper()
                if not symbol:
                    st.warning("Enter a symbol")
                else:
                    err = supabase_add_watchlist_item(token, selected_watchlist_id, symbol, add_note.strip())
                    if err:
                        st.error(err)
                    else:
                        st.success(f"Added {symbol}")
                        st.session_state.watchlist_items_cache, _ = supabase_list_watchlist_items(token, selected_watchlist_id)
                        st.rerun()

            remove_symbol = st.selectbox(
                "Remove Symbol",
                item_symbols if item_symbols else [""],
                key="remove_watchlist_symbol",
            )
            if st.button("Remove Symbol From Watchlist", key="remove_symbol_btn") and remove_symbol:
                err = supabase_remove_watchlist_item(token, selected_watchlist_id, remove_symbol)
                if err:
                    st.error(err)
                else:
                    st.success(f"Removed {remove_symbol}")
                    st.session_state.watchlist_items_cache, _ = supabase_list_watchlist_items(token, selected_watchlist_id)
                    st.rerun()

            if st.button("Delete Watchlist", key="delete_watchlist_btn"):
                err = supabase_delete_watchlist(token, selected_watchlist_id)
                if err:
                    st.error(err)
                else:
                    st.success("Watchlist deleted")
                    st.session_state.watchlists_cache, _ = supabase_list_watchlists(token)
                    st.session_state.watchlist_items_cache = []
                    st.rerun()

    st.divider()
    st.subheader("Trade Setup")
    st.caption("All filters are visible here for better mobile usability.")
    setup_mode = st.radio("Mode", ["Basic", "Advanced"], horizontal=True, index=0, key="setup_mode")

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

    watchlist_symbols = [str(x.get("symbol", "")).upper() for x in st.session_state.get("watchlist_items_cache", [])]
    use_watchlist_for_analysis = st.checkbox(
        "Use selected watchlist symbols for analysis",
        value=bool(watchlist_symbols),
        key="use_watchlist_for_analysis",
    )

    if use_watchlist_for_analysis and watchlist_symbols:
        tickers = sorted(set([s for s in watchlist_symbols if s]))
        st.multiselect(
            "Watchlist Symbols (Active)",
            tickers,
            default=tickers,
            disabled=True,
            key="watchlist_symbols_preview",
        )
    else:
        manual_universe = sorted(set(BLUE_CHIP_TECH + watchlist_symbols))
        tickers = st.multiselect(
            "Blue-Chip / Manual Universe",
            manual_universe,
            default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
        )

    if setup_mode == "Advanced":
        allowed_strategy_labels = st.multiselect(
            "Allowed Strategies",
            list(STRATEGY_LABELS.values()),
            default=list(STRATEGY_LABELS.values()),
        )
    else:
        allowed_strategy_labels = list(STRATEGY_LABELS.values())
        st.caption("Basic mode uses all strategies by default.")

    strategy_reverse = {v: k for k, v in STRATEGY_LABELS.items()}
    allowed_strategies = [strategy_reverse[label] for label in allowed_strategy_labels]

    if setup_mode == "Advanced":
        st.markdown("Shares owned per ticker (for covered calls)")
        shares_default = pd.DataFrame({"Ticker": tickers, "SharesOwned": [0] * len(tickers)})
        shares_df = st.data_editor(
            shares_default,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
        )
        shares_map = {row["Ticker"]: int(row["SharesOwned"]) for _, row in shares_df.iterrows()}
    else:
        shares_map = {ticker: 0 for ticker in tickers}

    default_risk = 1.0 if risk_level == "conservative" else 2.0
    risk_size_col, csp_col, call_col = st.columns(3)
    with risk_size_col:
        if setup_mode == "Advanced":
            max_risk_per_trade_pct = st.slider("Max Risk Per Trade (%)", 0.25, 5.0, float(default_risk), 0.25)
        else:
            max_risk_per_trade_pct = st.selectbox("Risk Budget (%)", [0.5, 1.0, 1.5, 2.0], index=1 if risk_level == "conservative" else 3)
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
            "Max premium budget for long options/spreads ($)",
            min_value=100.0,
            max_value=25000.0,
            value=750.0,
            step=50.0,
        )

    if setup_mode == "Advanced":
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
                ["auto", "yahoo", "tradier"],
                index=0,
                help="Auto tries Yahoo first, then falls back to Tradier if available.",
            )
        with provider_col2:
            tradier_token_input = st.text_input(
                "Tradier Token (optional)",
                type="password",
                help="Used when Provider is set to tradier. You can also store TRADIER_TOKEN in Streamlit secrets.",
            )
    else:
        use_earnings_filter = True
        avoid_if_before_expiry = True
        earnings_buffer_days = 7
        options_provider = "auto"
        tradier_token_input = ""
        st.caption("Basic defaults: earnings filter ON, provider AUTO, standard guardrails.")

    tradier_token = tradier_token_input.strip()
    if not tradier_token:
        try:
            tradier_token = str(st.secrets.get("TRADIER_TOKEN", "")).strip()
        except Exception:
            tradier_token = ""

    if setup_mode == "Advanced":
        st.markdown("Liquidity Filters")
        liq_col1, liq_col2, liq_col3 = st.columns(3)
        with liq_col1:
            min_oi = st.number_input("Min Open Interest", min_value=0, max_value=50000, value=50, step=10)
        with liq_col2:
            min_volume = st.number_input("Min Option Volume", min_value=0, max_value=50000, value=10, step=5)
        with liq_col3:
            max_spread_pct = st.slider("Max Bid-Ask Spread %", min_value=1.0, max_value=35.0, value=15.0, step=0.5)
    else:
        min_oi = 50
        min_volume = 10
        max_spread_pct = 15.0

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
    if "chat_last_source" not in st.session_state:
        st.session_state.chat_last_source = "offline"
    if "chat_last_error" not in st.session_state:
        st.session_state.chat_last_error = ""
    if "journal_rows" not in st.session_state:
        st.session_state.journal_rows = []
    if "lifecycle_rows" not in st.session_state:
        st.session_state.lifecycle_rows = []

    def run_analysis() -> bool:
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
                    min_oi=int(min_oi),
                    min_volume=int(min_volume),
                    max_spread_pct=float(max_spread_pct),
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
        return True

    analyze_clicked = False
    if setup_mode == "Basic":
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            if st.button("Quick Run", type="primary", key="quick_run_btn"):
                analyze_clicked = True
        with a_col2:
            if st.button("Analyze and Generate Next-Day Trades", key="analyze_btn_basic"):
                analyze_clicked = True
        st.caption("Quick Run uses Basic defaults and launches analysis in one tap.")
    else:
        if st.button("Analyze and Generate Next-Day Trades", type="primary", key="analyze_btn_advanced"):
            analyze_clicked = True

    if analyze_clicked:
        run_analysis()

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
    view_mode = st.radio("View", ["Cards", "Table"], horizontal=True, index=0, key="trade_view_mode")
    if view_mode == "Cards":
        render_trade_cards(ideas)
    else:
        st.dataframe(ideas_df, use_container_width=True, height=320)

    st.subheader("Portfolio Exposure")
    exp_df = exposure_summary_df(ideas)
    st.dataframe(exp_df, use_container_width=True, height=220)
    c1, c2, c3 = st.columns(3)
    c1.metric("Net Delta (Approx)", f"{exp_df['DeltaApprox'].sum():.1f}")
    c2.metric("Net Theta (Approx)", f"{exp_df['ThetaApprox'].sum():.1f}")
    c3.metric("Total Est Max Loss", f"${exp_df['EstMaxLoss'].sum():,.0f}")

    st.subheader("P/L Scenario Simulator")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        move_min = st.slider("Min Underlying Move %", -30, -1, -10, 1, key="sim_min")
    with sim_col2:
        move_max = st.slider("Max Underlying Move %", 1, 30, 10, 1, key="sim_max")
    with sim_col3:
        move_step = st.selectbox("Step %", [1, 2, 5], index=1, key="sim_step")
    sim_df = scenario_dataframe(ideas, move_min=move_min, move_max=move_max, step=int(move_step))
    st.line_chart(sim_df.set_index("MovePct"))
    st.dataframe(sim_df, use_container_width=True, height=220)

    for idea in ideas:
        with st.expander(f"{idea.ticker}: {STRATEGY_LABELS[idea.strategy]} ({idea.confidence} confidence)"):
            st.write(f"Rationale: {idea.rationale}")
            st.write(f"Plan: {idea.trade_plan}")
            st.write(f"Contracts: {idea.contracts}")
            st.write(f"Estimated Capital Required: ${idea.capital_required:,.2f}")
            st.write(f"Estimated Max Loss: ${idea.est_max_loss_dollars:,.2f}")
            st.write(
                f"Tradability: {idea.tradability_score:.1f} | Spread: {idea.spread_pct:.2f}% | Volume: {idea.option_volume} | OI: {idea.option_oi}"
            )
            st.write(f"Max Profit: {idea.max_profit}")
            st.write(f"Max Risk: {idea.max_risk}")
            st.write(f"Breakeven: {idea.breakeven}")
            st.write(f"Next Earnings: {idea.earnings_date}")
            st.write(f"Provider: {idea.provider_used}")
            if st.button("Add To Journal", key=f"journal_add_{idea.ticker}_{idea.strategy}_{idea.expiry}"):
                st.session_state.journal_rows.append(
                    {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Ticker": idea.ticker,
                        "Strategy": STRATEGY_LABELS[idea.strategy],
                        "Contracts": idea.contracts,
                        "Expiry": idea.expiry,
                        "Strike": round(idea.strike, 2),
                        "Premium": round(idea.premium, 2),
                        "RiskLevel": result_risk_level,
                        "Plan": idea.trade_plan,
                    }
                )
                st.success("Added to journal")
            if st.button("Create Lifecycle Plan", key=f"lifecycle_add_{idea.ticker}_{idea.strategy}_{idea.expiry}"):
                new_row = lifecycle_row_from_idea(idea)
                existing_ids = {str(r.get("TicketID", "")) for r in st.session_state.lifecycle_rows}
                if new_row["TicketID"] in existing_ids:
                    new_row["TicketID"] = f"{new_row['TicketID']}-X"
                st.session_state.lifecycle_rows.append(new_row)
                st.success("Lifecycle plan created")

    st.subheader("AI Memo")
    memo_col1, memo_col2, memo_col3 = st.columns([2, 2, 1])
    with memo_col1:
        selected_ticker = st.selectbox(
            "Memo For Ticker",
            [idea.ticker for idea in ideas],
            key="memo_ticker",
        )
    with memo_col2:
        memo_model = st.text_input("AI Model", value=DEFAULT_OPENAI_MODEL, key="memo_model")
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
                model=memo_model.strip() or DEFAULT_OPENAI_MODEL,
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
        chat_model = st.text_input("Chat Model", value=DEFAULT_OPENAI_MODEL, key="chat_model")
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

    manual_key_present = bool(chat_api_key_input.strip())
    secret_key_present = False
    try:
        secret_key_present = bool(str(st.secrets.get("OPENAI_API_KEY", "")).strip())
    except Exception:
        secret_key_present = False

    key_source = "manual input" if manual_key_present else ("Streamlit secret" if secret_key_present else "none")
    st.caption(f"Chat key source: {key_source} | Last response source: {st.session_state.chat_last_source}")
    if st.session_state.chat_last_error:
        st.caption(f"Last OpenAI error: {st.session_state.chat_last_error}")

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
            requested_model = chat_model.strip() or DEFAULT_OPENAI_MODEL
            ordered_models = [requested_model]
            for m in OPENAI_MODEL_FALLBACKS:
                if m not in ordered_models:
                    ordered_models.append(m)

            bot_reply = ""
            last_exc: Optional[Exception] = None
            for candidate_model in ordered_models:
                try:
                    bot_reply = call_openai_chat(
                        api_key=key,
                        model=candidate_model,
                        user_text=user_chat,
                        risk_level=result_risk_level,
                        context_tickers=context_tickers,
                        chat_history=st.session_state.chat_history,
                        persona=chat_persona,
                    )
                    st.session_state.chat_last_source = f"openai:{candidate_model}"
                    st.session_state.chat_last_error = ""
                    break
                except Exception as exc:
                    last_exc = exc
                    continue

            if not bot_reply:
                bot_reply = fallback_chat_reply(user_chat, context_tickers, result_risk_level, chat_persona)
                st.session_state.chat_last_source = "offline-fallback"
                st.session_state.chat_last_error = (
                    f"{type(last_exc).__name__}: {str(last_exc)[:180]}"
                    if last_exc
                    else "No OpenAI response from attempted models"
                )
        else:
            bot_reply = fallback_chat_reply(user_chat, context_tickers, result_risk_level, chat_persona)
            st.session_state.chat_last_source = "offline-no-key"
            st.session_state.chat_last_error = ""

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

    st.subheader("Trade Journal")
    journal_df = pd.DataFrame(st.session_state.journal_rows)
    if journal_df.empty:
        st.caption("No journal entries yet. Use 'Add To Journal' on a trade card.")
    else:
        st.dataframe(journal_df, use_container_width=True, height=240)
        st.download_button(
            "Download Journal (CSV)",
            data=journal_df.to_csv(index=False).encode("utf-8"),
            file_name=f"trade_journal_{result_next_day}.csv",
            mime="text/csv",
        )
        if st.button("Clear Journal", key="clear_journal_btn"):
            st.session_state.journal_rows = []
            st.rerun()

    st.subheader("Trade Lifecycle Manager")
    if not st.session_state.lifecycle_rows:
        st.caption("No lifecycle plans yet. Click 'Create Lifecycle Plan' on any generated trade.")
    else:
        lifecycle_df = pd.DataFrame(st.session_state.lifecycle_rows)
        editable_cols = [
            "TicketID",
            "Ticker",
            "Strategy",
            "Contracts",
            "EntryMark",
            "CurrentMark",
            "TargetReturnPct",
            "StopLossPct",
            "MaxHoldDays",
            "MaxSlippagePct",
            "EventExit",
            "EventRiskNow",
            "NextEarnings",
            "EventNote",
            "UnderlyingEntry",
            "UnderlyingNow",
            "InvalidationUnderlying",
            "Status",
            "ExitReason",
            "OpenedAt",
            "ClosedAt",
        ]
        lifecycle_view = lifecycle_df[editable_cols].copy()
        lifecycle_view = st.data_editor(
            lifecycle_view,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="lifecycle_editor",
        )

        base_cols = [col for col in lifecycle_df.columns if col not in lifecycle_view.columns]
        merged = lifecycle_view.copy()
        for col in base_cols:
            merged[col] = lifecycle_df[col].values
        st.session_state.lifecycle_rows = merged.to_dict(orient="records")

        event_col1, event_col2, event_col3 = st.columns(3)
        with event_col1:
            lifecycle_event_window = st.slider(
                "Lifecycle Event Window (days)",
                min_value=1,
                max_value=21,
                value=5,
                key="lifecycle_event_window_days",
            )
        with event_col2:
            macro_dates_raw = st.text_input(
                "Macro Dates CSV (YYYY-MM-DD)",
                value="",
                key="lifecycle_macro_dates",
                help="Example: 2026-03-18, 2026-05-06",
            )
        with event_col3:
            st.write("")
            st.write("")
            if st.button("Auto Update Event Flags", key="auto_update_event_flags"):
                macro_dates = parse_macro_dates_csv(macro_dates_raw)
                st.session_state.lifecycle_rows = update_event_flags(
                    st.session_state.lifecycle_rows,
                    window_days=int(lifecycle_event_window),
                    macro_dates=macro_dates,
                )
                st.rerun()

        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            if st.button("Activate Pending Trades", key="activate_pending_lifecycle"):
                updated = []
                for row in st.session_state.lifecycle_rows:
                    item = dict(row)
                    if str(item.get("Status", "")) == "Pending":
                        entry = float(item.get("EntryMark", 0.0) or 0.0)
                        current = float(item.get("CurrentMark", entry) or entry)
                        slip = float(item.get("MaxSlippagePct", 5.0) or 5.0)
                        slippage_pct = abs((current - entry) / entry) * 100.0 if entry > 0 else 0.0
                        if slippage_pct <= slip:
                            item["Status"] = "Active"
                            item["OpenedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            item["ExitReason"] = ""
                        else:
                            item["ExitReason"] = "Activation blocked: slippage above max"
                    updated.append(item)
                st.session_state.lifecycle_rows = updated
                st.rerun()
        with lcol2:
            if st.button("Run Rule Check", key="run_rule_check"):
                st.session_state.lifecycle_rows = [evaluate_lifecycle_row(r) for r in st.session_state.lifecycle_rows]
                st.rerun()
        with lcol3:
            ticket_choices = [str(r.get("TicketID")) for r in st.session_state.lifecycle_rows]
            close_ticket = st.selectbox("Manual Close Ticket", ticket_choices, key="manual_close_ticket")
            if st.button("Close Selected", key="manual_close_btn"):
                updated = []
                for row in st.session_state.lifecycle_rows:
                    item = dict(row)
                    if str(item.get("TicketID")) == close_ticket and str(item.get("Status")) in {"Active", "Pending"}:
                        item["Status"] = "Closed"
                        item["ExitReason"] = "Manual close"
                        item["CloseMark"] = float(item.get("CurrentMark", item.get("EntryMark", 0.0)) or 0.0)
                        item["ClosedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    updated.append(item)
                st.session_state.lifecycle_rows = updated
                st.rerun()

        review_df = lifecycle_review_df(st.session_state.lifecycle_rows)
        st.markdown("Post-Trade Review")
        if review_df.empty:
            st.caption("No closed trades yet for review.")
        else:
            st.dataframe(review_df, use_container_width=True, height=220)
            st.download_button(
                "Download Lifecycle Review (CSV)",
                data=review_df.to_csv(index=False).encode("utf-8"),
                file_name=f"lifecycle_review_{result_next_day}.csv",
                mime="text/csv",
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
