# Blue-Chip Tech Options Analyst

Analyst-style app for generating **next-trading-day** options ideas on blue-chip tech stocks.

## Core capabilities

- Scans a blue-chip tech universe (`AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`, `META`, `AVGO`, `ORCL`, `ADBE`, `CRM`)
- Builds trend/momentum/volatility signals from 6 months of daily price data
- Pulls options chains and filters ideas to only:
  - Buy Naked Call
  - Buy Naked Put
  - Sell Covered Call
  - Sell Cash-Secured Put
  - Buy Bull Call Spread
  - Buy Bear Put Spread
- Applies risk profile (`conservative` or `moderate`) and account constraints
- Returns one top trade idea per ticker with rationale, breakeven, and risk notes

## Phase 2 features added

- Portfolio-aware position sizing:
  - Inputs: portfolio size and max risk-per-trade %
  - Output: recommended contracts, estimated capital required, estimated max loss
- Earnings-date filter:
  - Optional switch to avoid trades near earnings
  - Adjustable event window (days)
  - Optional strict rule to avoid trades if earnings lands before option expiry
- Trade ticket export:
  - CSV download for systematic execution workflows
  - PDF ticket report for review or sharing

## UX + trader feature upgrades

- Liquidity filters:
  - Minimum Open Interest
  - Minimum Option Volume
  - Maximum bid-ask spread %
- Tradability Score on every trade idea (combines spread, OI, volume, and setup quality)
- Mobile-friendly trade card view (default) with optional table view
- Auto provider failover:
  - `auto` mode tries Yahoo first
  - falls back to Tradier when Yahoo is rate-limited/unavailable

## AI memo feature

- Generate an analyst-style "Trade Committee Memo" per trade idea with:
  - Executive summary
  - Key risks
  - Pre-market and intraday checklists
  - Beginner-friendly explanation
  - Desk-style analyst note
- Works in two modes:
  - OpenAI mode (enter API key in app or set `OPENAI_API_KEY` in Streamlit secrets)
  - Offline fallback mode (built-in memo template)

## AI chatbot feature

- Interactive coach chatbot embedded in app results section
- Understands current analyzed tickers and risk mode context
- Works in two modes:
  - OpenAI chat mode (API key in app field or `OPENAI_API_KEY` in secrets)
  - Offline coaching fallback if no key/API error
- Chat transcript can be downloaded as `.txt`

## Run

```bash
cd /Users/shivali/bluechip-options-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Inputs you control

- Risk level: `conservative` or `moderate`
- Allowed strategy types
- Shares owned per ticker (required for covered calls)
- Portfolio size and max risk-per-trade %
- Cash reserved per CSP trade
- Max premium budget for naked call
- Earnings filter behavior and window size
- Options data provider: `yahoo` or `tradier`

## Handling Yahoo rate limits

If Streamlit Cloud keeps showing Yahoo rate-limit errors for expiries/chains, switch provider to `tradier`.

Set in Streamlit secrets:

```toml
TRADIER_TOKEN="your_tradier_token"
OPENAI_API_KEY="sk-..."
```

## Important

This tool is for educational support, not investment advice.
Options trading carries significant risk, including total loss of premium (long calls) or assignment risk (short options).

## Added trader productivity features

- Portfolio exposure dashboard (approx net delta/theta and aggregate risk)
- P/L scenario simulator (portfolio-level payout across underlying move ranges)
- Trade journal workflow:
  - Add generated ideas to journal
  - Persist during session
  - Export journal CSV

## Trade lifecycle manager

- Create lifecycle plan from any generated trade idea
- Activate pending trades with slippage check
- Rule-based management triggers:
  - profit target
  - stop loss
  - time stop
  - event risk exit
  - underlying invalidation
- Manual close by ticket ID
- Post-trade review table with plan-vs-actual and coach note
- Lifecycle review CSV export
