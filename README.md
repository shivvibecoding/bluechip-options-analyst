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
- Calendar-driven event flagging in lifecycle manager:
  - Auto-update `EventRiskNow` from upcoming earnings dates
  - Optional macro date CSV input (e.g., FOMC/CPI dates)
  - Configurable event window days

## Supabase setup (signup, profile, watchlists)

This app now supports:
- user signup/login
- profile storage
- watchlist CRUD and analysis from selected watchlist symbols

### 1) Create free Supabase project

- Go to Supabase and create a new project (free tier is enough for POC).
- In project settings, copy:
  - Project URL
  - anon/public API key

### 2) Run SQL (Supabase SQL editor)

```sql
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text default '',
  experience_level text default 'beginner',
  default_risk text default 'conservative',
  updated_at timestamptz default now()
);

create table if not exists public.watchlists (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  created_at timestamptz default now()
);

create table if not exists public.watchlist_items (
  id uuid primary key default gen_random_uuid(),
  watchlist_id uuid not null references public.watchlists(id) on delete cascade,
  symbol text not null,
  note text default '',
  added_at timestamptz default now()
);

create unique index if not exists watchlist_items_unique_symbol
on public.watchlist_items (watchlist_id, symbol);

alter table public.profiles enable row level security;
alter table public.watchlists enable row level security;
alter table public.watchlist_items enable row level security;

-- profiles policies
drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own"
on public.profiles for select
using (auth.uid() = id);

drop policy if exists "profiles_insert_own" on public.profiles;
create policy "profiles_insert_own"
on public.profiles for insert
with check (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own"
on public.profiles for update
using (auth.uid() = id);

-- watchlists policies
drop policy if exists "watchlists_select_own" on public.watchlists;
create policy "watchlists_select_own"
on public.watchlists for select
using (auth.uid() = user_id);

drop policy if exists "watchlists_insert_own" on public.watchlists;
create policy "watchlists_insert_own"
on public.watchlists for insert
with check (auth.uid() = user_id);

drop policy if exists "watchlists_delete_own" on public.watchlists;
create policy "watchlists_delete_own"
on public.watchlists for delete
using (auth.uid() = user_id);

-- watchlist_items policies
drop policy if exists "watchlist_items_select_own" on public.watchlist_items;
create policy "watchlist_items_select_own"
on public.watchlist_items for select
using (
  exists (
    select 1 from public.watchlists w
    where w.id = watchlist_items.watchlist_id
      and w.user_id = auth.uid()
  )
);

drop policy if exists "watchlist_items_insert_own" on public.watchlist_items;
create policy "watchlist_items_insert_own"
on public.watchlist_items for insert
with check (
  exists (
    select 1 from public.watchlists w
    where w.id = watchlist_items.watchlist_id
      and w.user_id = auth.uid()
  )
);

drop policy if exists "watchlist_items_delete_own" on public.watchlist_items;
create policy "watchlist_items_delete_own"
on public.watchlist_items for delete
using (
  exists (
    select 1 from public.watchlists w
    where w.id = watchlist_items.watchlist_id
      and w.user_id = auth.uid()
  )
);
```

### 3) Add Streamlit secrets

In Streamlit Cloud -> Manage app -> Secrets:

```toml
SUPABASE_URL="https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_ANON_KEY="YOUR_SUPABASE_ANON_KEY"
OPENAI_API_KEY="sk-..."
TRADIER_TOKEN="..."
```

### 4) Auth settings

- In Supabase Auth settings, enable Email/Password provider.
- If email confirmation is ON, users must verify email before first login.

### 5) In-app usage

- Open `Account & Watchlists` section.
- Sign up / log in.
- Save profile fields.
- Create watchlist, add symbols, then enable:
  - `Use selected watchlist symbols for analysis`

## Setup modes

- `Basic` mode: streamlined setup with safe defaults (all strategies enabled, earnings filter on, provider auto, standard liquidity guardrails).
- `Advanced` mode: full control over strategy selection, provider, earnings rules, liquidity thresholds, and sizing knobs.
