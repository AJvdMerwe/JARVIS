"""
tools/financial_tools.py
─────────────────────────
LangChain tools for the Financial Agent, powered by yfinance.

Available tools:
  • StockQuoteTool          – Current price, daily change, volume, market cap.
  • CompanyInfoTool         – Business description, sector, employees, exchange.
  • FinancialStatementsTool – Income statement, balance sheet, cash flow
                              (annual or quarterly).
  • FinancialRatiosTool     – Valuation, profitability, liquidity and
                              leverage ratios with plain-English grades.
  • PriceHistoryTool        – OHLCV history for any period/interval with
                              derived statistics (return, volatility, CAGR).
  • StockComparisonTool     – Side-by-side comparison of up to 5 tickers
                              across key metrics.

All tools:
  • Handle missing data gracefully — yfinance returns None / NaN frequently.
  • Format numbers to 2 d.p. with appropriate units (B, M, K, %).
  • Cache-friendly: pure functions of the ticker symbol and parameters.
  • Work with the @cached_tool decorator in FinancialAgent.

Requires:
    pip install yfinance

Data comes from Yahoo Finance (free, no API key). Rate limits apply for
heavy use — the FinancialAgent applies TTL caching to stay within them.
"""
from __future__ import annotations

import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
#  Formatting helpers
# =============================================================================

def _fmt_number(value, prefix: str = "", suffix: str = "") -> str:
    """Format a numeric value with K/M/B scaling, or return 'N/A'."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if abs(v) >= 1e12:
            return f"{prefix}{v / 1e12:.2f}T{suffix}"
        if abs(v) >= 1e9:
            return f"{prefix}{v / 1e9:.2f}B{suffix}"
        if abs(v) >= 1e6:
            return f"{prefix}{v / 1e6:.2f}M{suffix}"
        if abs(v) >= 1e3:
            return f"{prefix}{v / 1e3:.2f}K{suffix}"
        return f"{prefix}{v:.2f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(value) -> str:
    """Format a ratio (e.g. 0.0523) as a percentage string."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_ratio(value, decimals: int = 2) -> str:
    """Format a ratio / multiple with N/A fallback."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}x"
    except (TypeError, ValueError):
        return "N/A"


def _safe(info: dict, key: str, default=None):
    """Safely get a key from yfinance info dict, returning default on absence."""
    v = info.get(key, default)
    return default if v in (None, "N/A", float("inf"), float("-inf")) else v


def _grade_ratio(metric: str, value: float | None) -> str:
    """Return a simple qualitative grade for common financial ratios."""
    if value is None:
        return ""
    thresholds: dict[str, tuple] = {
        "pe":        (10, 20, 35),   # low, fair, expensive
        "pb":        (1, 3, 5),
        "roe":       (0.15, 0.20, 0.30),   # higher is better; thresholds reversed
        "roa":       (0.05, 0.10, 0.20),
        "current":   (1.5, 2.5, 4.0),     # liquidity
        "de":        (0.5, 1.5, 3.0),     # debt/equity; lower is better
    }
    t = thresholds.get(metric)
    if not t:
        return ""
    lo, mid, hi = t
    if metric in ("roe", "roa", "current"):      # higher = better
        if value >= hi:   return "  ★ Excellent"
        if value >= mid:  return "  ✓ Good"
        if value >= lo:   return "  ~ Fair"
        return "  ✗ Weak"
    else:                                         # lower = better
        if value <= lo:   return "  ★ Attractive"
        if value <= mid:  return "  ✓ Fair"
        if value <= hi:   return "  ~ Expensive"
        return "  ✗ Very expensive"


# =============================================================================
#  Input schemas
# =============================================================================

class TickerInput(BaseModel):
    ticker: str = Field(
        ...,
        description="Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'BTC-USD'",
    )


class TickerPeriodInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    period: str = Field(
        default="1y",
        description=(
            "Period of history: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"
        ),
    )
    interval: str = Field(
        default="1d",
        description="Data interval: 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo",
    )


class StatementInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    statement: str = Field(
        default="income",
        description="Statement type: 'income', 'balance', or 'cashflow'",
    )
    quarterly: bool = Field(
        default=False,
        description="If True, return quarterly data; otherwise annual",
    )


class CompareInput(BaseModel):
    tickers: str = Field(
        ...,
        description="Comma-separated ticker symbols, e.g. 'AAPL,MSFT,GOOGL'",
    )


# =============================================================================
#  Tool 1 — Stock Quote
# =============================================================================

class StockQuoteTool(BaseTool):
    """
    Fetch the current price and key market data for a stock.

    Returns: current price, daily change/%, bid/ask, volume, market cap,
    52-week range, beta, and trading day summary.
    """

    name: str = "stock_quote"
    description: str = (
        "Get the current stock price and key market data for a ticker symbol. "
        "Returns price, change, volume, market cap, and 52-week range. "
        "Use for: 'What is AAPL trading at?', 'Current price of Tesla'."
    )
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf
            t    = yf.Ticker(ticker)
            info = t.fast_info

            price       = getattr(info, "last_price", None)
            prev_close  = getattr(info, "previous_close", None)
            open_price  = getattr(info, "open", None)
            day_high    = getattr(info, "day_high", None)
            day_low     = getattr(info, "day_low", None)
            volume      = getattr(info, "last_volume", None)
            market_cap  = getattr(info, "market_cap", None)
            shares      = getattr(info, "shares", None)
            year_high   = getattr(info, "year_high", None)
            year_low    = getattr(info, "year_low", None)
            exchange    = getattr(info, "exchange", "")

            change    = (price - prev_close) if price and prev_close else None
            change_pc = (change / prev_close * 100) if change and prev_close else None
            direction = "▲" if (change or 0) >= 0 else "▼"

            lines = [
                f"## {ticker}  —  Stock Quote",
                f"Exchange:       {exchange}",
                "",
                f"Price:          ${price:.2f}"          if price     else "Price: N/A",
                f"Change:         {direction} {abs(change):.2f}  "
                f"({abs(change_pc):.2f}%)"               if change    else "Change: N/A",
                "",
                f"Open:           ${open_price:.2f}"     if open_price else "Open: N/A",
                f"Day range:      ${day_low:.2f} – ${day_high:.2f}"
                if day_low and day_high else "Day range: N/A",
                f"52-week range:  ${year_low:.2f} – ${year_high:.2f}"
                if year_low and year_high else "52-week range: N/A",
                "",
                f"Volume:         {_fmt_number(volume)}",
                f"Market cap:     {_fmt_number(market_cap, prefix='$')}",
                f"Shares out:     {_fmt_number(shares)}",
            ]
            return "\n".join(lines)

        except Exception as exc:
            logger.error("StockQuoteTool failed for %s: %s", ticker, exc)
            return f"Could not fetch quote for '{ticker}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 2 — Company Info
# =============================================================================

class CompanyInfoTool(BaseTool):
    """
    Fetch a comprehensive company profile for a ticker.

    Returns: full name, sector, industry, description, employees,
    headquarters, website, and key executives.
    """

    name: str = "company_info"
    description: str = (
        "Get a company profile: business description, sector, industry, "
        "headquarters, number of employees, and executives. "
        "Use for: 'What does Apple do?', 'Tell me about Tesla as a company'."
    )
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info

            name        = _safe(info, "longName") or _safe(info, "shortName") or ticker
            sector      = _safe(info, "sector",       "N/A")
            industry    = _safe(info, "industry",     "N/A")
            country     = _safe(info, "country",      "N/A")
            city        = _safe(info, "city",         "")
            employees   = _safe(info, "fullTimeEmployees")
            website     = _safe(info, "website",      "N/A")
            description = _safe(info, "longBusinessSummary", "No description available.")
            currency    = _safe(info, "currency",     "USD")

            # Top executives
            officers = info.get("companyOfficers", [])[:3]
            exec_lines = [
                f"  • {o.get('name', '?')}  —  {o.get('title', '?')}"
                for o in officers
            ]

            lines = [
                f"## {name}  ({ticker})",
                "",
                f"Sector:      {sector}",
                f"Industry:    {industry}",
                f"Employees:   {_fmt_number(employees) if employees else 'N/A'}",
                f"HQ:          {city + ', ' if city else ''}{country}",
                f"Website:     {website}",
                f"Currency:    {currency}",
                "",
                "### Business Summary",
                description[:800] + ("…" if len(description) > 800 else ""),
            ]
            if exec_lines:
                lines += ["", "### Key Executives"] + exec_lines

            return "\n".join(lines)

        except Exception as exc:
            logger.error("CompanyInfoTool failed for %s: %s", ticker, exc)
            return f"Could not fetch company info for '{ticker}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 3 — Financial Statements
# =============================================================================

class FinancialStatementsTool(BaseTool):
    """
    Fetch income statement, balance sheet, or cash flow statement.

    Returns the most recent 4 periods (annual or quarterly) with
    the most important line items formatted for readability.
    """

    name: str = "financial_statements"
    description: str = (
        "Fetch financial statements for a company: income statement, "
        "balance sheet, or cash flow statement. "
        "Use for: 'Show me Apple's income statement', "
        "'TSLA balance sheet quarterly', 'Amazon revenue last 4 years'."
    )
    args_schema: Type[BaseModel] = StatementInput

    _INCOME_KEYS = [
        ("Total Revenue",           "Total Revenue"),
        ("Gross Profit",            "Gross Profit"),
        ("Operating Income",        "Operating Income"),
        ("EBITDA",                  "EBITDA"),
        ("Net Income",              "Net Income"),
        ("Basic EPS",               "EPS (Basic)"),
        ("Diluted EPS",             "EPS (Diluted)"),
        ("Research And Development","R&D Expense"),
        ("Selling General And Administrative", "SG&A"),
    ]
    _BALANCE_KEYS = [
        ("Total Assets",            "Total Assets"),
        ("Total Liabilities Net Minority Interest", "Total Liabilities"),
        ("Stockholders Equity",     "Shareholders' Equity"),
        ("Cash And Cash Equivalents", "Cash & Equivalents"),
        ("Total Debt",              "Total Debt"),
        ("Current Assets",          "Current Assets"),
        ("Current Liabilities",     "Current Liabilities"),
        ("Inventory",               "Inventory"),
        ("Accounts Receivable",     "Accounts Receivable"),
        ("Retained Earnings",       "Retained Earnings"),
    ]
    _CASHFLOW_KEYS = [
        ("Operating Cash Flow",     "Operating Cash Flow"),
        ("Investing Cash Flow",     "Investing Cash Flow"),
        ("Financing Cash Flow",     "Financing Cash Flow"),
        ("Free Cash Flow",          "Free Cash Flow"),
        ("Capital Expenditure",     "CapEx"),
        ("Dividends Paid",          "Dividends Paid"),
        ("Net Income",              "Net Income"),
    ]

    def _run(
        self,
        ticker: str,
        statement: str = "income",
        quarterly: bool = False,
    ) -> str:
        ticker = ticker.strip().upper()
        stmt   = statement.lower().strip()

        try:
            import yfinance as yf
            import pandas as pd

            t    = yf.Ticker(ticker)
            freq = "Quarterly" if quarterly else "Annual"

            if stmt in ("income", "income_statement", "profit", "pl", "p&l"):
                df        = t.quarterly_income_stmt if quarterly else t.income_stmt
                title     = "Income Statement"
                key_map   = self._INCOME_KEYS
            elif stmt in ("balance", "balance_sheet", "bs"):
                df        = t.quarterly_balance_sheet if quarterly else t.balance_sheet
                title     = "Balance Sheet"
                key_map   = self._BALANCE_KEYS
            elif stmt in ("cashflow", "cash_flow", "cf"):
                df        = t.quarterly_cashflow if quarterly else t.cashflow
                title     = "Cash Flow Statement"
                key_map   = self._CASHFLOW_KEYS
            else:
                return (
                    f"Unknown statement type '{statement}'. "
                    "Use 'income', 'balance', or 'cashflow'."
                )

            if df is None or df.empty:
                return f"No {title} data available for {ticker}."

            # Take the most recent 4 periods, oldest-first for natural reading
            df    = df.iloc[:, :4]
            cols  = [c.strftime("%Y") if quarterly is False else c.strftime("%b %Y")
                     for c in df.columns]

            lines = [
                f"## {ticker}  —  {freq} {title}",
                "",
                f"{'Metric':<30}" + "".join(f"{c:>16}" for c in cols),
                "─" * (30 + 16 * len(cols)),
            ]

            for raw_key, label in key_map:
                if raw_key in df.index:
                    row   = df.loc[raw_key]
                    cells = []
                    for v in row:
                        try:
                            cells.append(f"{float(v) / 1e9:>14.2f}B")
                        except (TypeError, ValueError):
                            cells.append(f"{'N/A':>16}")
                    lines.append(f"{label:<30}" + "".join(cells))

            lines.append("")
            lines.append("Values in USD billions unless noted. Source: Yahoo Finance.")
            return "\n".join(lines)

        except Exception as exc:
            logger.error("FinancialStatementsTool failed for %s: %s", ticker, exc)
            return f"Could not fetch {statement} statement for '{ticker}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 4 — Financial Ratios
# =============================================================================

class FinancialRatiosTool(BaseTool):
    """
    Calculate and interpret key financial ratios for a company.

    Covers valuation (P/E, P/B, P/S, EV/EBITDA), profitability (ROE, ROA,
    margins), liquidity (current ratio), leverage (D/E), and dividends.
    Each ratio is graded with a plain-English quality signal.
    """

    name: str = "financial_ratios"
    description: str = (
        "Calculate valuation, profitability, liquidity, and leverage ratios. "
        "Each ratio includes a qualitative grade. "
        "Use for: 'Is Apple overvalued?', 'MSFT profitability ratios', "
        "'Compare P/E of Tesla vs industry'."
    )
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info

            name = _safe(info, "longName") or ticker

            # ── Valuation ──────────────────────────────────────────────────
            pe          = _safe(info, "trailingPE")
            forward_pe  = _safe(info, "forwardPE")
            pb          = _safe(info, "priceToBook")
            ps          = _safe(info, "priceToSalesTrailing12Months")
            ev_ebitda   = _safe(info, "enterpriseToEbitda")
            peg         = _safe(info, "pegRatio")

            # ── Profitability ──────────────────────────────────────────────
            gross_margin  = _safe(info, "grossMargins")
            oper_margin   = _safe(info, "operatingMargins")
            profit_margin = _safe(info, "profitMargins")
            roe           = _safe(info, "returnOnEquity")
            roa           = _safe(info, "returnOnAssets")
            revenue_growth= _safe(info, "revenueGrowth")
            earnings_growth= _safe(info, "earningsGrowth")

            # ── Liquidity ──────────────────────────────────────────────────
            current_ratio = _safe(info, "currentRatio")
            quick_ratio   = _safe(info, "quickRatio")

            # ── Leverage ───────────────────────────────────────────────────
            de_ratio      = _safe(info, "debtToEquity")
            total_debt    = _safe(info, "totalDebt")
            fcf           = _safe(info, "freeCashflow")

            # ── Dividends ──────────────────────────────────────────────────
            div_yield     = _safe(info, "dividendYield")
            div_rate      = _safe(info, "dividendRate")
            payout        = _safe(info, "payoutRatio")
            ex_date       = _safe(info, "exDividendDate")

            lines = [
                f"## {name}  ({ticker})  —  Financial Ratios",
                "",
                "### Valuation",
                f"  P/E  (trailing):   {_fmt_ratio(pe)}{_grade_ratio('pe', pe)}",
                f"  P/E  (forward):    {_fmt_ratio(forward_pe)}{_grade_ratio('pe', forward_pe)}",
                f"  P/B:               {_fmt_ratio(pb)}{_grade_ratio('pb', pb)}",
                f"  P/S:               {_fmt_ratio(ps)}",
                f"  EV/EBITDA:         {_fmt_ratio(ev_ebitda)}",
                f"  PEG ratio:         {_fmt_ratio(peg)}",
                "",
                "### Profitability",
                f"  Gross margin:      {_fmt_pct(gross_margin)}",
                f"  Operating margin:  {_fmt_pct(oper_margin)}",
                f"  Net margin:        {_fmt_pct(profit_margin)}",
                f"  Return on Equity:  {_fmt_pct(roe)}{_grade_ratio('roe', roe)}",
                f"  Return on Assets:  {_fmt_pct(roa)}{_grade_ratio('roa', roa)}",
                f"  Revenue growth:    {_fmt_pct(revenue_growth)}",
                f"  Earnings growth:   {_fmt_pct(earnings_growth)}",
                "",
                "### Liquidity",
                f"  Current ratio:     {_fmt_ratio(current_ratio)}{_grade_ratio('current', current_ratio)}",
                f"  Quick ratio:       {_fmt_ratio(quick_ratio)}",
                "",
                "### Leverage",
                f"  Debt / Equity:     {_fmt_ratio(de_ratio)}{_grade_ratio('de', de_ratio)}",
                f"  Total debt:        {_fmt_number(total_debt, prefix='$')}",
                f"  Free cash flow:    {_fmt_number(fcf, prefix='$')}",
            ]

            if div_yield or div_rate:
                lines += [
                    "",
                    "### Dividends",
                    f"  Annual yield:      {_fmt_pct(div_yield)}",
                    f"  Annual rate:       {_fmt_number(div_rate, prefix='$')} per share",
                    f"  Payout ratio:      {_fmt_pct(payout)}",
                ]

            lines += ["", "Source: Yahoo Finance. Grades are qualitative guides only."]
            return "\n".join(lines)

        except Exception as exc:
            logger.error("FinancialRatiosTool failed for %s: %s", ticker, exc)
            return f"Could not calculate ratios for '{ticker}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 5 — Price History
# =============================================================================

class PriceHistoryTool(BaseTool):
    """
    Fetch historical OHLCV price data and compute derived statistics.

    Returns: first/last price, total return, annualised return (CAGR),
    volatility (annualised std dev), max drawdown, and a condensed
    table of recent closing prices.
    """

    name: str = "price_history"
    description: str = (
        "Fetch historical price data and performance statistics for a ticker. "
        "Returns total return, CAGR, annualised volatility, and max drawdown. "
        "Use for: 'AAPL performance last 5 years', "
        "'How has MSFT stock done since 2020?', 'BTC-USD 6 month history'."
    )
    args_schema: Type[BaseModel] = TickerPeriodInput

    def _run(
        self,
        ticker:   str,
        period:   str = "1y",
        interval: str = "1d",
    ) -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf
            import math

            hist = yf.Ticker(ticker).history(period=period, interval=interval)

            if hist.empty:
                return (
                    f"No price history found for '{ticker}' "
                    f"(period={period}, interval={interval})."
                )

            close      = hist["Close"]
            first_px   = float(close.iloc[0])
            last_px    = float(close.iloc[-1])
            n_periods  = len(close)
            start_date = hist.index[0].strftime("%Y-%m-%d")
            end_date   = hist.index[-1].strftime("%Y-%m-%d")

            # Total and annualised return
            total_ret  = (last_px - first_px) / first_px
            years      = n_periods / 252 if interval == "1d" else n_periods / 52
            cagr       = ((1 + total_ret) ** (1 / max(years, 0.01))) - 1 if years else None

            # Annualised volatility (std dev of daily log returns × √252)
            log_returns = close.pct_change().dropna()
            ann_vol     = float(log_returns.std() * math.sqrt(252)) if interval == "1d" else None

            # Maximum drawdown
            rolling_max = close.cummax()
            drawdowns   = (close - rolling_max) / rolling_max
            max_dd      = float(drawdowns.min())

            # Recent price table (last 10 data points)
            recent      = hist.tail(10)[["Open", "High", "Low", "Close", "Volume"]]
            table_lines = [
                f"{'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12}",
                "─" * 60,
            ]
            for ts, row in recent.iterrows():
                date_str = ts.strftime("%Y-%m-%d")
                table_lines.append(
                    f"{date_str:<12} "
                    f"${row['Open']:>7.2f} "
                    f"${row['High']:>7.2f} "
                    f"${row['Low']:>7.2f} "
                    f"${row['Close']:>7.2f} "
                    f"{_fmt_number(row['Volume']):>12}"
                )

            direction = "▲" if total_ret >= 0 else "▼"

            lines = [
                f"## {ticker}  —  Price History  ({period}, {interval})",
                f"Period:             {start_date}  →  {end_date}  ({n_periods} bars)",
                "",
                f"Start price:        ${first_px:.2f}",
                f"End price:          ${last_px:.2f}",
                f"Total return:       {direction} {abs(total_ret) * 100:.2f}%",
            ]
            if cagr is not None:
                lines.append(f"CAGR:               {cagr * 100:.2f}%")
            if ann_vol is not None:
                lines.append(f"Annualised vol:     {ann_vol * 100:.2f}%")
            lines.append(f"Max drawdown:       {max_dd * 100:.2f}%")
            lines.append(f"High (period):      ${float(hist['High'].max()):.2f}")
            lines.append(f"Low  (period):      ${float(hist['Low'].min()):.2f}")
            lines += ["", "### Recent Prices"] + table_lines
            return "\n".join(lines)

        except Exception as exc:
            logger.error("PriceHistoryTool failed for %s: %s", ticker, exc)
            return f"Could not fetch price history for '{ticker}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 6 — Stock Comparison
# =============================================================================

class StockComparisonTool(BaseTool):
    """
    Compare up to 5 tickers side-by-side on key financial metrics.

    Returns a formatted table covering price, market cap, P/E, P/B,
    revenue, net income, gross margin, ROE, dividend yield, and beta.
    """

    name: str = "stock_comparison"
    description: str = (
        "Compare multiple stocks side-by-side on price, valuation, "
        "profitability, and dividend metrics. "
        "Use for: 'Compare AAPL vs MSFT', 'FAANG stock comparison', "
        "'Which is cheaper: TSLA or GM?'"
    )
    args_schema: Type[BaseModel] = CompareInput

    _METRICS: list[tuple[str, str, callable]] = [
        ("Price",            "currentPrice",                  lambda v: f"${v:.2f}"),
        ("Market Cap",       "marketCap",                     lambda v: _fmt_number(v, "$")),
        ("P/E (trailing)",   "trailingPE",                    lambda v: f"{v:.1f}x"),
        ("P/E (forward)",    "forwardPE",                     lambda v: f"{v:.1f}x"),
        ("P/B",              "priceToBook",                   lambda v: f"{v:.2f}x"),
        ("Revenue (TTM)",    "totalRevenue",                  lambda v: _fmt_number(v, "$")),
        ("Net Income (TTM)", "netIncomeToCommon",             lambda v: _fmt_number(v, "$")),
        ("Gross Margin",     "grossMargins",                  lambda v: _fmt_pct(v)),
        ("Net Margin",       "profitMargins",                 lambda v: _fmt_pct(v)),
        ("ROE",              "returnOnEquity",                lambda v: _fmt_pct(v)),
        ("Debt/Equity",      "debtToEquity",                  lambda v: f"{v:.2f}x"),
        ("Dividend Yield",   "dividendYield",                 lambda v: _fmt_pct(v)),
        ("Beta",             "beta",                          lambda v: f"{v:.2f}"),
        ("52-wk Return",     "52WeekChange",                  lambda v: _fmt_pct(v)),
    ]

    def _run(self, tickers: str) -> str:
        ticker_list = [t.strip().upper() for t in tickers.split(",")][:5]
        if len(ticker_list) < 2:
            return "Please provide at least 2 tickers separated by commas."

        try:
            import yfinance as yf

            infos: dict[str, dict] = {}
            for sym in ticker_list:
                try:
                    infos[sym] = yf.Ticker(sym).info
                except Exception as exc:
                    logger.warning("Could not fetch %s: %s", sym, exc)
                    infos[sym] = {}

            # Column widths
            col_w     = max(12, max(len(t) for t in ticker_list) + 2)
            label_w   = 20
            header    = f"{'Metric':<{label_w}}" + "".join(
                f"{sym:>{col_w}}" for sym in ticker_list
            )
            separator = "─" * (label_w + col_w * len(ticker_list))

            rows = [
                f"## Stock Comparison: {' vs '.join(ticker_list)}",
                "",
                header,
                separator,
            ]

            for label, key, fmt in self._METRICS:
                cells = []
                for sym in ticker_list:
                    v = _safe(infos.get(sym, {}), key)
                    try:
                        cells.append(fmt(v) if v is not None else "N/A")
                    except Exception:
                        cells.append("N/A")
                row = f"{label:<{label_w}}" + "".join(
                    f"{c:>{col_w}}" for c in cells
                )
                rows.append(row)

            rows += ["", "Source: Yahoo Finance. N/A = data unavailable."]
            return "\n".join(rows)

        except Exception as exc:
            logger.error("StockComparisonTool failed: %s", exc)
            return f"Could not compare tickers: {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Registry
# =============================================================================

def get_financial_tools() -> list[BaseTool]:
    """Return all financial analysis tools."""
    return [
        StockQuoteTool(),
        CompanyInfoTool(),
        FinancialStatementsTool(),
        FinancialRatiosTool(),
        PriceHistoryTool(),
        StockComparisonTool(),
    ]
