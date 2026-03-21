"""
tools/financial_tools.py
─────────────────────────
LangChain tools for the Financial Agent.

Data retrieval strategy (two-tier):
  1. yfinance  — attempted first; gives real-time market data when the
                 network can reach Yahoo Finance.
  2. LLM fallback — when yfinance raises any network/parse error, the tool
                    calls get_llm() and asks for the same information from
                    the model's training knowledge.  This ensures the agent
                    always returns a useful answer even in offline, sandboxed,
                    or rate-limited environments.

Tools:
  StockQuoteTool          – Price, change, volume, market cap, 52-week range.
  CompanyInfoTool         – Business description, sector, employees, executives.
  FinancialStatementsTool – Income statement / balance sheet / cash flow.
  FinancialRatiosTool     – Valuation, profitability, liquidity and leverage.
  PriceHistoryTool        – Historical performance statistics.
  StockComparisonTool     – Side-by-side table for up to 5 tickers.

Ticker handling:
  Every tool receives the ticker UPPERCASED by the caller.
  Tools never attempt to re-extract a ticker from free text.
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
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_ratio(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}x"
    except (TypeError, ValueError):
        return "N/A"


def _safe(info: dict, key: str, default=None):
    v = info.get(key, default)
    return default if v in (None, "N/A", float("inf"), float("-inf")) else v


def _grade_ratio(metric: str, value) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    thresholds = {
        "pe":      (10, 20, 35),
        "pb":      (1, 3, 5),
        "roe":     (0.15, 0.20, 0.30),
        "roa":     (0.05, 0.10, 0.20),
        "current": (1.5, 2.5, 4.0),
        "de":      (0.5, 1.5, 3.0),
    }
    t = thresholds.get(metric)
    if not t:
        return ""
    lo, mid, hi = t
    if metric in ("roe", "roa", "current"):
        if value >= hi:   return "  ★ Excellent"
        if value >= mid:  return "  ✓ Good"
        if value >= lo:   return "  ~ Fair"
        return "  ✗ Weak"
    else:
        if value <= lo:   return "  ★ Attractive"
        if value <= mid:  return "  ✓ Fair"
        if value <= hi:   return "  ~ Expensive"
        return "  ✗ Very expensive"


# =============================================================================
#  LLM fallback helper
# =============================================================================

def _llm_fallback(prompt: str, context_label: str) -> str:
    """
    Use the configured LLM to answer a financial query when yfinance is
    unavailable (network restricted, rate-limited, etc.).

    The LLM is instructed to respond from its training knowledge and to
    clearly note that the data is not live.

    Returns the LLM response, or an informative error string on failure.
    """
    try:
        from core.llm_manager import get_llm
        llm    = get_llm()
        result = llm.invoke(prompt)
        text   = str(result.content).strip()
        logger.info("LLM knowledge fallback used for: %s", context_label)
        return text
    except Exception as exc:
        logger.error("LLM fallback also failed for %s: %s", context_label, exc)
        return (
            f"Unable to retrieve financial data for {context_label}. "
            f"yfinance is network-restricted and the LLM fallback encountered: {exc}. "
            "Please run with a network connection to Yahoo Finance."
        )


# =============================================================================
#  Schemas
# =============================================================================

class TickerInput(BaseModel):
    ticker: str = Field(...,
                        description="Stock ticker symbol in UPPERCASE, e.g. AAPL, MSFT, BTC-USD")


class TickerPeriodInput(BaseModel):
    ticker:   str = Field(..., description="Stock ticker symbol")
    period:   str = Field(default="1y",
                          description="Period: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max")
    interval: str = Field(default="1d",
                          description="Interval: 1m 5m 15m 30m 60m 1d 1wk 1mo")


class StatementInput(BaseModel):
    ticker:    str  = Field(..., description="Stock ticker symbol")
    statement: str  = Field(default="income",
                            description="Statement type: income | balance | cashflow")
    quarterly: bool = Field(default=False,
                            description="True for quarterly data, False for annual")


class CompareInput(BaseModel):
    tickers: str = Field(...,
                         description="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL")


# =============================================================================
#  Tool 1 — Stock Quote
# =============================================================================

class StockQuoteTool(BaseTool):
    name: str        = "stock_quote"
    description: str = (
        "Get the current stock price and key market data for a ticker symbol. "
        "Returns price, change, volume, market cap, and 52-week range. "
        "Use for: What is AAPL trading at?, Current Tesla price."
    )
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf
            t         = yf.Ticker(ticker)
            fast      = t.fast_info
            price     = getattr(fast, "last_price",     None)
            prev      = getattr(fast, "previous_close", None)
            day_high  = getattr(fast, "day_high",       None)
            day_low   = getattr(fast, "day_low",        None)
            volume    = getattr(fast, "last_volume",    None)
            mkt_cap   = getattr(fast, "market_cap",     None)
            yr_high   = getattr(fast, "year_high",      None)
            yr_low    = getattr(fast, "year_low",       None)
            exchange  = getattr(fast, "exchange",       "")
            if price is None:
                raise ValueError("No price data from yfinance")
            change    = (price - prev) if price and prev else None
            change_pc = (change / prev * 100) if change and prev else None
            direction = "▲" if (change or 0) >= 0 else "▼"
            lines = [
                f"## {ticker}  —  Stock Quote",
                f"Exchange:       {exchange}",
                "",
                f"Price:          ${price:.2f}",
                (f"Change:         {direction} ${abs(change):.2f}  ({abs(change_pc):.2f}%)"
                 if change else "Change:         N/A"),
                "",
                (f"Day range:      ${day_low:.2f} – ${day_high:.2f}"
                 if day_low and day_high else "Day range:      N/A"),
                (f"52-week range:  ${yr_low:.2f} – ${yr_high:.2f}"
                 if yr_low and yr_high else "52-week range:  N/A"),
                "",
                f"Volume:         {_fmt_number(volume)}",
                f"Market cap:     {_fmt_number(mkt_cap, prefix='$')}",
            ]
            return "\n".join(lines)
        except Exception as yf_exc:
            logger.warning("yfinance quote failed for %s (%s); LLM fallback.", ticker, yf_exc)
        prompt = (
            f"Provide a stock quote summary for the ticker {ticker}. "
            "Include: approximate current price, recent daily change, "
            "approximate market cap, 52-week high/low, exchange, "
            "and a one-line business description. "
            "Format with clearly labelled fields. "
            "State clearly that figures are from training knowledge, not live data."
        )
        return _llm_fallback(prompt, f"{ticker} quote")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 2 — Company Info
# =============================================================================

class CompanyInfoTool(BaseTool):
    name: str        = "company_info"
    description: str = (
        "Get a company profile: business description, sector, industry, "
        "headquarters, employees, and key executives. "
        "Use for: What does Apple do?, Tell me about Microsoft."
    )
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            if not info or len(info) < 5:
                raise ValueError("yfinance returned empty info")
            name        = _safe(info, "longName") or ticker
            sector      = _safe(info, "sector",   "N/A")
            industry    = _safe(info, "industry", "N/A")
            country     = _safe(info, "country",  "N/A")
            city        = _safe(info, "city",     "")
            employees   = _safe(info, "fullTimeEmployees")
            website     = _safe(info, "website",  "N/A")
            description = _safe(info, "longBusinessSummary", "No description.")
            officers    = info.get("companyOfficers", [])[:3]
            exec_lines  = [f"  • {o.get('name','?')  }  —  {o.get('title','?')}"
                           for o in officers]
            lines = [
                f"## {name}  ({ticker})",
                "",
                f"Sector:      {sector}",
                f"Industry:    {industry}",
                f"Employees:   {_fmt_number(employees) if employees else 'N/A'}",
                f"HQ:          {city + ', ' if city else ''}{country}",
                f"Website:     {website}",
                "",
                "### Business Summary",
                description[:800] + ("…" if len(description) > 800 else ""),
                ]
            if exec_lines:
                lines += ["", "### Key Executives"] + exec_lines
            return "\n".join(lines)
        except Exception as yf_exc:
            logger.warning("yfinance info failed for %s (%s); LLM fallback.", ticker, yf_exc)
        prompt = (
            f"Provide a detailed company profile for the stock ticker {ticker}. "
            "Include: full company name, sector, industry, country/HQ, "
            "approximate employee count, website URL, "
            "a 2–3 sentence business description, "
            "and 2–3 key executives with their titles. "
            "Format with clearly labelled sections. "
            "State that this is from training knowledge, not live data."
        )
        return _llm_fallback(prompt, f"{ticker} company info")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 3 — Financial Statements
# =============================================================================

class FinancialStatementsTool(BaseTool):
    name: str        = "financial_statements"
    description: str = (
        "Fetch financial statements: income statement, balance sheet, or cash flow. "
        "Use for: Apple income statement, TSLA balance sheet quarterly, "
        "Amazon revenue last 4 years."
    )
    args_schema: Type[BaseModel] = StatementInput

    _INCOME_KEYS  = [
        ("Total Revenue",                          "Total Revenue"),
        ("Gross Profit",                           "Gross Profit"),
        ("Operating Income",                       "Operating Income"),
        ("EBITDA",                                 "EBITDA"),
        ("Net Income",                             "Net Income"),
        ("Basic EPS",                              "EPS (Basic)"),
        ("Research And Development",               "R&D Expense"),
        ("Selling General And Administrative",     "SG&A"),
    ]
    _BALANCE_KEYS = [
        ("Total Assets",                           "Total Assets"),
        ("Total Liabilities Net Minority Interest","Total Liabilities"),
        ("Stockholders Equity",                    "Shareholders Equity"),
        ("Cash And Cash Equivalents",              "Cash & Equivalents"),
        ("Total Debt",                             "Total Debt"),
        ("Current Assets",                        "Current Assets"),
        ("Current Liabilities",                    "Current Liabilities"),
    ]
    _CASHFLOW_KEYS = [
        ("Operating Cash Flow",                    "Operating Cash Flow"),
        ("Investing Cash Flow",                    "Investing Cash Flow"),
        ("Financing Cash Flow",                    "Financing Cash Flow"),
        ("Free Cash Flow",                         "Free Cash Flow"),
        ("Capital Expenditure",                    "CapEx"),
    ]

    def _run(self, ticker: str, statement: str = "income", quarterly: bool = False) -> str:
        ticker = ticker.strip().upper()
        stmt   = statement.lower().strip()
        try:
            import yfinance as yf
            t    = yf.Ticker(ticker)
            freq = "Quarterly" if quarterly else "Annual"
            if stmt in ("income", "income_statement", "profit", "pl", "p&l"):
                df, title, key_map = (t.quarterly_income_stmt if quarterly else t.income_stmt), "Income Statement", self._INCOME_KEYS
            elif stmt in ("balance", "balance_sheet", "bs"):
                df, title, key_map = (t.quarterly_balance_sheet if quarterly else t.balance_sheet), "Balance Sheet", self._BALANCE_KEYS
            elif stmt in ("cashflow", "cash_flow", "cf"):
                df, title, key_map = (t.quarterly_cashflow if quarterly else t.cashflow), "Cash Flow Statement", self._CASHFLOW_KEYS
            else:
                return f"Unknown statement type '{statement}'. Use income, balance, or cashflow."
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            df   = df.iloc[:, :4]
            cols = [c.strftime("%Y") if not quarterly else c.strftime("%b %Y") for c in df.columns]
            lines = [
                f"## {ticker}  —  {freq} {title}",
                "",
                f"{'Metric':<30}" + "".join(f"{c:>16}" for c in cols),
                "─" * (30 + 16 * len(cols)),
                ]
            for raw_key, label in key_map:
                if raw_key in df.index:
                    row = df.loc[raw_key]
                    cells = []
                    for v in row:
                        try:    cells.append(f"{float(v)/1e9:>14.2f}B")
                        except: cells.append(f"{'N/A':>16}")
                    lines.append(f"{label:<30}" + "".join(cells))
            lines += ["", "Values in USD billions. Source: Yahoo Finance."]
            return "\n".join(lines)
        except Exception as yf_exc:
            logger.warning("yfinance statements failed for %s (%s); LLM fallback.", ticker, yf_exc)
        freq_str = "quarterly" if quarterly else "annual"
        stmt_map = {
            "income":   "income statement covering: revenue, gross profit, operating income, EBITDA, net income, EPS, R&D, SG&A",
            "balance":  "balance sheet covering: total assets, total liabilities, equity, cash, total debt, current assets/liabilities",
            "cashflow": "cash flow statement covering: operating, investing, financing cash flows, free cash flow, capex",
        }
        stmt_desc = stmt_map.get(stmt, stmt_map["income"])
        prompt = (
            f"Provide the {freq_str} {stmt_desc} for {ticker} "
            "for the most recent 3–4 periods. "
            "Format as a table with period columns and metric rows. "
            "Express values in billions USD. "
            "State clearly this is from training knowledge, not live filings."
        )
        return _llm_fallback(prompt, f"{ticker} {stmt} statement")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 4 — Financial Ratios
# =============================================================================

class FinancialRatiosTool(BaseTool):
    name: str        = "financial_ratios"
    description: str = (
        "Calculate key financial ratios with qualitative grades: "
        "P/E, P/B, ROE, ROA, margins, debt/equity, dividend yield. "
        "Use for: Is Apple overvalued?, MSFT profitability metrics."
    )
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            if not info or len(info) < 5:
                raise ValueError("insufficient info from yfinance")
            name        = _safe(info, "longName") or ticker
            pe          = _safe(info, "trailingPE")
            fwd_pe      = _safe(info, "forwardPE")
            pb          = _safe(info, "priceToBook")
            ps          = _safe(info, "priceToSalesTrailing12Months")
            ev_ebitda   = _safe(info, "enterpriseToEbitda")
            peg         = _safe(info, "pegRatio")
            gross_m     = _safe(info, "grossMargins")
            oper_m      = _safe(info, "operatingMargins")
            net_m       = _safe(info, "profitMargins")
            roe         = _safe(info, "returnOnEquity")
            roa         = _safe(info, "returnOnAssets")
            rev_gr      = _safe(info, "revenueGrowth")
            earn_gr     = _safe(info, "earningsGrowth")
            cur_ratio   = _safe(info, "currentRatio")
            quick       = _safe(info, "quickRatio")
            de          = _safe(info, "debtToEquity")
            tot_debt    = _safe(info, "totalDebt")
            fcf         = _safe(info, "freeCashflow")
            div_yield   = _safe(info, "dividendYield")
            div_rate    = _safe(info, "dividendRate")
            payout      = _safe(info, "payoutRatio")
            lines = [
                f"## {name}  ({ticker})  —  Financial Ratios",
                "",
                "### Valuation",
                f"  P/E  (trailing):   {_fmt_ratio(pe)}{_grade_ratio('pe', pe)}",
                f"  P/E  (forward):    {_fmt_ratio(fwd_pe)}{_grade_ratio('pe', fwd_pe)}",
                f"  P/B:               {_fmt_ratio(pb)}{_grade_ratio('pb', pb)}",
                f"  P/S:               {_fmt_ratio(ps)}",
                f"  EV/EBITDA:         {_fmt_ratio(ev_ebitda)}",
                f"  PEG ratio:         {_fmt_ratio(peg)}",
                "",
                "### Profitability",
                f"  Gross margin:      {_fmt_pct(gross_m)}",
                f"  Operating margin:  {_fmt_pct(oper_m)}",
                f"  Net margin:        {_fmt_pct(net_m)}",
                f"  Return on Equity:  {_fmt_pct(roe)}{_grade_ratio('roe', roe)}",
                f"  Return on Assets:  {_fmt_pct(roa)}{_grade_ratio('roa', roa)}",
                f"  Revenue growth:    {_fmt_pct(rev_gr)}",
                f"  Earnings growth:   {_fmt_pct(earn_gr)}",
                "",
                "### Liquidity",
                f"  Current ratio:     {_fmt_ratio(cur_ratio)}{_grade_ratio('current', cur_ratio)}",
                f"  Quick ratio:       {_fmt_ratio(quick)}",
                "",
                "### Leverage",
                f"  Debt / Equity:     {_fmt_ratio(de)}{_grade_ratio('de', de)}",
                f"  Total debt:        {_fmt_number(tot_debt, prefix='$')}",
                f"  Free cash flow:    {_fmt_number(fcf, prefix='$')}",
            ]
            if div_yield or div_rate:
                lines += [
                    "",
                    "### Dividends",
                    f"  Annual yield:  {_fmt_pct(div_yield)}",
                    f"  Annual rate:   {_fmt_number(div_rate, prefix='$')} per share",
                    f"  Payout ratio:  {_fmt_pct(payout)}",
                ]
            lines += ["", "Source: Yahoo Finance. Grades are qualitative guides only."]
            return "\n".join(lines)
        except Exception as yf_exc:
            logger.warning("yfinance ratios failed for %s (%s); LLM fallback.", ticker, yf_exc)
        prompt = (
            f"Provide key financial ratios for the stock {ticker} with commentary. Include:\n"
            "Valuation: P/E (trailing and forward), P/B, P/S, EV/EBITDA, PEG\n"
            "Profitability: gross margin, operating margin, net margin, ROE, ROA, "
            "revenue growth, earnings growth\n"
            "Liquidity: current ratio, quick ratio\n"
            "Leverage: debt/equity, total debt, free cash flow\n"
            "Dividends: yield and payout ratio (if applicable)\n"
            "For each section add a brief qualitative comment comparing to industry norms.\n"
            "Format with clearly labelled sections. "
            "State this is from training knowledge, not live data."
        )
        return _llm_fallback(prompt, f"{ticker} ratios")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 5 — Price History
# =============================================================================

class PriceHistoryTool(BaseTool):
    name: str        = "price_history"
    description: str = (
        "Fetch historical price data and performance statistics: "
        "total return, CAGR, volatility, max drawdown, recent prices. "
        "Use for: AAPL performance last 5 years, How has MSFT done since 2020?"
    )
    args_schema: Type[BaseModel] = TickerPeriodInput

    def _run(self, ticker: str, period: str = "1y", interval: str = "1d") -> str:
        ticker = ticker.strip().upper()
        try:
            import yfinance as yf, math
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            if hist is None or hist.empty:
                raise ValueError("empty history")
            close      = hist["Close"]
            first_px   = float(close.iloc[0])
            last_px    = float(close.iloc[-1])
            n          = len(close)
            start_date = hist.index[0].strftime("%Y-%m-%d")
            end_date   = hist.index[-1].strftime("%Y-%m-%d")
            total_ret  = (last_px - first_px) / first_px
            years      = n / 252 if interval == "1d" else n / 52
            cagr       = ((1 + total_ret) ** (1 / max(years, 0.01))) - 1
            ann_vol    = float(close.pct_change().dropna().std() * math.sqrt(252)) if interval=="1d" else None
            max_dd     = float(((close - close.cummax()) / close.cummax()).min())
            direction  = "▲" if total_ret >= 0 else "▼"
            recent     = hist.tail(8)[["Open","High","Low","Close","Volume"]]
            table = [f"{'Date':<12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12}",
                     "─" * 60]
            for ts, row in recent.iterrows():
                table.append(
                    f"{ts.strftime('%Y-%m-%d'):<12} "
                    f"${row['Open']:>7.2f} ${row['High']:>7.2f} "
                    f"${row['Low']:>7.2f} ${row['Close']:>7.2f} "
                    f"{_fmt_number(row['Volume']):>12}"
                )
            lines = [
                f"## {ticker}  —  Price History  ({period}, {interval})",
                f"Period: {start_date} → {end_date}  ({n} bars)",
                "",
                f"Start price:    ${first_px:.2f}",
                f"End price:      ${last_px:.2f}",
                f"Total return:   {direction} {abs(total_ret) * 100:.2f}%",
                f"CAGR:           {cagr * 100:.2f}%",
            ]
            if ann_vol:
                lines.append(f"Ann. volatility:{ann_vol * 100:.2f}%")
            lines.append(f"Max drawdown:   {max_dd * 100:.2f}%")
            lines.append(f"Period high:    ${float(hist['High'].max()):.2f}")
            lines.append(f"Period low:     ${float(hist['Low'].min()):.2f}")
            lines += ["", "### Recent Prices"] + table
            return "\n".join(lines)
        except Exception as yf_exc:
            logger.warning("yfinance history failed for %s (%s); LLM fallback.", ticker, yf_exc)
        period_text = {
            "1d":"the past day","5d":"the past 5 days","1mo":"the past month",
            "3mo":"the past 3 months","6mo":"the past 6 months","1y":"the past year",
            "2y":"the past 2 years","5y":"the past 5 years","10y":"the past 10 years",
            "ytd":"year-to-date","max":"all time",
        }.get(period, period)
        prompt = (
            f"Provide a stock price performance summary for {ticker} over {period_text}. "
            "Include: approximate start and end prices, total return percentage, "
            "CAGR (if multi-year), approximate annualised volatility, "
            "estimated max drawdown, and period high/low. "
            "Also write 2–3 sentences on the key drivers of performance. "
            "State clearly this is from training knowledge, not live price data."
        )
        return _llm_fallback(prompt, f"{ticker} history {period}")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Tool 6 — Stock Comparison
# =============================================================================

class StockComparisonTool(BaseTool):
    name: str        = "stock_comparison"
    description: str = (
        "Compare up to 5 stocks side-by-side on price, valuation, "
        "profitability, and dividend metrics. "
        "Use for: Compare AAPL vs MSFT, FAANG comparison."
    )
    args_schema: Type[BaseModel] = CompareInput

    _METRICS = [
        ("Price",          "currentPrice"),
        ("Market Cap",     "marketCap"),
        ("P/E (trailing)", "trailingPE"),
        ("P/E (forward)",  "forwardPE"),
        ("P/B",            "priceToBook"),
        ("Revenue (TTM)",  "totalRevenue"),
        ("Net Income",     "netIncomeToCommon"),
        ("Gross Margin",   "grossMargins"),
        ("Net Margin",     "profitMargins"),
        ("ROE",            "returnOnEquity"),
        ("Debt/Equity",    "debtToEquity"),
        ("Div. Yield",     "dividendYield"),
        ("Beta",           "beta"),
        ("52-wk Return",   "52WeekChange"),
    ]

    def _fmt_cell(self, key: str, value) -> str:
        if value is None:
            return "N/A"
        try:
            v = float(value)
            if key == "currentPrice":      return f"${v:.2f}"
            if key in ("marketCap","totalRevenue","netIncomeToCommon"):
                return _fmt_number(v, "$")
            if key in ("grossMargins","profitMargins","returnOnEquity",
                       "dividendYield","52WeekChange"):
                return _fmt_pct(v)
            if key in ("trailingPE","forwardPE","priceToBook","debtToEquity"):
                return f"{v:.2f}x"
            return f"{v:.2f}"
        except (TypeError, ValueError):
            return "N/A"

    def _run(self, tickers: str) -> str:
        ticker_list = [t.strip().upper() for t in tickers.split(",")][:5]
        if len(ticker_list) < 2:
            return "Please provide at least 2 tickers separated by commas."
        try:
            import yfinance as yf
            infos = {}
            failures = 0
            for sym in ticker_list:
                try:
                    info = yf.Ticker(sym).info
                    infos[sym] = info if info and len(info) > 5 else {}
                    if not infos[sym]:
                        failures += 1
                except Exception:
                    infos[sym] = {}
                    failures += 1
            if failures == len(ticker_list):
                raise ValueError("yfinance failed for all tickers")
            col_w   = max(12, max(len(t) for t in ticker_list) + 2)
            label_w = 18
            rows = [
                f"## Stock Comparison: {' vs '.join(ticker_list)}",
                "",
                f"{'Metric':<{label_w}}" + "".join(f"{s:>{col_w}}" for s in ticker_list),
                "─" * (label_w + col_w * len(ticker_list)),
                ]
            for label, key in self._METRICS:
                cells = [self._fmt_cell(key, _safe(infos.get(s, {}), key))
                         for s in ticker_list]
                rows.append(f"{label:<{label_w}}" + "".join(f"{c:>{col_w}}" for c in cells))
            rows += ["", "Source: Yahoo Finance. N/A = data unavailable."]
            return "\n".join(rows)
        except Exception as yf_exc:
            logger.warning("yfinance comparison failed (%s); LLM fallback.", yf_exc)
        prompt = (
            f"Compare these stocks side by side: {', '.join(ticker_list)}.\n"
            "Provide a markdown comparison table with these metrics per company:\n"
            "current price, market cap, P/E (trailing), P/B, revenue (TTM), "
            "net income, gross margin, net margin, ROE, debt/equity, "
            "dividend yield, beta.\n"
            "After the table, write 2–3 sentences on which company looks strongest "
            "on valuation vs profitability.\n"
            "State clearly this is from training knowledge, not live data."
        )
        return _llm_fallback(prompt, f"comparison: {','.join(ticker_list)}")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# =============================================================================
#  Registry
# =============================================================================

def get_financial_tools() -> list[BaseTool]:
    """Return all six financial analysis tools."""
    return [
        StockQuoteTool(),
        CompanyInfoTool(),
        FinancialStatementsTool(),
        FinancialRatiosTool(),
        PriceHistoryTool(),
        StockComparisonTool(),
    ]