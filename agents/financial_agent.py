"""
agents/financial_agent.py
──────────────────────────
Specialist agent for financial data retrieval and analysis.

Capabilities:
  • Real-time stock quotes (price, change, volume, market cap)
  • Company profiles (sector, business description, executives)
  • Financial statements (income statement, balance sheet, cash flow)
  • Valuation and quality ratios with qualitative grades
  • Historical price performance (return, CAGR, volatility, drawdown)
  • Multi-stock comparison tables
  • LLM-powered synthesis and interpretation of raw financial data

Routing strategy (avoids the full ReAct loop for well-understood queries):
  ┌──────────────────┬────────────────────────────────────────────────────┐
  │ Task             │ Tools invoked                                      │
  ├──────────────────┼────────────────────────────────────────────────────┤
  │ quote            │ stock_quote                                        │
  │ profile          │ company_info                                       │
  │ ratios           │ financial_ratios                                   │
  │ history          │ price_history                                      │
  │ statements       │ financial_statements                               │
  │ compare          │ stock_comparison                                   │
  │ analysis         │ stock_quote + financial_ratios + price_history     │
  │ unknown          │ ReAct executor (all 6 tools available)             │
  └──────────────────┴────────────────────────────────────────────────────┘

All tool results are passed to the LLM for synthesis, interpretation,
and plain-English explanation before being returned as the final answer.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory
from tools.financial_tools import get_financial_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert financial analyst and investment research assistant.

You have access to real-time market data, company financials, and analytical tools.
Your role is to provide accurate, objective, and insightful financial analysis.

Guidelines:
- Always present financial data clearly, with appropriate units ($, %, x).
- Explain financial ratios and metrics in plain English alongside raw numbers.
- Provide context: is a P/E of 25x high or low for this sector?
- Highlight both strengths and risks — balanced analysis, not promotion.
- Use comparisons (vs sector average, vs peers, vs historical) where possible.
- Note when data is unavailable rather than guessing.
- Include appropriate disclaimers: this is informational, not investment advice.
- For multi-ticker queries, structure the response with clear sections.

You are not a licensed financial advisor. Always remind users to do their own
due diligence and consult a professional before making investment decisions.
"""

# Disclaimer appended to every financial analysis response.
_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This analysis is for informational purposes only and does not "
    "constitute investment advice. Always conduct your own research and consult "
    "a qualified financial advisor before making investment decisions.*"
)


class FinancialAgent(BaseAgent):
    """
    Agent that retrieves and analyses financial data using yfinance-backed tools.

    Parameters
    ----------
    llm : BaseChatModel, optional
        LLM to use. Defaults to the configured model via get_llm().
    memory : AssistantMemory, optional
        Shared session memory.
    verbose : bool
        Whether to emit verbose ReAct chain logs.
    add_disclaimer : bool
        Append an investment-advice disclaimer to every response (default True).
    """

    def __init__(
            self,
            llm: Optional[BaseChatModel] = None,
            memory: Optional[AssistantMemory] = None,
            verbose: bool = False,
            add_disclaimer: bool = True,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=verbose)
        self._executor       = self._build_react_agent(system_prompt=_SYSTEM_PROMPT)
        self._add_disclaimer = add_disclaimer

    # ── BaseAgent interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "financial_agent"

    @property
    def description(self) -> str:
        return (
            "Financial data lookup and analysis: stock quotes, company profiles, "
            "income statements, balance sheets, valuation ratios, price history, "
            "and multi-stock comparisons. "
            "Use for questions about stock prices, company financials, investment "
            "metrics, or 'how has X performed?'."
        )

    def get_tools(self) -> list[BaseTool]:
        return get_financial_tools()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Handle a financial data or analysis request.

        Routing strategy
        ----------------
        1. Classify the task (quote / profile / ratios / history /
           statements / compare / analysis / unknown).
        2. Extract ticker symbol(s) and any parameters from the query.
        3. Call the most specific tool(s) directly.
        4. Pass the raw tool output(s) to the LLM for synthesis.
        5. Append the disclaimer and return.

        Falls back to the full ReAct executor when classification is
        ambiguous or any tool call fails completely.

        Parameters
        ----------
        query : str
            Natural-language financial question.
        ticker : str, optional
            Explicit ticker symbol override (bypasses extraction).

        Returns
        -------
        AgentResponse
        """
        self._logger.info("FinancialAgent handling: %s", query[:80])

        tools       = {t.name: t for t in self.get_tools()}
        tool_calls: list[tuple[str, str, str]] = []
        evidence:   list[str]                  = []

        try:
            task    = self._classify_task(query)
            tickers = kwargs.get("ticker") or self._extract_tickers(query)
            period  = self._extract_period(query)
            stmt    = self._extract_statement_type(query)

            # If no ticker was found via regex/name-map, ask the LLM to identify it
            if not tickers:
                resolved = self._resolve_ticker(query)
                if resolved:
                    tickers = [resolved]

            self._logger.debug(
                "task=%s  tickers=%s  period=%s  stmt=%s",
                task, tickers, period, stmt,
            )

            # ── Single-ticker primary tool calls ─────────────────────────────
            primary = tickers[0] if tickers else None

            if task == "quote" and primary:
                out = tools["stock_quote"]._run(ticker=primary)
                tool_calls.append(("stock_quote", primary, out))
                evidence.append(out)

            elif task == "profile" and primary:
                out = tools["company_info"]._run(ticker=primary)
                tool_calls.append(("company_info", primary, out))
                evidence.append(out)

            elif task == "ratios" and primary:
                out = tools["financial_ratios"]._run(ticker=primary)
                tool_calls.append(("financial_ratios", primary, out))
                evidence.append(out)

            elif task == "history" and primary:
                out = tools["price_history"]._run(
                    ticker=primary, period=period, interval="1d"
                )
                tool_calls.append(("price_history", f"{primary}/{period}", out))
                evidence.append(out)

            elif task == "statements" and primary:
                quarterly = self._wants_quarterly(query)
                out = tools["financial_statements"]._run(
                    ticker=primary,
                    statement=stmt,
                    quarterly=quarterly,
                )
                tool_calls.append(("financial_statements", f"{primary}/{stmt}", out))
                evidence.append(out)

            elif task == "compare" and len(tickers) >= 2:
                tickers_str = ",".join(tickers)
                out = tools["stock_comparison"]._run(tickers=tickers_str)
                tool_calls.append(("stock_comparison", tickers_str, out))
                evidence.append(out)

            elif task == "analysis" and primary:
                # Full analysis: quote + ratios + 1-year history
                for tool_name, call_kwargs in [
                    ("stock_quote",      {"ticker": primary}),
                    ("company_info",     {"ticker": primary}),
                    ("financial_ratios", {"ticker": primary}),
                    ("price_history",    {"ticker": primary, "period": period, "interval": "1d"}),
                ]:
                    out = tools[tool_name]._run(**call_kwargs)
                    call_key = f"{primary}" + (f"/{period}" if "history" in tool_name else "")
                    tool_calls.append((tool_name, call_key, out))
                    if "Could not" not in out:
                        evidence.append(out)

            # ── Fallback: ReAct executor ──────────────────────────────────────
            if not evidence:
                self._logger.info(
                    "No direct evidence gathered (task=%s, tickers=%s); "
                    "falling back to ReAct.", task, tickers,
                )
                return self._run_react(query)

            # ── Synthesise ────────────────────────────────────────────────────
            output = self._synthesise(query, task, evidence)
            if self._add_disclaimer:
                output += _DISCLAIMER

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
            )

        except Exception as exc:
            self._logger.error("FinancialAgent error: %s", exc, exc_info=True)
            try:
                return self._run_react(query)
            except Exception:
                return AgentResponse(
                    output=f"I encountered an error while retrieving financial data: {exc}",
                    agent_name=self.name,
                    error=str(exc),
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_react(self, query: str) -> AgentResponse:
        """Full ReAct executor fallback."""
        result     = self._executor.invoke({"input": query})
        output     = result.get("output", "No output generated.")
        tool_calls = self._extract_tool_calls(result.get("intermediate_steps", []))
        if self._add_disclaimer:
            output += _DISCLAIMER
        return AgentResponse(
            output=output,
            agent_name=self.name,
            tool_calls=tool_calls,
        )

    def _synthesise(self, query: str, task: str, evidence: list[str]) -> str:
        """Ask the LLM to interpret the raw tool data and produce an answer."""
        task_instructions = {
            "quote":      "Present the price and key market data clearly. Highlight notable movements.",
            "profile":    "Summarise the company's business, position, and key facts concisely.",
            "ratios":     ("Interpret the ratios. Are they attractive? How do they compare to "
                           "typical benchmarks? Highlight the most significant signals."),
            "history":    ("Analyse the price performance. Contextualise the return and volatility. "
                           "What does the trend suggest?"),
            "statements": ("Summarise the financial results. Highlight revenue trends, "
                           "profitability changes, and any notable items."),
            "compare":    ("Compare the companies. Which looks stronger on valuation? "
                           "On profitability? Summarise the key trade-offs."),
            "analysis":   ("Provide a comprehensive analysis: business overview, current valuation, "
                           "profitability quality, recent performance, and key risks. "
                           "Be balanced and specific."),
        }
        instruction = task_instructions.get(task, "Answer the question using the data provided.")

        combined = "\n\n---\n\n".join(evidence)
        prompt   = (
            f"User question: {query}\n\n"
            f"Financial data retrieved:\n"
            f"{'═' * 60}\n"
            f"{combined[:8000]}\n"
            f"{'═' * 60}\n\n"
            f"Task: {instruction}\n\n"
            f"Provide a clear, structured response. Use the actual numbers from "
            f"the data above. Do not add information not present in the data."
        )
        response = self._llm.invoke(prompt)
        return str(response.content).strip()

    # ── Task classification ───────────────────────────────────────────────────

    _QUOTE_PATTERNS = re.compile(
        r"\b(price|quote|trading at|share price|stock price|current price|"
        r"how much is|what is .{0,20} (trading|worth|at)|"
        r"market cap|bid|ask|today.s (price|close)|"
        r"premarket|after.?hours|intraday)\b",
        re.IGNORECASE,
    )
    _PROFILE_PATTERNS = re.compile(
        r"\b(what (does|is)|who (is|are)|tell me about|describe|"
        r"company profile|about .{0,20} (company|corp|inc)|"
        r"business (model|overview|description)|"
        r"sector|industry|employees|headquarters|founded)\b",
        re.IGNORECASE,
    )
    _RATIOS_PATTERNS = re.compile(
        r"\b(ratio|p/?e|p/?b|p/?s|peg|ev/?ebitda|roe|roa|"
        r"return on (equity|assets)|margin(s)?|valuation|"
        r"overvalued|undervalued|cheap|expensive|"
        r"debt.to.equity|leverage|liquidity|current ratio|"
        r"dividend yield|payout)|"
        r"\bprofitab|\bprofit margin|\bmetric",  # prefix matches — no trailing \b
        re.IGNORECASE,
    )
    _HISTORY_PATTERNS = re.compile(
        r"\b(history|historical|stock performance|price history|"
        r"price performance|performance (over|in|last|past)|"
        r"how.{0,15} (done|performed|changed)|"
        r"(over|in) the (last|past) \d|"
        r"since \d{4}|ytd|year.to.date|"
        r"(past|last) (week|month|year|quarter|decade)|"
        r"\d+ year(s)? (history|performance|return)|"
        r"\d+ (year|month|week)s? ago|"
        r"annuali[sz]ed return|total return|"
        r"volatility|drawdown|trend analysis|cagr)\b",
        re.IGNORECASE,
    )
    _STATEMENT_PATTERNS = re.compile(
        r"\b(income statement|profit.{0,5}loss|p&l|revenue|"
        r"balance sheet|assets|liabilities|equity|"
        r"cash flow|free cash|capex|ebitda|earnings|eps|"
        r"quarterly results|annual results|financial results|"
        r"(net |gross |operating )income|fiscal)\b",
        re.IGNORECASE,
    )
    _COMPARE_PATTERNS = re.compile(
        r"\b(compare|vs\.?|versus|against|better|which (is|has)|"
        r"side.by.side|head.to.head|peer|competitor|"
        r"(better|worse) than|ranking)\b",
        re.IGNORECASE,
    )
    _ANALYSIS_PATTERNS = re.compile(
        r"\b(analyse|analyze|analysis|should i (buy|sell|invest)|"
        r"investment thesis|worth investing|"
        r"fair value|intrinsic value|deep dive|overview of|"
        r"comprehensive|full report|research)\b",
        re.IGNORECASE,
    )

    def _classify_task(self, query: str) -> str:
        """
        Classify the query into one of:
        quote · profile · ratios · history · statements · compare · analysis · unknown

        Priority (highest → lowest):
          compare > analysis > ratios > statements > history > quote > profile > unknown
        Compare only fires when a compare keyword is present — multiple tickers alone
        are not sufficient to avoid false positives on ratio/history queries.
        """
        tickers = self._extract_tickers(query)

        # Analysis is the broadest — check before everything else
        if self._ANALYSIS_PATTERNS.search(query):
            return "analysis"

        # Compare: explicit compare keyword + multiple tickers
        if self._COMPARE_PATTERNS.search(query) and len(tickers) >= 2:
            return "compare"

        # History: performance-over-time language — check BEFORE ratios to
        # prevent "profitability" from capturing "how has X performed?" queries
        if self._HISTORY_PATTERNS.search(query):
            return "history"

        # Statements: specific financial statement line items
        # Exclude generic "equity" since that fires inside ratios too
        stmt_match = self._STATEMENT_PATTERNS.search(query)
        if stmt_match and stmt_match.group(0).lower() not in ("equity", "earnings"):
            return "statements"
        # Re-allow equity/earnings if stronger statement signals also present
        if stmt_match and any(
                kw in query.lower()
                for kw in ("income statement", "balance sheet", "cash flow",
                           "revenue", "net income", "ebitda", "quarterly results",
                           "annual results", "fiscal")
        ):
            return "statements"

        # Ratios: explicit ratio / profitability / valuation language
        if self._RATIOS_PATTERNS.search(query):
            return "ratios"

        # Quote: price / market-cap specific language — must come BEFORE profile
        # so "what is X's price?" doesn't get caught by the "what is" profile pattern
        if self._QUOTE_PATTERNS.search(query):
            return "quote"

        # Profile: general company description queries
        if self._PROFILE_PATTERNS.search(query):
            return "profile"

        # Multiple tickers with no other signal → comparison
        if len(tickers) >= 2:
            return "compare"

        # Single ticker with no other signal → quote
        if tickers:
            return "quote"

        return "unknown"

    # ── Ticker extraction ─────────────────────────────────────────────────────

    # Regex: explicit ticker symbols written in UPPERCASE (AAPL, BRK.B, BTC-USD)
    _TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-][A-Z]{1,5})?)\b")

    # Generic English words that are not ticker symbols
    _STOPWORDS = frozenset({
        "A", "I", "AT", "AS", "BE", "BY", "DO", "IN", "IS", "IT", "ME",
        "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "VS",
        "THE", "FOR", "AND", "ARE", "BUT", "CAN", "DID", "GET", "GOT",
        "HAS", "HAD", "HOW", "ITS", "LET", "MAY", "NOT", "NOW", "OFF",
        "OUT", "OWN", "SAY", "SHE", "SIX", "TEN", "TOO", "TWO", "USE",
        "VIA", "WAS", "WHO", "WHY", "YET", "YOU",
        "PRICE", "STOCK", "SHARE", "DATA", "SHOW", "LAST", "NEXT",
        "YEARS", "YEAR", "MONTH", "WEEK", "DAYS", "HOUR", "MINS",
        "HIGH", "LOWS", "OPEN", "CLOSE", "SELL", "HOLD",
        "TELL", "GIVE", "WHAT", "WHEN", "WHERE", "WILL", "BEEN",
        "WELL", "GOOD", "BEST", "FULL", "OVER",
        "FROM", "WITH", "THIS", "THAT", "THEM", "THEY", "THAN",
        "HAVE", "HERE", "INTO", "JUST", "LIKE", "MAKE", "MANY",
        "MORE", "MOST", "MUCH", "NEED", "ONLY", "SOME", "SUCH",
        "TAKE", "THEN", "WERE", "WOULD", "ABOUT", "AFTER",
        # Finance abbreviations
        "EPS", "P", "Q", "YTD", "ROE", "ROA", "ETF", "CEO", "CFO",
        "IPO", "GDP", "FED", "ECB", "PE", "PB", "PS", "PEG",
        "CAGR", "EBIT", "CAPEX", "OPEX", "FCF", "TTM", "LTM",
    })

    # Company-name → ticker map for the most commonly queried companies.
    # Covers plain English names that users type without knowing the ticker.
    _COMPANY_NAME_MAP: dict[str, str] = {
        # Big Tech
        "apple":        "AAPL",
        "microsoft":    "MSFT",
        "google":       "GOOGL",
        "alphabet":     "GOOGL",
        "amazon":       "AMZN",
        "meta":         "META",
        "facebook":     "META",
        "netflix":      "NFLX",
        "nvidia":       "NVDA",
        "tesla":        "TSLA",
        "intel":        "INTC",
        "amd":          "AMD",
        "qualcomm":     "QCOM",
        "broadcom":     "AVGO",
        "salesforce":   "CRM",
        "oracle":       "ORCL",
        "ibm":          "IBM",
        "cisco":        "CSCO",
        "adobe":        "ADBE",
        "paypal":       "PYPL",
        "uber":         "UBER",
        "lyft":         "LYFT",
        "twitter":      "TWTR",
        "x corp":       "TWTR",
        "snap":         "SNAP",
        "spotify":      "SPOT",
        "airbnb":       "ABNB",
        "palantir":     "PLTR",
        "shopify":      "SHOP",
        "square":       "SQ",
        "block":        "SQ",
        "coinbase":     "COIN",
        "robinhood":    "HOOD",
        # Finance
        "jpmorgan":     "JPM",
        "jp morgan":    "JPM",
        "goldman sachs":"GS",
        "goldman":      "GS",
        "morgan stanley":"MS",
        "bank of america":"BAC",
        "bofa":         "BAC",
        "wells fargo":  "WFC",
        "citigroup":    "C",
        "citi":         "C",
        "visa":         "V",
        "mastercard":   "MA",
        "american express":"AXP",
        "amex":         "AXP",
        "blackrock":    "BLK",
        "berkshire":    "BRK-B",
        # Healthcare
        "johnson":      "JNJ",
        "johnson & johnson":"JNJ",
        "pfizer":       "PFE",
        "moderna":      "MRNA",
        "unitedhealth": "UNH",
        "abbvie":       "ABBV",
        "eli lilly":    "LLY",
        "lilly":        "LLY",
        "merck":        "MRK",
        "novartis":     "NVS",
        # Consumer / Retail
        "walmart":      "WMT",
        "costco":       "COST",
        "target":       "TGT",
        "home depot":   "HD",
        "nike":         "NKE",
        "starbucks":    "SBUX",
        "mcdonalds":    "MCD",
        "mcdonald's":   "MCD",
        "coca cola":    "KO",
        "coca-cola":    "KO",
        "pepsi":        "PEP",
        "pepsico":      "PEP",
        "procter":      "PG",
        "procter & gamble":"PG",
        # Auto
        "ford":         "F",
        "general motors":"GM",
        "gm":           "GM",
        "toyota":       "TM",
        "volkswagen":   "VWAGY",
        "porsche":      "POAHY",
        # Energy
        "exxon":        "XOM",
        "exxonmobil":   "XOM",
        "chevron":      "CVX",
        "shell":        "SHEL",
        # Indices / Crypto
        "bitcoin":      "BTC-USD",
        "btc":          "BTC-USD",
        "ethereum":     "ETH-USD",
        "eth":          "ETH-USD",
        "sp500":        "^GSPC",
        "s&p 500":      "^GSPC",
        "s&p":          "^GSPC",
        "dow jones":    "^DJI",
        "dow":          "^DJI",
        "nasdaq":       "^IXIC",
    }

    @classmethod
    def _extract_tickers(cls, query: str) -> list[str]:
        """
        Extract stock ticker symbols from a natural-language query.

        Resolution order (highest confidence first):
          1. Explicit parenthetical form: "Apple (AAPL)" → ["AAPL"]
          2. Company-name lookup:  "How has Apple performed?" → ["AAPL"]
          3. Bare UPPERCASE symbols: "TSLA income statement" → ["TSLA"]
          4. LLM extraction: called by _resolve_ticker() when no match found

        This handles both "AAPL" and "Apple" style references.
        """
        seen:    set[str]  = set()
        tickers: list[str] = []

        def _add(t: str) -> None:
            t = t.strip().upper()
            if t and t not in seen:
                seen.add(t)
                tickers.append(t)

        # 1. Parenthetical: "Apple (AAPL)"
        for t in re.findall(r"\(([A-Z]{1,5}(?:[.\-][A-Z]{1,5})?)\)", query):
            _add(t)

        # 2. Company name lookup (case-insensitive, longest match first)
        q_lower = query.lower()
        for name in sorted(cls._COMPANY_NAME_MAP, key=len, reverse=True):
            if re.search(r"\b" + re.escape(name) + r"\b", q_lower):
                _add(cls._COMPANY_NAME_MAP[name])

        # 3. Bare UPPERCASE ticker symbols already in the query
        for t in cls._TICKER_RE.findall(query):   # query is already mixed case
            if t.upper() not in cls._STOPWORDS and len(t) >= 2:
                _add(t)

        return tickers[:5]

    @classmethod
    def _resolve_ticker(cls, query: str) -> Optional[str]:
        """
        Return a single best-guess ticker for the query.
        Tries _extract_tickers first, then asks the LLM as a last resort.
        Never returns None — falls back to passing the raw query to tools
        which will use their own LLM fallback.
        """
        tickers = cls._extract_tickers(query)
        if tickers:
            return tickers[0]

        # LLM extraction: ask the model to identify the company/ticker
        try:
            from core.llm_manager import get_llm
            llm    = get_llm()
            prompt = (
                f"Extract the stock ticker symbol from this query: \"{query}\"\n\n"
                "Rules:\n"
                "- Return ONLY the ticker symbol (e.g. AAPL, MSFT, TSLA, BTC-USD).\n"
                "- If you can identify the company, return its primary US exchange ticker.\n"
                "- If you cannot identify any specific company, return UNKNOWN.\n"
                "Ticker:"
            )
            result = llm.invoke(prompt)
            ticker = str(result.content).strip().upper().split()[0]
            if ticker and ticker != "UNKNOWN" and len(ticker) <= 7:
                logger.info("LLM resolved ticker '%s' from query: %s", ticker, query[:60])
                return ticker
        except Exception as exc:
            logger.debug("LLM ticker resolution failed: %s", exc)

        return None

    # ── Period extraction ─────────────────────────────────────────────────────

    _PERIOD_MAP = {
        "day":   "1d",  "1 day":  "1d",
        "week":  "5d",  "1 week": "5d",  "5 day": "5d",
        "month": "1mo", "1 month": "1mo",
        "3 month": "3mo", "quarter": "3mo",
        "6 month": "6mo", "half year": "6mo",
        "year":  "1y",  "1 year": "1y",  "12 month": "1y",
        "2 year": "2y", "3 year": "3y",
        "5 year": "5y", "decade": "10y", "10 year": "10y",
        "ytd": "ytd", "year to date": "ytd",
        "max": "max",  "all time": "max", "all-time": "max",
    }

    @classmethod
    def _extract_period(cls, query: str) -> str:
        """Return the best matching yfinance period string, defaulting to '1y'."""
        q = query.lower()
        for phrase, period in sorted(cls._PERIOD_MAP.items(), key=lambda x: -len(x[0])):
            if phrase in q:
                return period
        return "1y"

    # ── Statement type ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_statement_type(query: str) -> str:
        """Return 'income', 'balance', or 'cashflow'."""
        q = query.lower()
        if any(k in q for k in ("balance sheet", "assets", "liabilities", "equity")):
            return "balance"
        if any(k in q for k in ("cash flow", "free cash", "capex", "operating cash")):
            return "cashflow"
        return "income"  # default

    @staticmethod
    def _wants_quarterly(query: str) -> bool:
        """Return True if the query asks for quarterly data."""
        return bool(re.search(
            r"\b(quarter|quarterly|q1|q2|q3|q4|qtr)\b",
            query, re.IGNORECASE,
        ))