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
        r"return on (equity|assets)|margin|valuation|"
        r"overvalued|undervalued|cheap|expensive|"
        r"debt.to.equity|leverage|liquidity|current ratio|"
        r"profitab|dividend yield|payout)\b",
        re.IGNORECASE,
    )
    _HISTORY_PATTERNS = re.compile(
        r"\b(history|historical|performance|return|gain|loss|"
        r"how.{0,10} (done|performed|changed)|"
        r"since|over the (last|past)|ytd|year.to.date|"
        r"volatility|drawdown|trend|chart|cagr|annualised)\b",
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
        """
        # Compare requires multiple tickers — check first
        tickers = self._extract_tickers(query)
        if len(tickers) >= 2 and self._COMPARE_PATTERNS.search(query):
            return "compare"

        # Analysis is a superset — check before ratios / history
        if self._ANALYSIS_PATTERNS.search(query):
            return "analysis"

        # Statement queries contain very specific keywords
        if self._STATEMENT_PATTERNS.search(query):
            return "statements"

        if self._RATIOS_PATTERNS.search(query):
            return "ratios"

        if self._HISTORY_PATTERNS.search(query):
            return "history"

        if self._PROFILE_PATTERNS.search(query):
            return "profile"

        if self._QUOTE_PATTERNS.search(query):
            return "quote"

        # Multiple tickers without a compare keyword → comparison
        if len(tickers) >= 2:
            return "compare"

        # Single ticker with no other signal → quote is safest default
        if tickers:
            return "quote"

        return "unknown"

    # ── Ticker extraction ─────────────────────────────────────────────────────

    # Common ticker patterns: 1-5 uppercase letters, optionally with . or -
    # (e.g. BRK.B, BTC-USD)
    _TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-][A-Z]{1,5})?)\b")

    # Words that look like tickers but aren't
    _STOPWORDS = frozenset({
        "A", "I", "AT", "AS", "BE", "BY", "DO", "IN", "IS", "IT", "ME",
        "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "VS",
        "THE", "FOR", "AND", "ARE", "BUT", "CAN", "DID", "GET", "GOT",
        "HAS", "HAD", "HOW", "ITS", "LET", "MAY", "NOT", "NOW", "OFF",
        "OUT", "OWN", "SAY", "SHE", "SIX", "TEN", "TOO", "TWO", "USE",
        "VIA", "WAS", "WHO", "WHY", "YET", "YOU",
        # Finance-adjacent words that are not tickers
        "EPS", "P", "Q", "YTD", "ROE", "ROA", "ETF", "CEO", "CFO",
        "IPO", "GDP", "FED", "ECB",
    })

    @classmethod
    def _extract_tickers(cls, query: str) -> list[str]:
        """
        Extract stock ticker symbols from the query.

        Heuristics:
        - Must be 1–5 uppercase letters (with optional ./-suffix).
        - Must not be a known stopword.
        - Must follow a financial context signal or appear in a parenthetical.
        """
        # First: explicit parenthetical form → "Apple (AAPL)"
        parens = re.findall(r"\(([A-Z]{1,5}(?:[.\-][A-Z]{1,5})?)\)", query)

        # Second: all-caps words that look like tickers
        candidates = cls._TICKER_RE.findall(query.upper())
        filtered   = [
            t for t in candidates
            if t not in cls._STOPWORDS and len(t) >= 2
        ]

        # Merge: parenthetical tickers first (highest confidence), then others
        seen:    set[str]  = set()
        tickers: list[str] = []
        for t in parens + filtered:
            if t not in seen:
                seen.add(t)
                tickers.append(t)

        return tickers[:5]  # cap at 5

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
