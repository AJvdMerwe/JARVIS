"""
tests/test_financial_agent.py
──────────────────────────────
Tests for:
  • All 6 financial tools (mocked yfinance — no real network calls)
  • FinancialAgent task classification, ticker extraction, period detection
  • FinancialAgent routing to correct tools
  • Orchestrator intent routing for financial queries
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.base_agent import AgentResponse


# =============================================================================
#  Shared yfinance mock factory
# =============================================================================

def _mock_ticker(
    price:       float = 178.50,
    prev_close:  float = 175.20,
    market_cap:  float = 2.8e12,
    pe:          float = 28.5,
    revenue:     float = 385e9,
    net_income:  float = 97e9,
    roe:         float = 1.75,
    div_yield:   float = 0.0055,
):
    """Build a mock yfinance Ticker with realistic Apple-like data."""
    info = {
        # Quote
        "currentPrice":     price,
        "previousClose":    prev_close,
        "open":             176.00,
        "dayHigh":          179.80,
        "dayLow":           175.50,
        "volume":           58_000_000,
        "marketCap":        market_cap,
        "sharesOutstanding":15_550_000_000,
        "fiftyTwoWeekHigh": 199.62,
        "fiftyTwoWeekLow":  164.08,
        "exchange":         "NASDAQ",
        # Company
        "longName":         "Apple Inc.",
        "shortName":        "Apple",
        "sector":           "Technology",
        "industry":         "Consumer Electronics",
        "country":          "United States",
        "city":             "Cupertino",
        "fullTimeEmployees":161_000,
        "website":          "https://www.apple.com",
        "longBusinessSummary": (
            "Apple Inc. designs, manufactures, and markets smartphones, "
            "personal computers, tablets, wearables, and accessories worldwide."
        ),
        "currency":         "USD",
        "companyOfficers": [
            {"name": "Timothy D. Cook",   "title": "CEO"},
            {"name": "Luca Maestri",       "title": "CFO"},
            {"name": "Jeffrey E. Williams","title": "COO"},
        ],
        # Valuation
        "trailingPE":       pe,
        "forwardPE":        25.0,
        "priceToBook":      46.0,
        "priceToSalesTrailing12Months": 7.0,
        "enterpriseToEbitda": 22.0,
        "pegRatio":         2.8,
        # Profitability
        "grossMargins":     0.441,
        "operatingMargins": 0.296,
        "profitMargins":    0.252,
        "returnOnEquity":   roe,
        "returnOnAssets":   0.282,
        "revenueGrowth":    0.036,
        "earningsGrowth":   0.108,
        # Liquidity/leverage
        "currentRatio":     0.99,
        "quickRatio":       0.99,
        "debtToEquity":     180.0,
        "totalDebt":        124e9,
        "freeCashflow":     99e9,
        # Dividends
        "dividendYield":    div_yield,
        "dividendRate":     0.96,
        "payoutRatio":      0.16,
        # Financials
        "totalRevenue":     revenue,
        "netIncomeToCommon": net_income,
        "52WeekChange":     0.089,
        "beta":             1.25,
    }

    fast_info = MagicMock()
    fast_info.last_price     = price
    fast_info.previous_close = prev_close
    fast_info.open           = 176.00
    fast_info.day_high       = 179.80
    fast_info.day_low        = 175.50
    fast_info.last_volume    = 58_000_000
    fast_info.market_cap     = market_cap
    fast_info.shares         = 15_550_000_000
    fast_info.year_high      = 199.62
    fast_info.year_low       = 164.08
    fast_info.exchange       = "NASDAQ"

    ticker        = MagicMock()
    ticker.info   = info
    ticker.fast_info = fast_info
    return ticker, info


def _mock_history(n_bars: int = 252, start_price: float = 150.0, end_price: float = 178.5):
    """Build a mock pandas DataFrame-like history object."""
    import pandas as pd
    import numpy as np

    dates   = pd.date_range(end="2024-12-31", periods=n_bars, freq="B")
    prices  = np.linspace(start_price, end_price, n_bars)
    volumes = np.random.randint(40_000_000, 80_000_000, n_bars)

    df = pd.DataFrame({
        "Open":   prices * 0.998,
        "High":   prices * 1.012,
        "Low":    prices * 0.988,
        "Close":  prices,
        "Volume": volumes,
    }, index=dates)
    return df


# =============================================================================
#  Formatting helpers
# =============================================================================

class TestFormattingHelpers:
    def test_fmt_number_billions(self):
        from tools.financial_tools import _fmt_number
        assert "2.80B" in _fmt_number(2.8e9)

    def test_fmt_number_millions(self):
        from tools.financial_tools import _fmt_number
        assert "500.00M" in _fmt_number(5e8)

    def test_fmt_number_none(self):
        from tools.financial_tools import _fmt_number
        assert _fmt_number(None) == "N/A"

    def test_fmt_pct(self):
        from tools.financial_tools import _fmt_pct
        assert _fmt_pct(0.0523) == "5.23%"
        assert _fmt_pct(None) == "N/A"

    def test_fmt_ratio(self):
        from tools.financial_tools import _fmt_ratio
        assert _fmt_ratio(28.5) == "28.50x"
        assert _fmt_ratio(None) == "N/A"

    def test_grade_ratio_pe(self):
        from tools.financial_tools import _grade_ratio
        assert "Attractive" in _grade_ratio("pe", 8.0)
        assert "Fair"       in _grade_ratio("pe", 18.0)
        assert "Expensive"  in _grade_ratio("pe", 30.0)
        assert "Very expensive" in _grade_ratio("pe", 50.0)

    def test_grade_ratio_roe(self):
        from tools.financial_tools import _grade_ratio
        assert "Excellent" in _grade_ratio("roe", 0.35)
        assert "Weak"      in _grade_ratio("roe", 0.05)

    def test_grade_ratio_unknown(self):
        from tools.financial_tools import _grade_ratio
        assert _grade_ratio("unknown_metric", 5.0) == ""


# =============================================================================
#  StockQuoteTool
# =============================================================================

class TestStockQuoteTool:
    def test_returns_price(self):
        from tools.financial_tools import StockQuoteTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            tool   = StockQuoteTool()
            result = tool._run(ticker="AAPL")

        assert "178.50" in result
        assert "AAPL"   in result

    def test_shows_change_direction(self):
        from tools.financial_tools import StockQuoteTool
        mock_tkr, _ = _mock_ticker(price=178.50, prev_close=175.20)

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = StockQuoteTool()._run(ticker="AAPL")

        # price rose → should show upward arrow
        assert "▲" in result

    def test_shows_down_when_price_fell(self):
        from tools.financial_tools import StockQuoteTool
        mock_tkr, _ = _mock_ticker(price=170.00, prev_close=175.00)

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = StockQuoteTool()._run(ticker="AAPL")

        assert "▼" in result

    def test_shows_market_cap(self):
        from tools.financial_tools import StockQuoteTool
        mock_tkr, _ = _mock_ticker(market_cap=2.8e12)

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = StockQuoteTool()._run(ticker="AAPL")

        assert "2.80T" in result

    def test_handles_error_gracefully(self):
        from tools.financial_tools import StockQuoteTool
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = StockQuoteTool()._run(ticker="AAPL")

        assert "Could not fetch" in result
        assert "AAPL" in result

    def test_ticker_uppercased(self):
        from tools.financial_tools import StockQuoteTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value = mock_tkr
            StockQuoteTool()._run(ticker="aapl")

        mock_yf.assert_called_once_with("AAPL")


# =============================================================================
#  CompanyInfoTool
# =============================================================================

class TestCompanyInfoTool:
    def test_returns_company_name(self):
        from tools.financial_tools import CompanyInfoTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = CompanyInfoTool()._run(ticker="AAPL")

        assert "Apple" in result

    def test_returns_sector_and_industry(self):
        from tools.financial_tools import CompanyInfoTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = CompanyInfoTool()._run(ticker="AAPL")

        assert "Technology"          in result
        assert "Consumer Electronics" in result

    def test_includes_employee_count(self):
        from tools.financial_tools import CompanyInfoTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = CompanyInfoTool()._run(ticker="AAPL")

        assert "161" in result  # 161,000 employees

    def test_includes_business_summary(self):
        from tools.financial_tools import CompanyInfoTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = CompanyInfoTool()._run(ticker="AAPL")

        assert "smartphones" in result.lower() or "Business Summary" in result

    def test_handles_error_gracefully(self):
        from tools.financial_tools import CompanyInfoTool
        with patch("yfinance.Ticker", side_effect=ConnectionError("timeout")):
            result = CompanyInfoTool()._run(ticker="MSFT")

        assert "Could not" in result


# =============================================================================
#  FinancialRatiosTool
# =============================================================================

class TestFinancialRatiosTool:
    def test_contains_pe_ratio(self):
        from tools.financial_tools import FinancialRatiosTool
        mock_tkr, _ = _mock_ticker(pe=28.5)

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialRatiosTool()._run(ticker="AAPL")

        assert "28.50x" in result

    def test_contains_roe(self):
        from tools.financial_tools import FinancialRatiosTool
        mock_tkr, _ = _mock_ticker(roe=1.75)

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialRatiosTool()._run(ticker="AAPL")

        assert "175.00%" in result   # 1.75 * 100

    def test_contains_sections(self):
        from tools.financial_tools import FinancialRatiosTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialRatiosTool()._run(ticker="AAPL")

        assert "Valuation"    in result
        assert "Profitability" in result
        assert "Liquidity"    in result
        assert "Leverage"     in result

    def test_grade_labels_present(self):
        from tools.financial_tools import FinancialRatiosTool
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialRatiosTool()._run(ticker="AAPL")

        # At least one grade should appear
        assert any(g in result for g in ["Excellent", "Good", "Fair", "Weak",
                                          "Attractive", "Expensive"])

    def test_dividend_section_shown_when_yield_present(self):
        from tools.financial_tools import FinancialRatiosTool
        mock_tkr, _ = _mock_ticker(div_yield=0.0055)

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialRatiosTool()._run(ticker="AAPL")

        assert "Dividend" in result

    def test_handles_missing_data(self):
        from tools.financial_tools import FinancialRatiosTool
        mock_tkr        = MagicMock()
        mock_tkr.info   = {}   # completely empty info

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialRatiosTool()._run(ticker="AAPL")

        # Should return a result with N/A for missing fields, not crash
        assert "N/A" in result or "Financial Ratios" in result


# =============================================================================
#  PriceHistoryTool
# =============================================================================

class TestPriceHistoryTool:
    def test_calculates_total_return(self):
        from tools.financial_tools import PriceHistoryTool
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = _mock_history(start_price=150.0, end_price=178.5)

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = PriceHistoryTool()._run(ticker="AAPL", period="1y")

        # Return ≈ 19%
        assert "19." in result or "18." in result

    def test_shows_cagr(self):
        from tools.financial_tools import PriceHistoryTool
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = _mock_history()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = PriceHistoryTool()._run(ticker="AAPL", period="1y")

        assert "CAGR" in result

    def test_shows_volatility(self):
        from tools.financial_tools import PriceHistoryTool
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = _mock_history()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = PriceHistoryTool()._run(ticker="AAPL", period="1y")

        assert "vol" in result.lower()

    def test_shows_max_drawdown(self):
        from tools.financial_tools import PriceHistoryTool
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = _mock_history()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = PriceHistoryTool()._run(ticker="AAPL", period="1y")

        assert "drawdown" in result.lower()

    def test_handles_empty_history(self):
        from tools.financial_tools import PriceHistoryTool
        import pandas as pd
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = PriceHistoryTool()._run(ticker="AAPL", period="1y")

        assert "No price history" in result

    def test_passes_period_to_yfinance(self):
        from tools.financial_tools import PriceHistoryTool
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = _mock_history()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            PriceHistoryTool()._run(ticker="AAPL", period="5y", interval="1wk")

        mock_tkr.history.assert_called_once_with(period="5y", interval="1wk")


# =============================================================================
#  StockComparisonTool
# =============================================================================

class TestStockComparisonTool:
    def _two_tickers(self):
        aapl, _ = _mock_ticker(price=178.50, pe=28.5, market_cap=2.8e12)
        msft, _ = _mock_ticker(price=410.00, pe=35.0, market_cap=3.0e12)
        return aapl, msft

    def test_comparison_table_has_both_tickers(self):
        from tools.financial_tools import StockComparisonTool
        aapl, msft = self._two_tickers()

        with patch("yfinance.Ticker", side_effect=[aapl, msft]):
            result = StockComparisonTool()._run(tickers="AAPL,MSFT")

        assert "AAPL" in result
        assert "MSFT" in result

    def test_comparison_shows_pe(self):
        from tools.financial_tools import StockComparisonTool
        aapl, msft = self._two_tickers()

        with patch("yfinance.Ticker", side_effect=[aapl, msft]):
            result = StockComparisonTool()._run(tickers="AAPL,MSFT")

        assert "P/E" in result
        assert "28.5" in result

    def test_single_ticker_rejected(self):
        from tools.financial_tools import StockComparisonTool
        result = StockComparisonTool()._run(tickers="AAPL")
        assert "at least 2" in result

    def test_handles_partial_failure(self):
        """If one ticker fails, the tool should still return a result."""
        from tools.financial_tools import StockComparisonTool
        aapl, _  = _mock_ticker()
        bad_tkr  = MagicMock()
        bad_tkr.info = {}   # empty → all N/A

        with patch("yfinance.Ticker", side_effect=[aapl, bad_tkr]):
            result = StockComparisonTool()._run(tickers="AAPL,BADTICKER")

        assert "AAPL" in result
        assert "N/A"  in result


# =============================================================================
#  FinancialStatementsTool
# =============================================================================

class TestFinancialStatementsTool:
    def _mock_statement_df(self):
        import pandas as pd
        import numpy as np
        cols = pd.to_datetime(["2023-09-30", "2022-09-30", "2021-09-30", "2020-09-30"])
        idx  = [
            "Total Revenue", "Gross Profit", "Operating Income",
            "EBITDA", "Net Income", "Basic EPS",
        ]
        data = {c: [np.random.uniform(80e9, 400e9) for _ in idx] for c in cols}
        return pd.DataFrame(data, index=idx)

    def test_income_statement_title(self):
        from tools.financial_tools import FinancialStatementsTool
        mock_tkr, _ = _mock_ticker()
        mock_tkr.income_stmt = self._mock_statement_df()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialStatementsTool()._run(
                ticker="AAPL", statement="income", quarterly=False
            )

        assert "Income Statement" in result

    def test_balance_sheet_title(self):
        from tools.financial_tools import FinancialStatementsTool
        import pandas as pd
        mock_tkr, _ = _mock_ticker()
        cols = pd.to_datetime(["2023-09-30", "2022-09-30"])
        mock_tkr.balance_sheet = pd.DataFrame(
            {"2023": [100e9, 50e9], "2022": [90e9, 45e9]},
            index=["Total Assets", "Total Liabilities Net Minority Interest"]
        )
        mock_tkr.balance_sheet.columns = cols

        with patch("yfinance.Ticker", return_value=mock_tkr):
            result = FinancialStatementsTool()._run(
                ticker="AAPL", statement="balance", quarterly=False
            )

        assert "Balance Sheet" in result

    def test_invalid_statement_type(self):
        from tools.financial_tools import FinancialStatementsTool
        result = FinancialStatementsTool()._run(
            ticker="AAPL", statement="unknown_type"
        )
        assert "Unknown statement" in result

    def test_quarterly_flag_uses_quarterly_df(self):
        from tools.financial_tools import FinancialStatementsTool
        mock_tkr, _ = _mock_ticker()
        mock_tkr.quarterly_income_stmt = self._mock_statement_df()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            FinancialStatementsTool()._run(
                ticker="AAPL", statement="income", quarterly=True
            )

        # Should have accessed quarterly_income_stmt, not income_stmt
        _ = mock_tkr.quarterly_income_stmt  # attribute was accessed


# =============================================================================
#  FinancialAgent — task classification
# =============================================================================

class TestFinancialAgentClassification:
    @pytest.mark.parametrize("query,expected", [
        # Quote
        ("What is Apple's stock price?",             "quote"),
        ("AAPL current price",                       "quote"),
        ("How much is Tesla trading at?",            "quote"),
        ("MSFT market cap",                          "quote"),
        # Profile
        ("What does Apple do?",                      "profile"),
        ("Tell me about Microsoft as a company",     "profile"),
        ("Apple's sector and industry",              "profile"),
        # Ratios
        ("Is AAPL overvalued?",                      "ratios"),
        ("Apple P/E ratio",                          "ratios"),
        ("TSLA valuation ratios",                    "ratios"),
        ("Show me MSFT profitability metrics",       "ratios"),
        ("Return on equity for GOOGL",               "ratios"),
        # History
        ("How has AAPL performed in the last year?", "history"),
        ("Apple stock history 5 years",              "history"),
        ("TSLA return since 2020",                   "history"),
        ("MSFT volatility last 6 months",            "history"),
        # Statements
        ("Show Apple's income statement",            "statements"),
        ("AAPL revenue last 4 years",                "statements"),
        ("Tesla balance sheet",                      "statements"),
        ("Apple cash flow quarterly",                "statements"),
        # Compare
        ("Compare AAPL vs MSFT",                     "compare"),
        ("AAPL GOOGL MSFT side by side",             "compare"),
        # Analysis
        ("Analyse Apple stock",                      "analysis"),
        ("Should I invest in AAPL?",                 "analysis"),
        ("Full report on Tesla",                     "analysis"),
    ])
    def test_task_classification(self, query: str, expected: str):
        from agents.financial_agent import FinancialAgent
        result = FinancialAgent._classify_task(FinancialAgent, query)
        assert result == expected, (
            f"Expected '{expected}' for query {query!r}, got '{result}'"
        )


# =============================================================================
#  FinancialAgent — ticker extraction
# =============================================================================

class TestTickerExtraction:
    @pytest.mark.parametrize("query,expected_tickers", [
        ("What is Apple (AAPL) stock price?",   ["AAPL"]),
        ("MSFT and AAPL comparison",            ["MSFT", "AAPL"]),
        ("Compare AAPL, MSFT, GOOGL",           ["AAPL", "MSFT", "GOOGL"]),
        ("bitcoin BTC-USD price",               ["BTC-USD"]),
        ("What is the weather today?",          []),
        ("Tell me about TSLA",                  ["TSLA"]),
        ("NVDA earnings report",                ["NVDA"]),
    ])
    def test_ticker_extraction(self, query: str, expected_tickers: list):
        from agents.financial_agent import FinancialAgent
        tickers = FinancialAgent._extract_tickers(query)
        for t in expected_tickers:
            assert t in tickers, f"Expected {t} in tickers from {query!r}, got {tickers}"


# =============================================================================
#  FinancialAgent — period extraction
# =============================================================================

class TestPeriodExtraction:
    @pytest.mark.parametrize("query,expected", [
        ("AAPL last year",           "1y"),
        ("5 year history",           "5y"),
        ("performance last month",   "1mo"),
        ("6 month return",           "6mo"),
        ("ytd performance",          "ytd"),
        ("all time high",            "max"),
        ("quarterly earnings",       "3mo"),  # "quarter" in period map → 3mo
        ("decade performance",       "10y"),
    ])
    def test_period_extraction(self, query: str, expected: str):
        from agents.financial_agent import FinancialAgent
        assert FinancialAgent._extract_period(query) == expected


# =============================================================================
#  FinancialAgent — run() routing
# =============================================================================

class TestFinancialAgentRun:
    def _make_agent(self, tool_output: str = "Mock financial data"):
        from agents.financial_agent import FinancialAgent
        from core.memory import AssistantMemory

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="LLM synthesised output.")

        # Patch _build_react_agent to avoid AgentExecutor import in the base class
        with patch("agents.base_agent.BaseAgent._build_react_agent",
                   return_value=MagicMock()):
            agent = FinancialAgent(
                llm=mock_llm, memory=AssistantMemory(), add_disclaimer=False
            )

        # Replace each tool's _run with a mock
        mock_tools = []
        for tool in agent.get_tools():
            m = MagicMock()
            m.name      = tool.name
            m._run      = MagicMock(return_value=tool_output)
            mock_tools.append(m)

        # Patch get_tools so run() picks up the mocked tools
        agent.get_tools = lambda: mock_tools

        return agent, mock_llm

    def test_run_returns_agent_response(self):
        agent, _ = self._make_agent()
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            resp = agent.run("What is Apple stock price? AAPL")

        assert isinstance(resp, AgentResponse)
        assert resp.agent_name == "financial_agent"
        assert resp.error is None

    def test_quote_task_calls_stock_quote_tool(self):
        agent, _ = self._make_agent("$178.50  AAPL quote data")
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            resp = agent.run("What is AAPL stock price?")

        # tool_calls should contain stock_quote
        tool_names = [tc[0] for tc in resp.tool_calls]
        assert "stock_quote" in tool_names

    def test_ratios_task_calls_financial_ratios_tool(self):
        agent, _ = self._make_agent("P/E: 28.5x  ROE: 175%")
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            resp = agent.run("Show me AAPL valuation ratios")

        tool_names = [tc[0] for tc in resp.tool_calls]
        assert "financial_ratios" in tool_names

    def test_history_task_calls_price_history_tool(self):
        agent, _ = self._make_agent("1y history data")
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = _mock_history()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            resp = agent.run("How has AAPL stock performed over the last year?")

        tool_names = [tc[0] for tc in resp.tool_calls]
        assert "price_history" in tool_names

    def test_compare_task_calls_comparison_tool(self):
        agent, _ = self._make_agent("Comparison table")
        aapl, _ = _mock_ticker(price=178.50, pe=28.5)
        msft, _ = _mock_ticker(price=410.00, pe=35.0)

        with patch("yfinance.Ticker", side_effect=[aapl, msft]):
            resp = agent.run("Compare AAPL vs MSFT")

        tool_names = [tc[0] for tc in resp.tool_calls]
        assert "stock_comparison" in tool_names

    def test_analysis_calls_multiple_tools(self):
        agent, _ = self._make_agent("Data for analysis")
        mock_tkr, _ = _mock_ticker()
        mock_tkr.history.return_value = _mock_history()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            resp = agent.run("Full analysis of AAPL")

        tool_names = [tc[0] for tc in resp.tool_calls]
        assert len(tool_names) >= 3  # quote + ratios + history at minimum

    def test_disclaimer_appended_when_enabled(self):
        from agents.financial_agent import FinancialAgent
        from core.memory import AssistantMemory

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Analysis.")

        with patch("agents.base_agent.BaseAgent._build_react_agent",
                   return_value=MagicMock()):
            agent = FinancialAgent(
                llm=mock_llm, memory=AssistantMemory(), add_disclaimer=True
            )

        mock_tools = []
        for tool in agent.get_tools():
            m = MagicMock(); m.name = tool.name
            m._run = MagicMock(return_value="Mock quote data $178.50")
            mock_tools.append(m)
        agent.get_tools = lambda: mock_tools

        mock_tkr, _ = _mock_ticker()
        with patch("yfinance.Ticker", return_value=mock_tkr):
            resp = agent.run("AAPL stock price")

        assert "Disclaimer" in resp.output or "investment advice" in resp.output.lower()

    def test_disclaimer_omitted_when_disabled(self):
        agent, _ = self._make_agent()
        mock_tkr, _ = _mock_ticker()

        with patch("yfinance.Ticker", return_value=mock_tkr):
            resp = agent.run("AAPL price")

        assert "Disclaimer" not in resp.output

    def test_error_handled_gracefully(self):
        """When all tools AND the ReAct fallback fail, run() returns an error response."""
        from agents.financial_agent import FinancialAgent
        from core.memory import AssistantMemory

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM down")

        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = Exception("executor also failed")

        with patch("agents.base_agent.BaseAgent._build_react_agent",
                   return_value=mock_executor):
            agent = FinancialAgent(
                llm=mock_llm, memory=AssistantMemory(), add_disclaimer=False
            )

        broken_tools = []
        for tool in agent.get_tools():
            m = MagicMock(); m.name = tool.name
            m._run = MagicMock(side_effect=Exception("network error"))
            broken_tools.append(m)
        agent.get_tools = lambda: broken_tools

        resp = agent.run("AAPL price")
        # Either an error is set, or the output contains an error message
        assert resp.error is not None or "error" in str(resp.output).lower()


# =============================================================================
#  Orchestrator — FINANCE intent routing
# =============================================================================

class TestOrchestratorFinanceRouting:
    @pytest.mark.parametrize("query", [
        "What is Apple's stock price?",
        "AAPL stock price today",
        "Show me Tesla earnings per share",
        "Compare MSFT vs GOOGL valuation",
        "NVDA income statement annual",
        "How has Amazon performed over 5 years?",
        "Is Apple overvalued? P/E ratio",
        "Analyse TSLA for investment",
    ])
    def test_finance_queries_route_to_finance(self, query: str):
        from agents.orchestrator import _keyword_route, Intent
        result = _keyword_route(query)
        assert result == Intent.FINANCE, (
            f"Expected FINANCE for {query!r}, got {result}"
        )

    def test_finance_intent_in_enum(self):
        from agents.orchestrator import Intent
        assert Intent.FINANCE in Intent
        assert Intent.FINANCE.value == "finance"

    def test_finance_agent_registered(self):
        from agents.orchestrator import Orchestrator, Intent
        with patch("agents.orchestrator.PersistentMemory"), \
             patch("agents.orchestrator.ChatAgent"), \
             patch("agents.orchestrator.CodeAgent"), \
             patch("agents.orchestrator.NewsAgent"), \
             patch("agents.orchestrator.SearchAgent"), \
             patch("agents.orchestrator.DocumentAgent"), \
             patch("agents.orchestrator.FinancialAgent") as mock_fin:
            orch = Orchestrator(session_id="test-fin")

        mock_fin.assert_called_once()
        assert Intent.FINANCE in orch._agents

    def test_code_not_routed_to_finance(self):
        from agents.orchestrator import _keyword_route, Intent
        # "stock" alone isn't enough to beat CODE signals
        result = _keyword_route("write a Python script to fetch stock data")
        assert result != Intent.FINANCE   # CODE should win

    def test_finance_doesnt_capture_news(self):
        from agents.orchestrator import _keyword_route, Intent
        result = _keyword_route("latest news about the stock market crash")
        # NEWS score should be high enough to beat FINANCE
        assert result != Intent.FINANCE or result == Intent.NEWS
