"""
agents/orchestrator.py
───────────────────────
LangGraph-based multi-agent orchestrator — the single entry point for all
user interactions.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           Orchestrator                                  │
  │                                                                         │
  │  User Input                                                             │
  │      │                                                                  │
  │      ▼                                                                  │
  │  Intent Router  ──► keyword heuristics (fast, no LLM)                  │
  │      │          └──► LLM classifier    (fallback for ambiguous queries) │
  │      │                                                                  │
  │      ▼                                                                  │
  │  RAG Pre-check  ──► vector-store scan (fast, < 200 ms)                 │
  │      │          ├──► SUFFICIENT ──► post-process & return              │
  │      │          └──► INSUFFICIENT / SKIP ──► Primary Agent             │
  │      │                                                                  │
  │      ▼                                                                  │
  │  Primary Agent (chosen by intent)                                       │
  │      │                                                                  │
  │      ▼                                                                  │
  │  Response Quality Check                                                 │
  │      │                                                                  │
  │      ├── SUFFICIENT ──► Post-process & return                          │
  │      │                                                                  │
  │      └── INSUFFICIENT ──► Fallback Chain                               │
  │               │                                                         │
  │               ├── Fallback Agent 1 → quality check → ...               │
  │               ├── Fallback Agent 2 → quality check → ...               │
  │               └── Final synthesis (LLM merges best evidence)           │
  │                                                                         │
  │  Post-processing (memory · episodic · summariser · tracing)            │
  └─────────────────────────────────────────────────────────────────────────┘

Fallback chains (per intent, highest priority first):
  DOCUMENT  →  SEARCH  →  CHAT
  SEARCH    →  DOCUMENT  →  CHAT
  NEWS      →  SEARCH  →  CHAT
  FINANCE   →  SEARCH  →  CHAT
  CODE      →  SEARCH  →  CHAT
  CHAT      →  SEARCH
  (UNKNOWN  →  CHAT — no further fallback needed)

Quality gates (fast-path, no LLM cost):
  • Error flag set on the response
  • Output shorter than MIN_SUFFICIENT_LENGTH characters
  • Output contains known failure phrases ("I don't know", "no information",
    "cannot find", etc.)
  • No tool calls when tool usage was expected

Optional LLM quality judge:
  When enable_llm_quality_check=True the orchestrator asks the LLM after
  every agent call: "Does this response adequately answer the question?"
  This is more expensive but catches subtle non-answers.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Optional, TypedDict

from .base_agent import AgentResponse, BaseAgent
from .chat_agent import ChatAgent
from .code_agent import CodeAgent
from .data_analysis_agent import DataAnalysisAgent
from .deep_research_agent import DeepResearchAgent
from .document_agent import DocumentAgent
from .financial_agent import FinancialAgent
from .news_agent import NewsAgent
from .rag_precheck import rag_precheck
from .search_agent import SearchAgent
from .writing_agent import WritingAssistantAgent
from core.memory import PersistentMemory
from core.llm_manager import get_llm
from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
#  Intent enum
# =============================================================================

class Intent(str, Enum):
    CHAT     = "chat"      # General conversation, creative, explanatory
    CODE     = "code"      # Write / debug / execute code
    NEWS     = "news"      # Fetch / summarise news
    SEARCH   = "search"    # Web search, Wikipedia, calculations
    DOCUMENT = "document"  # KB / document Q&A
    FINANCE  = "finance"   # Stock quotes, financials, market analysis
    RESEARCH = "research"  # Multi-turn deep research with reasoning model
    DATA     = "data"      # CSV / Excel analysis, pandas, charts
    WRITING  = "writing"   # Long-form writing, articles, reports, DOCX export
    UNKNOWN  = "unknown"   # Unclassifiable — falls back to CHAT


# =============================================================================
#  LangGraph state
# =============================================================================

class OrchestratorState(TypedDict, total=False):
    """
    Shared state dict threaded through every node in the LangGraph pipeline.

    Fields
    ------
    query:    str
        Original user query string.
    intent:   str
        Detected or forced intent value (Intent enum .value).
    response: AgentResponse
        Produced by the chosen agent node.
    history:  list[dict[str, str]]
        Snapshot of recent conversation turns (for logging / debugging).
    metadata: dict[str, Any]
        Arbitrary extra context — user_id, doc_title, request_id, etc.
    error:    str
        Set if a node raises an exception; empty string otherwise.
    """
    query:    str
    intent:   str
    response: AgentResponse
    history:  list[dict[str, str]]
    metadata: dict[str, Any]
    error:    str


# =============================================================================
#  Quality constants
# =============================================================================

# Minimum characters in an agent output for it to be considered "sufficient".
# Shorter responses almost always indicate a failure or non-answer.
MIN_SUFFICIENT_LENGTH: int = 40

# Phrases that indicate the agent could not answer the question.
# These are checked case-insensitively against the output.
_FAILURE_PHRASES: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "no information",
    "no relevant",
    "cannot find",
    "could not find",
    "unable to find",
    "not found in",
    "not available",
    "i cannot answer",
    "i can't answer",
    "no results found",
    "nothing found",
    "no data",
    "insufficient information",
    "i encountered an error",
    "i encountered an issue",
    "failed to retrieve",
    "failed to fetch",
    "failed to find",
)


# =============================================================================
#  Fallback chains
# =============================================================================

# Per-intent ordered list of fallback intents to try when the primary agent
# fails to answer.  The primary intent itself is NOT included here.
_FALLBACK_CHAINS: dict[Intent, list[Intent]] = {
    Intent.DOCUMENT: [Intent.SEARCH, Intent.CHAT],
    Intent.SEARCH:   [Intent.DOCUMENT, Intent.CHAT],
    Intent.NEWS:     [Intent.SEARCH, Intent.CHAT],
    Intent.FINANCE:  [Intent.SEARCH, Intent.CHAT],
    Intent.CODE:     [Intent.SEARCH, Intent.CHAT],
    Intent.RESEARCH: [Intent.SEARCH, Intent.CHAT],
    Intent.DATA:     [Intent.CODE, Intent.CHAT],
    Intent.WRITING:  [Intent.CHAT],
    Intent.CHAT:     [Intent.SEARCH],
    Intent.UNKNOWN:  [Intent.CHAT],
}


# =============================================================================
#  Keyword-based fast router — Stage 1
# =============================================================================

_CHAT_KEYWORDS = re.compile(
    r"\b(hi|hello|hey|how are you|what do you think|tell me about yourself|"
    r"who are you|let'?s talk|can you help|i need help|chat|convers|talk to me|"
    r"what is your|do you know|explain|describe|what does|how does|why does|"
    r"compare|difference between|pros and cons|give me an example|"
    r"brainstorm|write (a|an|me) (poem|story|essay|letter|haiku|song|joke|"
    r"paragraph|blog|speech|narrative|tale|summary)|"
    r"tell me (a|an)|create (a|an) (poem|story|essay)|draft (a|an)|"
    r"poem|haiku|creative writing|opinion|advice|recommend|suggest|"
    r"what would you|translate|rewrite|improve|paraphrase|"
    r"summarise|summarize|like i'?m (five|5)|eli5)\b",
    re.IGNORECASE,
)

_CODE_KEYWORDS = re.compile(
    r"\b(code|program|function|class|algorithm|debug|bug|error|"
    r"script|python|javascript|typescript|java|c\+\+|rust|sql|html|css|"
    r"implement|refactor|execute|compile|unit test|"
    r"data structure|recursion|regex|api|endpoint|"
    r"write (a |an )?(function|class|script|program|algorithm|module|"
    r"method|loop|test|decorator|interface|enum|query|snippet|macro)|"
    r"review (this |the )?(code|function|class|script|program))\b",
    re.IGNORECASE,
)

_NEWS_KEYWORDS = re.compile(
    r"\b(news|headlines?|breaking|current events|what'?s happening|briefing|"
    r"rss|feed|journalist|coverage|press|broadcast|"
    r"latest (news|headlines?|stories|developments|updates)|"
    r"news (about|on|from)|top (stories|headlines?)|daily news)\b",
    re.IGNORECASE,
)

_DOCUMENT_KEYWORDS = re.compile(
    r"\b(document|pdf|docx|excel|spreadsheet|ppt|powerpoint|file|"
    r"upload|ingest|knowledge base|my docs?|this report|the report|"
    r"the paper|the document|the file|from the|according to|"
    r"what does the|summarise the|summarize the|in the doc)\b",
    re.IGNORECASE,
)

_FINANCE_KEYWORDS = re.compile(
    r"\b(stock|ticker|share(s)?|equity|nasdaq|nyse|s&p 500|"
    r"(stock|share) price|market cap|trading at|"
    r"p/?e( ratio)?|p/?b( ratio)?|dividend|earnings per share|eps|"
    r"income statement|balance sheet|cash flow|"
    r"revenue|net income|ebitda|free cash flow|capex|"
    r"return on equity|roe|roa|debt.to.equity|"
    r"bull|bear market|portfolio|"
    r"ipo|valuation|overvalued|undervalued|"
    r"financial (results|statements|ratios|analysis|performance)|"
    r"quarterly (results|earnings|report)|annual report|"
    r"stock (performance|history|return|gain)|"
    r"compare .{0,30} (stock|share|ticker)|"
    r"yfinance|yahoo finance|"
    r"(price|market) (history|performance|trend)|"
    r"volatility|drawdown|cagr|annuali[sz]ed return|"
    r"52.week|year.to.date stock|"
    r"(earnings|revenue|profit) (growth|margin)|"
    r"analyst (rating|target|recommendation)|"
    r"how has .{1,20} (stock|share|performed|done)|"
    r"(stock|share|investment) (analysis|report|thesis)|"
    r"invest(ment|ing|or) in .{1,20} (stock|share)|"
    r"(performed|performance) (over|in|last|past) .{1,20} year|"
    r"compare .{1,30} (stock|valuation|ratio|p.e|market cap)|"
    r"analy[sz]e .{1,30} (stock|share|invest|valuation)|"
    r"(invest(ment|ing) in|buy|sell) .{1,20} (stock|share|ticker)|"
    r"[A-Z]{2,5} (vs\.?|versus) [A-Z]{2,5} .{0,20}(valuation|ratio|stock|p.e|comparison))\b",
    re.IGNORECASE,
)

_FINANCE_PHRASES = re.compile(
    r"analy[sz]e .{1,30} (stock|for invest|valuat)|"
    r"invest(ment|ing|or)? in .{1,25} (stock|share)|"
    r"(how has|how did) .{1,20} (stock|share|perform)|"
    r"[A-Z]{2,5} vs\.? [A-Z]{2,5} .{0,25}(valuat|ratio|p/?e|margin|compar)|"
    r"(buy|sell|hold|short) .{1,20} stock|"
    r"[A-Z]{2,5} (stock|share) (price|performance|analysis|valuation)",
    re.IGNORECASE,
)


_RESEARCH_KEYWORDS = re.compile(
    r"\b(deep research|deep dive|comprehensive report|in.?depth analysis|"
    r"thoroughly (research|analyse|analyze|investigate|examine)|"
    r"detailed (report|study|analysis|overview|examination)|"
    r"literature (review|survey)|state of the art|"
    r"research (the|into|on|about)|"
    r"(what does|what do) (the )?(research|literature|science|studies) say|"
    r"systematic review|meta.?analysis|"
    r"investigate (the|thoroughly)|"
    r"full (report|analysis|study)|"
    r"everything (about|on)|tell me everything|"
    r"academic overview|scholarly (review|analysis))\b",
    re.IGNORECASE,
)

_DATA_KEYWORDS = re.compile(
    r"\b(csv|dataframe|dataset|spreadsheet|excel|tsv|"
    r"pandas|numpy|matplotlib|seaborn|plotly|"
    r"correlation|distribution|regression|clustering|"
    r"pivot.?table|group.?by|aggregate|"
    r"outlier|anomal|trend|forecast|"
    r"visuali[sz]e|visuali[sz]ation|plot|chart|graph|histogram|scatter|"
    r"analyse (the |this |my )?data|analyze (the |this |my )?data|"
    r"data analysis|statistical|statistics|"
    r"load (the |this |my )?(file|data|csv|sheet)|"
    r"what does (the |this |my )?data (show|say|mean)|"
    r"average|mean|median|mode|standard deviation|variance|"
    r"top (\d+ )?by|sum (of|by)|count (of|by))\b",
    re.IGNORECASE,
)

_WRITING_KEYWORDS = re.compile(
    r"\b(write (an?|the|a) (article|essay|report|blog|post|letter|email|document|"
    r"piece|section|paragraph|draft|intro|conclusion|outline)|"
    r"draft (an?|the|a) |"
    r"compose (an?|the|a) |"
    r"help me write|"
    r"create (an?|the|a) (article|essay|report|blog|document|letter|email)|"
    r"(write|create|generate) (an? )?(blog post|technical doc|white ?paper|case study)|"
    r"outline (for|of|about)|"
    r"long.?form|"
    r"proofread|edit (my|this|the) (draft|essay|article|report|email|letter)|"
    r"improve (my|this|the) writing|"
    r"export (to |as )?(docx|word|markdown)|"
    r"(formal|academic|professional) (writing|document|report)|"
    r"letter to|email to)\b",
    re.IGNORECASE,
)


def _keyword_route(query: str) -> Optional[Intent]:
    """Stage-1 intent detector. Returns None when ambiguous."""
    code_score     = len(_CODE_KEYWORDS.findall(query))
    news_score     = len(_NEWS_KEYWORDS.findall(query))
    doc_score      = len(_DOCUMENT_KEYWORDS.findall(query))
    chat_score     = len(_CHAT_KEYWORDS.findall(query))
    finance_score  = len(_FINANCE_KEYWORDS.findall(query))
    finance_score += len(_FINANCE_PHRASES.findall(query))
    research_score = len(_RESEARCH_KEYWORDS.findall(query))
    data_score     = len(_DATA_KEYWORDS.findall(query))
    writing_score  = len(_WRITING_KEYWORDS.findall(query))

    specialist_scores: dict[Intent, int] = {
        Intent.CODE:     code_score,
        Intent.NEWS:     news_score,
        Intent.DOCUMENT: doc_score,
        Intent.FINANCE:  finance_score,
        Intent.RESEARCH: research_score,
        Intent.DATA:     data_score,
        Intent.WRITING:  writing_score,
    }
    best_specialist, best_score = max(specialist_scores.items(), key=lambda x: x[1])
    second_best = sorted(specialist_scores.values(), reverse=True)[1]

    # Tie-break: DATA beats DOCUMENT when the query mentions a data file type
    _DATA_FILE_RE = re.compile(r"\b(csv|tsv|excel|\.xlsx?|spreadsheet|dataframe)\b", re.IGNORECASE)
    if (specialist_scores.get(Intent.DATA, 0) == specialist_scores.get(Intent.DOCUMENT, 0)
            and specialist_scores.get(Intent.DATA, 0) >= 1
            and _DATA_FILE_RE.search(query)):
        specialist_scores[Intent.DATA] += 1   # prefer DATA when data file clearly mentioned

    # Tie-break: WRITING beats CHAT when writing-specific creation verbs are present
    _WRITE_VERB_RE = re.compile(
        r"\b(write|draft|compose|create|generate|produce)\s+(an?|the|a)\s+"
        r"(article|essay|report|blog|post|letter|email|document|piece|outline)\b",
        re.IGNORECASE,
    )
    if (specialist_scores.get(Intent.WRITING, 0) >= 1
            and chat_score >= 1
            and specialist_scores[Intent.WRITING] >= chat_score
            and _WRITE_VERB_RE.search(query)):
        specialist_scores[Intent.WRITING] += 1   # prefer WRITING over CHAT
        chat_score = 0   # neutralise chat score for this query type

    best_specialist, best_score = max(specialist_scores.items(), key=lambda x: x[1])
    sorted_scores = sorted(specialist_scores.values(), reverse=True)
    second_best   = sorted_scores[1] if len(sorted_scores) > 1 else 0

    if best_score >= 1 and best_score > second_best and chat_score < best_score:
        return best_specialist
    if chat_score >= 1 and (best_score == 0 or chat_score >= best_score):
        return Intent.CHAT
    if best_score >= 2 and best_score > chat_score:
        return best_specialist
    if best_score == 0 and chat_score == 0:
        return Intent.SEARCH
    return None


def _llm_route(query: str) -> Intent:
    """Stage-2 classifier. Falls back to Intent.CHAT on failure."""
    llm = get_llm()
    prompt = (
        "Classify the following user query into exactly one category.\n\n"
        "Categories:\n"
        "- chat: general conversation, creative writing, brainstorming, opinions,\n"
        "  advice, explanations, follow-up questions, or anything conversational\n"
        "- code: writing, debugging, reviewing, or executing code / programs\n"
        "- news: fetching or summarising news articles and current events\n"
        "- document: questions about uploaded files or the knowledge base\n"
        "- search: factual questions that need real-time web lookup or Wikipedia\n\n"
        f"Query: {query}\n\n"
        "Respond with ONLY the category name (lowercase, no punctuation)."
    )
    try:
        result = llm.invoke(prompt)
        label  = str(result.content).strip().lower()
        return Intent(label) if label in Intent._value2member_map_ else Intent.CHAT
    except Exception:
        return Intent.CHAT


# =============================================================================
#  Response quality evaluation
# =============================================================================

def _is_sufficient_response(response: AgentResponse, query: str) -> bool:
    """
    Fast-path quality gate — no LLM call required.

    Returns True when the response is likely a genuine answer to the query.
    Returns False when any of these failure signals are detected:

    1. ``response.error`` is set.
    2. Output is shorter than MIN_SUFFICIENT_LENGTH characters.
    3. Output contains a known failure phrase (case-insensitive).

    Parameters
    ----------
    response : AgentResponse
        The candidate response to evaluate.
    query : str
        The original user query (reserved for future heuristics).

    Returns
    -------
    bool
    """
    # 1. Explicit error
    if response.error:
        logger.debug(
            "Quality: FAIL — agent '%s' returned error: %s",
            response.agent_name, response.error,
        )
        return False

    text = (response.output or "").strip()

    # 2. Too short to be a real answer
    if len(text) < MIN_SUFFICIENT_LENGTH:
        logger.debug(
            "Quality: FAIL — output too short (%d chars < %d) from '%s'.",
            len(text), MIN_SUFFICIENT_LENGTH, response.agent_name,
        )
        return False

    # 3. Known failure phrases
    text_lower = text.lower()
    for phrase in _FAILURE_PHRASES:
        if phrase in text_lower:
            logger.debug(
                "Quality: FAIL — failure phrase %r detected in '%s' output.",
                phrase, response.agent_name,
            )
            return False

    logger.debug(
        "Quality: PASS — '%s' response (%d chars).",
        response.agent_name, len(text),
    )
    return True


def _llm_quality_check(query: str, response: AgentResponse) -> bool:
    """
    Optional LLM-powered quality judge.

    Asks the configured LLM whether the response adequately answers the
    question.  More accurate than the fast-path gate but incurs a latency
    and token cost penalty.

    Parameters
    ----------
    query : str
        The original user question.
    response : AgentResponse
        The candidate response to evaluate.

    Returns
    -------
    bool
        True when the LLM judges the response as sufficient.
        Defaults to True on any error (fail-open to avoid unnecessary retries).
    """
    try:
        llm    = get_llm()
        prompt = (
            "You are a response quality evaluator.\n\n"
            f"User question: {query}\n\n"
            f"Agent response:\n{response.output[:1500]}\n\n"
            "Does this response adequately and directly answer the user's question?\n"
            "Consider it SUFFICIENT if it provides relevant, useful information.\n"
            "Consider it INSUFFICIENT only if it:\n"
            "  - Explicitly says it doesn't know or cannot answer\n"
            "  - Contains only an error message\n"
            "  - Is completely off-topic\n"
            "  - Provides no information relevant to the question\n\n"
            "Reply with exactly one word: SUFFICIENT or INSUFFICIENT"
        )
        result = llm.invoke(prompt)
        verdict = str(result.content).strip().upper()
        is_ok   = verdict == "SUFFICIENT"
        logger.debug(
            "LLM quality judge: %s for agent '%s'.",
            verdict, response.agent_name,
        )
        return is_ok
    except Exception as exc:
        logger.debug("LLM quality check failed (%s); defaulting to PASS.", exc)
        return True   # fail-open: don't retry on checker failure


def _synthesise_from_attempts(
    query:    str,
    attempts: list[tuple[str, AgentResponse]],
) -> AgentResponse:
    """
    Ask the LLM to synthesise the best possible answer from all partial
    attempts when every agent in the fallback chain failed to answer
    sufficiently on its own.

    The synthesiser is given all non-empty outputs as evidence and
    instructed to combine them into a coherent final response.

    Parameters
    ----------
    query : str
        The original user question.
    attempts : list[tuple[str, AgentResponse]]
        Each element is (agent_name, response) in the order they were tried.

    Returns
    -------
    AgentResponse
        Synthesised response.  Falls back to the best individual attempt
        (longest non-error output) if the LLM itself fails.
    """
    # Filter to responses that have at least some content
    useful = [
        (name, resp) for name, resp in attempts
        if resp.output and not resp.error and len(resp.output.strip()) > 10
    ]

    if not useful:
        # All agents errored — return the last attempt as-is
        last_name, last_resp = attempts[-1]
        return last_resp

    evidence_blocks = "\n\n---\n\n".join(
        f"[{name}]:\n{resp.output[:800]}"
        for name, resp in useful
    )

    try:
        llm    = get_llm()
        prompt = (
            f"The user asked: {query}\n\n"
            "The following agents were consulted but none gave a complete answer "
            "on their own. Synthesise the best possible response using all the "
            "information below:\n\n"
            f"{evidence_blocks}\n\n"
            "Instructions:\n"
            "- Directly answer the user's question.\n"
            "- Combine relevant information from the sources above.\n"
            "- If the information is insufficient, say so clearly.\n"
            "- Do not mention the internal agents or this synthesis process.\n\n"
            "Answer:"
        )
        result = llm.invoke(prompt)
        output = str(result.content).strip()
        # Collect all tool_calls and references from every attempt
        all_tool_calls = [tc for _, r in attempts for tc in r.tool_calls]
        all_references = list({ref for _, r in attempts for ref in r.references})
        logger.info(
            "Synthesised answer from %d attempt(s) for query: %s…",
            len(useful), query[:60],
        )
        return AgentResponse(
            output=output,
            agent_name="synthesised",
            tool_calls=all_tool_calls,
            references=all_references,
        )
    except Exception as exc:
        logger.error("Synthesis failed: %s; returning best single attempt.", exc)
        # Return the longest useful response as the best fallback
        best_name, best_resp = max(useful, key=lambda x: len(x[1].output))
        return best_resp


# =============================================================================
#  Orchestrator
# =============================================================================

class Orchestrator:
    """
    Routes user queries to the most appropriate specialist agent, evaluates
    response quality, and iterates through a per-intent fallback chain until
    a sufficient answer is found.

    Fallback behaviour
    ------------------
    After each agent call, the response is checked against quality gates:

    Fast gates (no LLM cost):
      • ``response.error`` is set
      • Output shorter than MIN_SUFFICIENT_LENGTH chars
      • Output contains a known failure phrase ("I don't know", etc.)

    Optional LLM gate (``enable_llm_quality_check=True``):
      • Asks the LLM to judge whether the response answers the question

    If the response fails, the next agent in the intent's fallback chain is
    tried.  After all agents in the chain are exhausted, the LLM synthesises
    the best possible answer from all collected evidence.

    Parameters
    ----------
    session_id : str
        Identifier for persistent memory, tracing, and episodic recall.
    enable_episodic : bool
        Store and recall facts from past sessions (EpisodicMemory).
    enable_summariser : bool
        Compress old conversation turns automatically.
    enable_plugins : bool
        Auto-discover agents/tools from plugins/ at startup.
    enable_scheduler : bool
        Start the background TaskScheduler.
    enable_llm_quality_check : bool
        Use the LLM to judge response quality (more accurate, slower).
        Default False — fast pattern-matching gates are used instead.
    enable_rag_precheck : bool
        Scan the vector store after intent detection and before calling any
        agent.  When a sufficiently relevant answer is found in the KB the
        primary agent is never invoked (faster, cheaper).  Default True.
    rag_similarity_threshold : float
        Minimum cosine similarity for a chunk to be considered relevant
        during the pre-check.  Range 0–1; default 0.55.
    rag_k : int
        Maximum number of chunks to retrieve during the pre-check.
        Default 4.
    max_fallback_attempts : int
        Maximum number of fallback agents to try after the primary fails.
        Default 2.  Set to 0 to disable fallback entirely.
    summarise_after : int
        Total message count that triggers history compression.
    recent_k : int
        Verbatim turns to keep after compression.
    """

    def __init__(
        self,
        session_id:                str   = "default",
        enable_episodic:           bool  = True,   # on by default for long-term memory
        enable_summariser:         bool  = False,
        enable_plugins:            bool  = True,
        enable_scheduler:          bool  = False,
        enable_llm_quality_check:  bool  = False,
        enable_rag_precheck:       bool  = True,
        rag_similarity_threshold:  float = 0.55,
        rag_k:                     int   = 4,
        max_fallback_attempts:     int   = 2,
        summarise_after:           int   = 40,
        recent_k:                  int   = 10,
    ) -> None:
        self._session_id               = session_id
        self._enable_llm_quality_check = enable_llm_quality_check
        self._enable_rag_precheck      = enable_rag_precheck
        self._rag_similarity_threshold = rag_similarity_threshold
        self._rag_k                    = rag_k
        self._max_fallback_attempts    = max_fallback_attempts

        self._memory = PersistentMemory(session_id=session_id)

        self._agents: dict[Intent, BaseAgent] = {
            Intent.CHAT:     ChatAgent(memory=self._memory),
            Intent.CODE:     CodeAgent(memory=self._memory),
            Intent.NEWS:     NewsAgent(memory=self._memory),
            Intent.SEARCH:   SearchAgent(memory=self._memory),
            Intent.DOCUMENT: DocumentAgent(memory=self._memory),
            Intent.FINANCE:  FinancialAgent(memory=self._memory),
            Intent.RESEARCH: DeepResearchAgent(memory=self._memory),
            Intent.DATA:     DataAnalysisAgent(memory=self._memory),
            Intent.WRITING:  WritingAssistantAgent(memory=self._memory),
        }

        self._summariser = None
        if enable_summariser:
            try:
                from core.summariser import ConversationSummariser
                self._summariser = ConversationSummariser(
                    summarise_after=summarise_after,
                    recent_k=recent_k,
                )
            except Exception as exc:
                logger.warning("ConversationSummariser init failed: %s", exc)

        self._episodic = None
        if enable_episodic:
            try:
                from core.long_term_memory import EpisodicMemory
                self._episodic = EpisodicMemory()
            except Exception as exc:
                logger.warning("EpisodicMemory init failed: %s", exc)

        if enable_plugins:
            self._load_plugins()

        if enable_scheduler:
            self.start_scheduler()

        self._user_id = "default"   # updated by run() per-call

        logger.info(
            "Orchestrator ready | session='%s' | agents=%d | "
            "fallback=%s (max=%d) | llm_quality=%s | rag_precheck=%s (thresh=%.2f)",
            session_id, len(self._agents),
            "on", self._max_fallback_attempts,
            self._enable_llm_quality_check,
            self._enable_rag_precheck,
            self._rag_similarity_threshold,
        )

    # -------------------------------------------------------------------------
    #  Core run  (with fallback loop)
    # -------------------------------------------------------------------------

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Route a query to the best agent, evaluate quality, and iterate
        through fallback agents until a sufficient answer is produced.

        Execution flow
        --------------
        1.  Detect intent (keyword → LLM fallback).
        2.  RAG pre-check: scan vector store for a direct answer.
            If sufficient answer found → post-process and return immediately.
        3.  Call the primary agent.
        4.  Evaluate response quality (fast gates + optional LLM judge).
        5.  If INSUFFICIENT and fallback is enabled:
              a. Try each agent in the intent's fallback chain.
              b. Re-evaluate quality after each attempt.
              c. Return the first SUFFICIENT response.
        6.  If all attempts fail: synthesise from all evidence via LLM.
        7.  Post-process: persist memory, extract episodic facts,
            maybe summarise, record trace.

        Parameters
        ----------
        query : str
            The user's input.
        intent : str | Intent, optional
            Force a specific intent, bypassing routing.
        user_id : str, optional
            UserPreferences for this user.
        **kwargs
            Forwarded to each agent's run().

        Returns
        -------
        AgentResponse
        """
        forced  = kwargs.pop("intent",  None)
        user_id = kwargs.get("user_id", None)
        if user_id:
            self._user_id = user_id

        intent = (
            Intent(forced) if isinstance(forced, str)
            else forced
            if forced
            else self._route(query)
        )

        logger.info(
            "session='%s' intent=%s query='%s…'",
            self._session_id, intent.value, query[:60],
        )

        trace_obj = self._open_trace(query)

        if getattr(self, "_episodic", None):
            self._inject_episodic_context(query, kwargs)
        if user_id:
            kwargs.setdefault("user_id", user_id)


        # ── Schedule detection ────────────────────────────────────────────────
        # If the query contains a scheduling intent, register a user task
        # and return a confirmation instead of running the normal agent.
        sched_response = self._maybe_schedule_task(query, intent, user_id or self._user_id)
        if sched_response is not None:
            self._close_trace(trace_obj, sched_response)
            self._memory.save_context(query, sched_response.output)
            return sched_response

        # ── RAG pre-check ─────────────────────────────────────────────────────
        # Quick vector-store scan: if the KB already contains a good answer
        # return it immediately without invoking any agent.
        if self._enable_rag_precheck:
            try:
                rag_response = rag_precheck(
                    query,
                    intent.value,
                    similarity_threshold=self._rag_similarity_threshold,
                    k=self._rag_k,
                )
            except Exception as exc:
                logger.warning(
                    "RAG pre-check raised an unexpected error: %s — continuing.", exc
                )
                rag_response = None

            if rag_response is not None:
                self._close_trace(trace_obj, rag_response)
                self._memory.save_context(query, rag_response.output)
                if getattr(self, "_episodic", None) and rag_response.output:
                    self._store_episodic_facts(query, rag_response.output)
                if getattr(self, "_summariser", None):
                    try:
                        self._summariser.maybe_summarise(self._memory)
                    except Exception as exc:
                        logger.debug("Summariser error: %s", exc)
                return rag_response

        # ── Primary attempt ───────────────────────────────────────────────────
        primary_agent = self._agents.get(intent) or self._agents[Intent.CHAT]
        response      = self._call_agent(primary_agent, query, **kwargs)
        attempts: list[tuple[str, AgentResponse]] = [(primary_agent.name, response)]

        # ── Fallback loop ─────────────────────────────────────────────────────
        if self._max_fallback_attempts > 0 and not self._is_sufficient(query, response):
            response = self._run_fallback_chain(
                query, intent, attempts, max_attempts=self._max_fallback_attempts,
                **kwargs,
            )

        # ── Post-processing ───────────────────────────────────────────────────
        self._close_trace(trace_obj, response)
        self._memory.save_context(query, response.output)

        if getattr(self, "_episodic", None) and response.output and not response.error:
            self._store_episodic_facts(query, response.output)

        # Profile extraction — lightweight heuristics + async LLM extraction
        if response.output and not response.error:
            self._update_user_profile(
                query, response.output, response.agent_name, user_id or self._user_id
            )

        if getattr(self, "_summariser", None):
            try:
                self._summariser.maybe_summarise(self._memory)
            except Exception as exc:
                logger.debug("Summariser error: %s", exc)

        return response

    # -------------------------------------------------------------------------
    #  Fallback chain implementation
    # -------------------------------------------------------------------------

    def _run_fallback_chain(
        self,
        query:        str,
        primary:      Intent,
        attempts:     list[tuple[str, AgentResponse]],
        max_attempts: int,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Iterate through the fallback chain for the given primary intent until
        a sufficient response is found or the chain is exhausted.

        Parameters
        ----------
        query : str
            The original user query.
        primary : Intent
            The primary intent that failed.
        attempts : list
            Already-collected (agent_name, response) pairs — mutated in place.
        max_attempts : int
            Maximum additional agents to try.

        Returns
        -------
        AgentResponse
            The first sufficient response found, or a synthesis of all
            attempts if none is sufficient.
        """
        fallback_intents = _FALLBACK_CHAINS.get(primary, [Intent.CHAT])
        tried_intents    = {primary}
        fallbacks_run    = 0

        for fb_intent in fallback_intents:
            if fallbacks_run >= max_attempts:
                logger.info(
                    "Fallback limit (%d) reached for query: %s…",
                    max_attempts, query[:60],
                )
                break

            if fb_intent in tried_intents:
                continue
            tried_intents.add(fb_intent)

            fb_agent = self._agents.get(fb_intent)
            if not fb_agent:
                continue

            logger.info(
                "Fallback %d/%d: trying '%s' after '%s' failed for query: %s…",
                fallbacks_run + 1, max_attempts,
                fb_intent.value, primary.value, query[:60],
            )

            fb_response = self._call_agent(fb_agent, query, **kwargs)
            attempts.append((fb_agent.name, fb_response))
            fallbacks_run += 1

            if self._is_sufficient(query, fb_response):
                logger.info(
                    "Fallback '%s' produced a sufficient response.",
                    fb_intent.value,
                )
                return fb_response

            logger.debug(
                "Fallback '%s' response also insufficient; continuing chain.",
                fb_intent.value,
            )

        # All agents in the chain failed — synthesise from collected evidence
        logger.info(
            "All %d agent(s) in fallback chain insufficient; synthesising answer.",
            len(attempts),
        )
        return _synthesise_from_attempts(query, attempts)

    # -------------------------------------------------------------------------
    #  Quality evaluation
    # -------------------------------------------------------------------------

    def _is_sufficient(self, query: str, response: AgentResponse) -> bool:
        """
        Determine whether a response adequately answers the query.

        Applies the fast-path pattern gates first.  If those pass and
        ``enable_llm_quality_check`` is True, also asks the LLM.

        Parameters
        ----------
        query : str
        response : AgentResponse

        Returns
        -------
        bool
        """
        if not _is_sufficient_response(response, query):
            return False
        if self._enable_llm_quality_check:
            return _llm_quality_check(query, response)
        return True

    # -------------------------------------------------------------------------
    #  Agent call wrapper
    # -------------------------------------------------------------------------

    @staticmethod
    def _call_agent(
        agent: BaseAgent,
        query: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Call an agent and catch all exceptions, converting them into a
        failed AgentResponse so the fallback loop can continue.
        """
        try:
            return agent.run(query, **kwargs)
        except Exception as exc:
            logger.error(
                "Agent '%s' raised an exception: %s", agent.name, exc
            )
            return AgentResponse(
                output=f"Agent error: {exc}",
                agent_name=agent.name,
                error=str(exc),
            )

    # -------------------------------------------------------------------------
    #  Routing helpers
    # -------------------------------------------------------------------------

    def route_only(self, query: str) -> Intent:
        """Classify a query without running any agent."""
        return self._route(query)

    # -------------------------------------------------------------------------
    #  Document convenience helpers
    # -------------------------------------------------------------------------

    def ingest_document(self, file_path: str) -> str:
        doc_agent: DocumentAgent = self._agents[Intent.DOCUMENT]  # type: ignore
        return doc_agent.ingest(file_path)

    def list_documents(self) -> list[dict]:
        doc_agent: DocumentAgent = self._agents[Intent.DOCUMENT]  # type: ignore
        return doc_agent.list_documents()

    # -------------------------------------------------------------------------
    #  Memory
    # -------------------------------------------------------------------------

    def clear_memory(self) -> None:
        self._memory.clear()
        logger.info("Cleared memory for session '%s'.", self._session_id)

    # -------------------------------------------------------------------------
    #  Agent registry
    # -------------------------------------------------------------------------

    def get_agent(self, intent: Intent) -> BaseAgent:
        return self._agents[intent]

    def add_agent(self, intent: Intent, agent: BaseAgent) -> None:
        self._agents[intent] = agent
        logger.info("Agent '%s' registered for intent '%s'.",
                    agent.name, intent.value)

    # -------------------------------------------------------------------------
    #  Scheduler
    # -------------------------------------------------------------------------

    def start_scheduler(self) -> None:
        try:
            from core.scheduler import get_scheduler
            sched = get_scheduler()
            if not sched.is_running:
                sched.start()
        except Exception as exc:
            logger.warning("Scheduler start failed: %s", exc)

    def stop_scheduler(self) -> None:
        try:
            from core.scheduler import get_scheduler
            get_scheduler().stop()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    #  Properties
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def memory(self) -> PersistentMemory:
        return self._memory

    @property
    def intents(self) -> list[Intent]:
        return list(self._agents.keys())

    # -------------------------------------------------------------------------
    #  Async variants
    # -------------------------------------------------------------------------


    def stream_response(self, query: str, **kwargs):
        """
        Stream an agent response token by token.

        Follows the same routing logic as ``run()`` — intent detection,
        RAG pre-check, then the primary agent — but yields tokens as they
        arrive rather than waiting for the full response.

        If the RAG pre-check returns a cached KB answer the full text is
        yielded as a single token (cached answers are pre-computed).

        Yields
        ------
        str  Token string fragments from the agent.

        Notes
        -----
        Post-processing (memory save, episodic store, trace) happens after
        the generator is exhausted, not during streaming.
        """
        forced  = kwargs.pop("intent", None)
        user_id = kwargs.get("user_id", None)

        intent = (
            Intent(forced) if isinstance(forced, str)
            else forced if forced
            else self._route(query)
        )


        # ── Schedule detection ────────────────────────────────────────────────
        # If the query contains a scheduling intent, register a user task
        # and return a confirmation instead of running the normal agent.
        sched_response = self._maybe_schedule_task(query, intent, user_id or self._user_id)
        if sched_response is not None:
            self._close_trace(trace_obj, sched_response)
            self._memory.save_context(query, sched_response.output)
            return sched_response

        # ── RAG pre-check ─────────────────────────────────────────────────────
        if self._enable_rag_precheck:
            try:
                rag_response = rag_precheck(
                    query, intent.value,
                    similarity_threshold=self._rag_similarity_threshold,
                    k=self._rag_k,
                )
            except Exception:
                rag_response = None

            if rag_response is not None:
                self._memory.save_context(query, rag_response.output)
                yield rag_response.output
                return

        # ── Stream from primary agent ─────────────────────────────────────────
        agent  = self._agents.get(intent) or self._agents[Intent.CHAT]
        buffer = []

        try:
            for token in agent.stream(query, **kwargs):
                buffer.append(token)
                yield token
        except Exception as exc:
            logger.error("Streaming error in agent '%s': %s", agent.name, exc)
            yield f"\n[Error: {exc}]"
            return

        # ── Post-processing ───────────────────────────────────────────────────
        full_output = "".join(buffer)
        self._memory.save_context(query, full_output)

    async def run_async(self, query: str, **kwargs: Any) -> AgentResponse:
        import asyncio
        return await asyncio.to_thread(self.run, query, **kwargs)

    async def fanout(
        self,
        query:   str,
        intents: list[Intent],
        **kwargs: Any,
    ) -> list[AgentResponse]:
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner()
        agents = [self._agents[i] for i in intents if i in self._agents]
        result = await runner.fanout(query, agents, **kwargs)
        return result.responses

    # -------------------------------------------------------------------------
    #  LangGraph pipeline
    # -------------------------------------------------------------------------

    def build_graph(self):
        """Build and return a compiled LangGraph StateGraph."""
        try:
            from langgraph.graph import StateGraph, END  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langgraph is not installed. Run: pip install langgraph"
            ) from exc

        graph = StateGraph(OrchestratorState)

        def route_node(state: OrchestratorState) -> OrchestratorState:
            intent = self._route(state["query"])
            return {**state, "intent": intent.value}

        def _agent_node(target: Intent):
            def _node(state: OrchestratorState) -> OrchestratorState:
                try:
                    resp = self._agents[target].run(state["query"])
                    return {**state, "response": resp}
                except Exception as exc:
                    err = AgentResponse(
                        output=f"Agent error: {exc}",
                        agent_name=target.value,
                        error=str(exc),
                    )
                    return {**state, "response": err, "error": str(exc)}
            _node.__name__ = f"{target.value}_node"
            return _node

        def dispatch(state: OrchestratorState) -> str:
            return state.get("intent", Intent.CHAT.value)

        graph.add_node("router",   route_node)
        graph.add_node("chat",     _agent_node(Intent.CHAT))
        graph.add_node("code",     _agent_node(Intent.CODE))
        graph.add_node("news",     _agent_node(Intent.NEWS))
        graph.add_node("search",   _agent_node(Intent.SEARCH))
        graph.add_node("document", _agent_node(Intent.DOCUMENT))
        graph.add_node("finance",  _agent_node(Intent.FINANCE))
        graph.add_node("research", _agent_node(Intent.RESEARCH))

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router", dispatch,
            {
                Intent.CHAT.value:     "chat",
                Intent.CODE.value:     "code",
                Intent.NEWS.value:     "news",
                Intent.SEARCH.value:   "search",
                Intent.DOCUMENT.value: "document",
                Intent.UNKNOWN.value:  "chat",
                Intent.FINANCE.value:  "finance",
                Intent.RESEARCH.value: "research",
            },
        )
        for node in ("chat", "code", "news", "search", "document", "finance", "research"):
            graph.add_edge(node, END)

        return graph.compile()

    # -------------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------------



    # ── Schedule detection ────────────────────────────────────────────────────

    _SCHEDULE_TRIGGERS = re.compile(
        r"\b(remind me|set a reminder|schedule|every (day|morning|evening|hour|week|"
        r"\d+ (minutes?|hours?|days?))|send me (a|the) .{0,30} (every|each|daily|weekly))"
        r"\b",
        re.IGNORECASE,
    )

    def _maybe_schedule_task(
        self,
        query:   str,
        intent:  "Intent",
        user_id: str,
    ) -> "Optional[AgentResponse]":
        """
        Check if the query is a scheduling request.

        Recognises patterns like:
          "Remind me to check Apple stock every hour"
          "Schedule a news briefing every morning"
          "Send me the weather every day"

        Returns an AgentResponse confirmation, or None if not a schedule request.
        """
        if not self._SCHEDULE_TRIGGERS.search(query):
            return None

        try:
            from core.user_task_scheduler import get_task_manager, parse_schedule

            interval_min, desc = parse_schedule(query)
            if interval_min <= 0:
                return None   # couldn't parse an interval — let the agent handle it

            mgr = get_task_manager()

            # Infer the underlying task query (strip scheduling language)
            task_query = re.sub(
                r"remind me (to |about )?|schedule (a |an )?|send me (a |the )?|"
                r"every (day|morning|evening|hour|\d+ (minutes?|hours?|days?))|"
                r"daily|weekly",
                "", query, flags=re.IGNORECASE,
            ).strip()
            if not task_query:
                task_query = query

            task = mgr.add_task(
                user_id=user_id,
                session_id=self._session_id,
                description=task_query[:80],
                query=task_query,
                intent=intent.value,
                interval_minutes=interval_min,
            )

            confirmation = (
                f"✓ Scheduled: **{task_query[:60]}**\n"
                f"Runs {desc} (Task ID: `{task.task_id}`).\n"
                f"Use `:tasks` in the REPL to manage your scheduled tasks."
            )
            logger.info(
                "Scheduled user task '%s' every %.0f min for user '%s'.",
                task_query[:50], interval_min, user_id,
            )
            return AgentResponse(
                output=confirmation,
                agent_name="scheduler",
                metadata={"task_id": task.task_id, "interval_minutes": interval_min},
            )
        except Exception as exc:
            logger.debug("Schedule detection failed (non-fatal): %s", exc)
            return None

    def _update_user_profile(
        self,
        query:      str,
        response:   str,
        agent_name: str,
        user_id:    str,
    ) -> None:
        """
        Run fast heuristic profile extraction inline, then schedule the
        heavier LLM-based extraction on a background thread to avoid
        adding latency to the response path.
        """
        try:
            from core.profile_extractor import (
                extract_and_update_profile,
                extract_and_update_profile_llm,
            )
            # Fast heuristics — synchronous, < 1 ms
            extract_and_update_profile(
                self._session_id, user_id, query, response, agent_name
            )

            # LLM extraction — background thread, every 5th turn
            msg_count = len(self._memory.messages)
            if msg_count % 10 == 0 and msg_count > 0:
                import threading
                t = threading.Thread(
                    target=extract_and_update_profile_llm,
                    args=(self._session_id, user_id, query, response),
                    daemon=True,
                )
                t.start()
        except Exception as exc:
            logger.debug("Profile update failed (non-fatal): %s", exc)

    def _route(self, query: str) -> Intent:
        intent = _keyword_route(query)
        return intent if intent is not None else _llm_route(query)

    def _load_plugins(self) -> None:
        try:
            from plugins import load_all_plugins
            agent_classes, _ = load_all_plugins()
            for name, cls in agent_classes.items():
                try:
                    self._agents[name] = cls(memory=self._memory)  # type: ignore
                    logger.info("Plugin agent '%s' registered.", name)
                except Exception as exc:
                    logger.warning("Plugin '%s' failed: %s", name, exc)
        except Exception as exc:
            logger.debug("Plugin loading skipped: %s", exc)

    def _open_trace(self, query: str):
        try:
            from core.tracing import get_tracer
            from core.tracing.tracer import Trace
            tracer = get_tracer()
            trace  = Trace(
                trace_id=tracer._next_id(),
                session_id=self._session_id,
                query=query,
            )
            return (tracer, trace)
        except Exception:
            return None

    def _close_trace(self, trace_ctx, response: AgentResponse) -> None:
        if trace_ctx is None:
            return
        try:
            tracer, trace = trace_ctx
            trace.set_outcome(
                response.output,
                error=response.error,
                agent_name=response.agent_name,
            )
            tracer.store.record(trace)
        except Exception as exc:
            logger.debug("Trace record failed: %s", exc)

    def _inject_episodic_context(self, query: str, kwargs: dict) -> None:
        try:
            context = self._episodic.recall_as_context(query)  # type: ignore
            if context:
                kwargs.setdefault("episodic_context", context)
        except Exception as exc:
            logger.debug("Episodic recall failed: %s", exc)

    def _store_episodic_facts(self, query: str, output: str) -> None:
        try:
            self._episodic.extract_and_store(  # type: ignore
                session_id=self._session_id,
                user_query=query,
                agent_response=output,
            )
        except Exception as exc:
            logger.debug("Episodic store failed: %s", exc)

    def __repr__(self) -> str:
        agents_str = ", ".join(
            i.value if isinstance(i, Intent) else str(i) for i in self._agents
        )
        return (
            f"<Orchestrator session='{self._session_id}' "
            f"agents=[{agents_str}] fallback_max={self._max_fallback_attempts} "
            f"rag_precheck={self._enable_rag_precheck}>"
        )
