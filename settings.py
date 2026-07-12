"""
config/settings.py
──────────────────
Central configuration for the AI Credit Analyst Platform.
All tunable constants live here; nothing is hardcoded in business logic.
"""

from __future__ import annotations
from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR  = DATA_DIR / "cache"

# Ensure runtime directories exist
for _dir in (UPLOAD_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Supported File Types ──────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}

# ── Statement-Type Detection Keywords ────────────────────────────────────────
# Used by document_processor to auto-classify an uploaded file.
STATEMENT_KEYWORDS: dict[str, list[str]] = {
    "income_statement": [
        "revenue", "net revenue", "total revenue", "sales",
        "gross profit", "operating income", "ebit", "ebitda",
        "net income", "net profit", "earnings", "cost of goods sold",
        "cogs", "operating expenses", "interest expense", "income tax",
    ],
    "balance_sheet": [
        "total assets", "current assets", "non-current assets",
        "total liabilities", "current liabilities", "long-term debt",
        "shareholders equity", "stockholders equity", "retained earnings",
        "cash and cash equivalents", "accounts receivable", "inventory",
        "accounts payable", "short-term debt", "goodwill",
    ],
    "cash_flow": [
        "cash flow from operations", "operating activities",
        "investing activities", "financing activities",
        "capital expenditure", "capex", "free cash flow",
        "net change in cash", "depreciation", "amortization",
        "dividends paid", "proceeds from debt",
    ],
}

# Minimum keyword matches required to classify a statement
STATEMENT_MATCH_THRESHOLD = 3

# ── LLM Configuration ────────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER = "anthropic"   # "anthropic" | "openai"
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
DEFAULT_MODEL_OPENAI    = "gpt-4o"
LLM_MAX_TOKENS          = 4096
LLM_TEMPERATURE         = 0.2        # Low temp → analytical, consistent output

# ── Credit Scoring Weights ────────────────────────────────────────────────────
SCORING_WEIGHTS = {
    "profitability": 0.20,
    "liquidity":     0.15,
    "leverage":      0.15,
    "cash_flow":     0.10,
    "ratios":        0.40,  # composite of all ratio scores
}

# Credit rating bands (min score inclusive)
RATING_BANDS: list[tuple[int, str]] = [
    (90, "AAA"),
    (80, "AA"),
    (70, "A"),
    (60, "BBB"),
    (50, "BB"),
    (40, "B"),
    (25, "CCC"),
    (0,  "D"),
]

# ── Dashboard Theme ───────────────────────────────────────────────────────────
THEME = {
    "bg_primary":    "#0D1117",
    "bg_secondary":  "#161B22",
    "bg_card":       "#1C2128",
    "accent_green":  "#00FF87",
    "accent_blue":   "#58A6FF",
    "accent_amber":  "#F0B429",
    "accent_red":    "#FF4040",
    "text_primary":  "#E6EDF3",
    "text_muted":    "#8B949E",
    "border":        "#30363D",
}

# ── Ratio Benchmarks ─────────────────────────────────────────────────────────
# Used by ratio_engine for interpretation and traffic-light scoring.
RATIO_BENCHMARKS = {
    "current_ratio":         {"good": 2.0, "acceptable": 1.0, "poor": 0.5},
    "quick_ratio":           {"good": 1.5, "acceptable": 1.0, "poor": 0.5},
    "cash_ratio":            {"good": 0.5, "acceptable": 0.2, "poor": 0.1},
    "gross_margin":          {"good": 0.40, "acceptable": 0.20, "poor": 0.10},
    "ebitda_margin":         {"good": 0.20, "acceptable": 0.10, "poor": 0.05},
    "net_margin":            {"good": 0.10, "acceptable": 0.05, "poor": 0.01},
    "roa":                   {"good": 0.10, "acceptable": 0.05, "poor": 0.01},
    "roe":                   {"good": 0.15, "acceptable": 0.08, "poor": 0.03},
    "debt_to_equity":        {"good": 0.50, "acceptable": 1.50, "poor": 3.00},
    "debt_to_ebitda":        {"good": 2.00, "acceptable": 4.00, "poor": 6.00},
    "interest_coverage":     {"good": 5.00, "acceptable": 2.00, "poor": 1.00},
    "asset_turnover":        {"good": 1.00, "acceptable": 0.50, "poor": 0.20},
    "receivable_days":       {"good": 30,   "acceptable": 60,   "poor": 90  },
    "payable_days":          {"good": 45,   "acceptable": 60,   "poor": 90  },
    "inventory_turnover":    {"good": 6.00, "acceptable": 3.00, "poor": 1.00},
    "ocf_ratio":             {"good": 0.20, "acceptable": 0.10, "poor": 0.00},
    "fcf_margin":            {"good": 0.10, "acceptable": 0.05, "poor": 0.00},
}
