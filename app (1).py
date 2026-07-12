"""
Corporate Credit Analyst AI — Main Streamlit Application
A professional credit assessment system for wholesale/corporate banking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ── Page config (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="Corporate Credit Analyst AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ───────────────────────────────────────────────────────
from modules.file_parser import parse_uploaded_file, detect_statement_type, preview_dataframe
from modules.financial_mapping import map_balance_sheet, map_income_statement, map_cash_flow
from modules.ratio_engine import RatioEngine, interpret_ratio
from modules.credit_engine import (
    calculate_credit_score, generate_five_cs_assessment,
    generate_loan_recommendation
)
from modules.ai_analysis import generate_ai_analysis
from modules.dashboard import (
    credit_score_gauge, component_scores_radar, ratio_bar_chart,
    risk_heatmap, benchmark_comparison_chart, income_breakdown_pie,
    trend_chart, waterfall_chart
)
from modules.benchmark import get_benchmark, compare_to_benchmark, benchmark_summary, available_industries
from modules.pdf_report import generate_pdf_report
from utils.helpers import fmt_currency, fmt_pct, fmt_ratio
from utils.config import APP_TITLE, INDUSTRY_BENCHMARKS


# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Root & Typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark Banking Theme ── */
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: #E2E8F0;
}

/* ── Header Banner ── */
.bank-header {
    background: linear-gradient(135deg, #1B3A6B 0%, #2563EB 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    border-left: 5px solid #F59E0B;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.3);
}
.bank-header h1 {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.bank-header p {
    color: #93C5FD;
    font-size: 14px;
    margin: 0;
}

/* ── Metric Cards ── */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 4px 0;
    transition: border-color 0.2s;
}
.metric-card:hover {
    border-color: #2563EB;
}
.metric-label {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.metric-value {
    color: #F1F5F9;
    font-size: 22px;
    font-weight: 700;
    line-height: 1.2;
}
.metric-sub {
    color: #64748B;
    font-size: 11px;
    margin-top: 2px;
}

/* ── Rating Badge ── */
.rating-aaa, .rating-aa, .rating-a { color: #10B981; font-weight: 700; }
.rating-bbb, .rating-bb { color: #F97316; font-weight: 700; }
.rating-b, .rating-ccc { color: #EF4444; font-weight: 700; }

/* ── Section Headers ── */
.section-title {
    color: #93C5FD;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    border-bottom: 1px solid #334155;
    padding-bottom: 8px;
    margin: 20px 0 12px 0;
}

/* ── AI Analysis Cards ── */
.ai-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}
.ai-card h4 {
    color: #60A5FA;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 8px 0;
}
.ai-card p {
    color: #CBD5E1;
    font-size: 14px;
    line-height: 1.7;
    margin: 0;
}

/* ── Decision Badge ── */
.decision-approve {
    background: #064E3B;
    border: 1px solid #059669;
    color: #34D399;
    border-radius: 8px;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
}
.decision-conditions {
    background: #451A03;
    border: 1px solid #D97706;
    color: #FBBF24;
    border-radius: 8px;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
}
.decision-reject {
    background: #450A0A;
    border: 1px solid #DC2626;
    color: #F87171;
    border-radius: 8px;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 16px;
    text-align: center;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F172A;
    border-right: 1px solid #1E293B;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #1E293B;
    border: 2px dashed #334155;
    border-radius: 10px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1E293B;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #2563EB !important;
    color: white !important;
}

/* ── Info / Warning pills ── */
.pill-green { background:#064E3B; color:#34D399; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.pill-orange { background:#451A03; color:#FBBF24; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
.pill-red { background:#450A0A; color:#F87171; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "bs_df": None, "is_df": None, "cf_df": None,
        "bs_data": {}, "is_data": {}, "cf_data": {},
        "ratios": None, "credit_score": None,
        "ai_analysis": None, "five_cs": None, "loan_rec": None,
        "company_name": "Company", "industry": "Manufacturing",
        "analysis_done": False,
        "api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏦 Credit Analyst AI")
    st.markdown("---")

    st.session_state.company_name = st.text_input(
        "Company Name", value=st.session_state.company_name
    )
    st.session_state.industry = st.selectbox(
        "Industry Sector", available_industries(),
        index=available_industries().index(st.session_state.industry)
        if st.session_state.industry in available_industries() else 0
    )
    currency = st.selectbox("Currency", ["AED", "USD", "EUR", "GBP", "INR", "SAR"], index=0)

    st.markdown("---")
    st.markdown("#### 🔑 AI Configuration")
    api_key_input = st.text_input(
        "Anthropic API Key (optional)",
        type="password",
        value=st.session_state.api_key,
        help="Provide your Anthropic API key for AI-powered analysis. Leave blank for rule-based analysis."
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("---")
    st.markdown("#### 💰 Loan Simulation")
    loan_amount = st.number_input("Loan Amount", min_value=0, value=5_000_000, step=500_000,
                                   format="%d")
    interest_rate = st.slider("Interest Rate (%)", 1.0, 15.0, 6.5, 0.25)
    tenure = st.slider("Tenure (Years)", 1, 20, 5)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#64748B;'>
    <b>Supported Formats:</b><br>
    Excel (.xlsx) · CSV · PDF*<br>
    *Text-based PDFs only<br><br>
    <b>Required Statements:</b><br>
    ① Balance Sheet<br>
    ② Income Statement (P&L)<br>
    ③ Cash Flow Statement
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="bank-header">
    <h1>🏦 Corporate Credit Analyst AI</h1>
    <p>Wholesale & Corporate Banking · Credit Assessment System · {datetime.now().strftime("%d %B %Y")}</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📂 Upload Statements",
    "📊 Financial Dashboard",
    "⚖️ Ratio Analysis",
    "🤖 AI Credit Assessment",
    "🏭 Benchmarking",
    "📄 Credit Report",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown('<div class="section-title">Upload Financial Statements</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    def upload_panel(col, label, key, emoji):
        with col:
            st.markdown(f"**{emoji} {label}**")
            f = st.file_uploader(
                label, type=["xlsx", "xls", "csv", "pdf"],
                key=key, label_visibility="collapsed"
            )
            if f:
                df, msg = parse_uploaded_file(f)
                if df is not None:
                    detected = detect_statement_type(df)
                    st.success(f"✅ {msg}")
                    st.caption(f"Auto-detected: **{detected}**")
                    with st.expander("Preview data"):
                        st.dataframe(preview_dataframe(df, 12), use_container_width=True)
                else:
                    st.error(f"❌ {msg}")
                    df = None
                return df
            return None

    bs_file  = upload_panel(col1, "Balance Sheet", "bs_upload", "📋")
    is_file  = upload_panel(col2, "Income Statement (P&L)", "is_upload", "💹")
    cf_file  = upload_panel(col3, "Cash Flow Statement", "cf_upload", "💸")

    st.markdown("---")

    # ── Demo data option ─────────────────────────────────
    with st.expander("🔬 Use Demo Data (no uploads required)"):
        st.info("Load a sample manufacturing company to explore all features instantly.")
        if st.button("Load Demo Company — AlNaseem Manufacturing LLC", type="primary"):
            _load_demo()
            st.rerun()

    # ── Analyse button ────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Run Credit Analysis", type="primary", use_container_width=True):
        bs_df = bs_file or st.session_state.bs_df
        is_df = is_file or st.session_state.is_df
        cf_df = cf_file or st.session_state.cf_df

        if bs_df is None or is_df is None or cf_df is None:
            st.warning("⚠️ Please upload all three financial statements, or use demo data.")
        else:
            with st.spinner("Analysing financial statements..."):
                try:
                    # Map to canonical fields
                    bs_data = map_balance_sheet(bs_df)
                    is_data = map_income_statement(is_df)
                    cf_data = map_cash_flow(cf_df)

                    # Calculate ratios
                    engine = RatioEngine(bs_data, is_data, cf_data)
                    ratios = engine.calculate_all()

                    # Credit scoring
                    credit_score = calculate_credit_score(ratios, st.session_state.industry)

                    # Five Cs
                    five_cs = generate_five_cs_assessment(bs_data, is_data, cf_data, ratios)

                    # Loan recommendation
                    loan_rec = generate_loan_recommendation(
                        ratios, credit_score,
                        loan_amount=loan_amount,
                        interest_rate=interest_rate,
                        tenure_years=tenure,
                    )

                    # AI analysis
                    ai_analysis = generate_ai_analysis(
                        st.session_state.company_name, ratios,
                        bs_data, is_data, cf_data,
                        credit_score, st.session_state.industry,
                        api_key=st.session_state.api_key,
                    )

                    # Store in session
                    st.session_state.update({
                        "bs_df": bs_df, "is_df": is_df, "cf_df": cf_df,
                        "bs_data": bs_data, "is_data": is_data, "cf_data": cf_data,
                        "ratios": ratios, "credit_score": credit_score,
                        "five_cs": five_cs, "loan_rec": loan_rec,
                        "ai_analysis": ai_analysis, "analysis_done": True,
                        "currency": currency,
                    })
                    st.success("✅ Analysis complete! Navigate to the tabs above.")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════
# GUARD — show placeholder if no analysis yet
# ══════════════════════════════════════════════════════════════════

def no_analysis_msg():
    st.info("👈 Upload financial statements and run analysis to view results.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════

with tabs[1]:
    if not st.session_state.analysis_done:
        no_analysis_msg()
    else:
        ratios = st.session_state.ratios
        cs = st.session_state.credit_score
        kf = ratios["key_figures"]
        curr = st.session_state.get("currency", "AED")

        st.markdown(f"### {st.session_state.company_name} — Financial Overview")
        st.caption(f"Industry: {st.session_state.industry} | Assessment Date: {datetime.now().strftime('%d %b %Y')}")

        # ── Top KPI row ────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi_items = [
            (c1, "Revenue",       fmt_currency(kf.get("revenue"), curr),       ""),
            (c2, "EBITDA",        fmt_currency(kf.get("ebitda"), curr),         fmt_pct(ratios["profitability"].get("ebitda_margin"))),
            (c3, "Net Income",    fmt_currency(kf.get("net_income"), curr),     fmt_pct(ratios["profitability"].get("net_profit_margin"))),
            (c4, "Total Assets",  fmt_currency(kf.get("total_assets"), curr),   ""),
            (c5, "Free Cash Flow",fmt_currency(kf.get("fcf"), curr),            ""),
        ]
        for col, label, value, sub in kpi_items:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Score gauge + radar ────────────────────────────
        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(
                credit_score_gauge(cs["total_score"], cs["rating"]),
                use_container_width=True
            )
        with g2:
            st.plotly_chart(
                component_scores_radar(cs["component_scores"]),
                use_container_width=True
            )

        # ── Risk heatmap ──────────────────────────────────
        st.plotly_chart(
            risk_heatmap(cs["component_scores"]),
            use_container_width=True
        )

        # ── Income breakdown ──────────────────────────────
        c_l, c_r = st.columns(2)
        with c_l:
            st.plotly_chart(
                income_breakdown_pie(st.session_state.is_data),
                use_container_width=True
            )
        with c_r:
            # Waterfall: Revenue → Gross Profit → EBITDA → EBIT → Net Income
            wf_items = {}
            is_d = st.session_state.is_data
            rev = is_d.get("revenue") or 0
            cogs = is_d.get("cost_of_goods_sold") or 0
            opex = is_d.get("operating_expenses") or 0
            da   = is_d.get("depreciation_amortization") or 0
            ie   = is_d.get("interest_expense") or 0
            tax  = is_d.get("tax_expense") or 0
            ni   = is_d.get("net_income") or 0

            if rev:
                wf_items["Revenue"] = rev
                if cogs: wf_items["(-) COGS"] = -abs(cogs)
                if opex: wf_items["(-) OpEx"] = -abs(opex)
                if ie:   wf_items["(-) Interest"] = -abs(ie)
                if tax:  wf_items["(-) Tax"] = -abs(tax)
                wf_items["Net Income"] = ni
                st.plotly_chart(
                    waterfall_chart(wf_items, "P&L Waterfall"),
                    use_container_width=True
                )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — RATIO ANALYSIS
# ══════════════════════════════════════════════════════════════════

with tabs[2]:
    if not st.session_state.analysis_done:
        no_analysis_msg()
    else:
        ratios = st.session_state.ratios

        def ratio_section(title, data_dict, key_ratios: list):
            st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(key_ratios), 4))
            for i, (key, label, unit) in enumerate(key_ratios):
                val = data_dict.get(key)
                interp = interpret_ratio(key, val)
                with cols[i % 4]:
                    if val is None:
                        display = "N/A"
                    elif unit == "%":
                        display = fmt_pct(val)
                    elif unit == "x":
                        display = fmt_ratio(val)
                    elif unit == "d":
                        display = f"{val:.0f} days"
                    else:
                        display = f"{val:.2f}"

                    color_map = {"green": "#10B981", "orange": "#F97316", "red": "#EF4444", "gray": "#64748B"}
                    c = color_map.get(interp["color"], "#64748B")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{c}">{display}</div>
                        <div class="metric-sub">{interp['label']}</div>
                        <div style="font-size:10px;color:#475569;margin-top:4px">{interp['comment']}</div>
                    </div>""", unsafe_allow_html=True)

        ratio_section("💧 Liquidity", ratios["liquidity"], [
            ("current_ratio", "Current Ratio", "x"),
            ("quick_ratio", "Quick Ratio", "x"),
            ("cash_ratio", "Cash Ratio", "x"),
        ])

        ratio_section("💰 Profitability", ratios["profitability"], [
            ("gross_profit_margin", "Gross Margin", "%"),
            ("ebitda_margin", "EBITDA Margin", "%"),
            ("net_profit_margin", "Net Margin", "%"),
            ("roe", "ROE", "%"),
            ("roa", "ROA", "%"),
            ("roce", "ROCE", "%"),
        ])

        ratio_section("⚖️ Leverage", ratios["leverage"], [
            ("debt_to_equity", "Debt / Equity", "x"),
            ("debt_ratio", "Debt Ratio", "x"),
            ("interest_coverage", "Interest Coverage", "x"),
            ("dscr", "DSCR", "x"),
            ("net_debt_to_ebitda", "Net Debt / EBITDA", "x"),
        ])

        ratio_section("🔄 Efficiency", ratios["efficiency"], [
            ("asset_turnover", "Asset Turnover", "x"),
            ("receivable_days", "Receivable Days", "d"),
            ("payable_days", "Payable Days", "d"),
            ("inventory_turnover", "Inventory Turnover", "x"),
            ("cash_conversion_cycle", "Cash Conv. Cycle", "d"),
        ])

        ratio_section("💸 Cash Flow", ratios["cash_flow"], [
            ("operating_cash_flow", "Operating CF", ""),
            ("free_cash_flow", "Free CF", ""),
            ("cash_flow_coverage", "CF Coverage", "x"),
            ("ocf_to_revenue", "OCF / Revenue", "%"),
        ])

        # Charts
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            bench = get_benchmark(st.session_state.industry)
            st.plotly_chart(
                ratio_bar_chart(
                    {k: v for k, v in ratios["liquidity"].items() if v is not None},
                    "Liquidity Ratios",
                    {"current_ratio": bench.get("current_ratio")},
                ),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                ratio_bar_chart(
                    {k: v for k, v in ratios["profitability"].items()
                     if v is not None and k in ("gross_profit_margin","ebitda_margin","net_profit_margin","roe","roa")},
                    "Profitability Ratios (decimal)",
                    {"gross_profit_margin": bench.get("gross_margin"),
                     "net_profit_margin": bench.get("net_margin"),
                     "roe": bench.get("roe"),
                     "roa": bench.get("roa")},
                ),
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════
# TAB 4 — AI CREDIT ASSESSMENT
# ══════════════════════════════════════════════════════════════════

with tabs[3]:
    if not st.session_state.analysis_done:
        no_analysis_msg()
    else:
        cs = st.session_state.credit_score
        ai = st.session_state.ai_analysis
        five_cs = st.session_state.five_cs
        loan_rec = st.session_state.loan_rec

        # Error banner if AI fell back
        if "_error" in ai:
            st.warning(f"ℹ️ {ai['_error']}")

        # ── Credit rating ────────────────────────────────
        rating = cs["rating"]
        score  = cs["total_score"]
        outlook = cs["outlook"]

        r1, r2, r3 = st.columns([1, 1, 1])
        with r1:
            rcolor = {"AAA":"#10B981","AA":"#10B981","A":"#10B981","BBB":"#F97316","BB":"#F97316","B":"#EF4444","CCC":"#EF4444"}.get(rating, "#94A3B8")
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:24px">
                <div class="metric-label">Credit Rating</div>
                <div style="font-size:52px;font-weight:800;color:{rcolor};line-height:1.1">{rating}</div>
                <div style="color:#94A3B8;font-size:13px">Outlook: {outlook}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:24px">
                <div class="metric-label">Credit Score</div>
                <div style="font-size:52px;font-weight:800;color:{rcolor};line-height:1.1">{score}</div>
                <div style="color:#94A3B8;font-size:13px">out of 100</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            dec = loan_rec["decision"]
            dec_class = {
                "APPROVE": "decision-approve",
                "APPROVE WITH CONDITIONS": "decision-conditions",
            }.get(dec.split("/")[0].strip(), "decision-reject")
            st.markdown(f"""
            <div class="{dec_class}" style="height:100%;display:flex;align-items:center;justify-content:center;min-height:120px">
                {loan_rec["decision_icon"]} {dec}
            </div>""", unsafe_allow_html=True)

        st.markdown(f"> **{cs['rating_description']}**")

        # ── Strengths / Weaknesses ─────────────────────────
        st.markdown("---")
        sw1, sw2, sw3 = st.columns(3)
        with sw1:
            st.markdown("#### ✅ Strengths")
            for s in cs.get("strengths", []) or ["No significant strengths identified."]:
                st.markdown(f"- {s.replace('✅ ', '')}")
        with sw2:
            st.markdown("#### ❌ Weaknesses")
            for w in cs.get("weaknesses", []) or ["No critical weaknesses identified."]:
                st.markdown(f"- {w.replace('❌ ', '')}")
        with sw3:
            st.markdown("#### ⚠️ Cautions")
            for c in cs.get("cautions", []) or ["No significant cautions."]:
                st.markdown(f"- {c.replace('⚠️ ', '')}")

        # ── AI Analysis Sections ──────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 AI Analyst Commentary")

        section_meta = [
            ("executive_summary",   "📋 Executive Summary"),
            ("financial_analysis",  "📊 Financial Analysis"),
            ("risk_assessment",     "⚠️ Risk Assessment"),
            ("loan_recommendation", "🏦 Loan Recommendation"),
            ("analyst_notes",       "📝 Analyst Notes"),
        ]
        for key, title in section_meta:
            text = ai.get(key, "")
            if text:
                st.markdown(f"""
                <div class="ai-card">
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>""", unsafe_allow_html=True)

        # ── Five Cs ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 5️⃣ Five Cs of Credit")
        c_icons = {"Character": "👤", "Capacity": "💪", "Capital": "🏛️",
                   "Collateral": "🔐", "Conditions": "🌍"}
        for c_name, c_text in five_cs.items():
            with st.expander(f"{c_icons.get(c_name, '•')} {c_name}"):
                st.write(c_text)

        # ── Loan Simulation ───────────────────────────────
        if loan_rec.get("loan_simulation"):
            sim = loan_rec["loan_simulation"]
            st.markdown("---")
            st.markdown("### 💰 Loan Simulation Results")
            curr = st.session_state.get("currency", "AED")

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Loan Amount", fmt_currency(sim["loan_amount"], curr))
            with s2:
                st.metric("Monthly Payment", fmt_currency(sim["monthly_payment"], curr))
            with s3:
                st.metric("Total Interest", fmt_currency(sim["total_interest"], curr))
            with s4:
                dscr_val = sim.get("new_dscr")
                afford = "✅ Affordable" if sim.get("affordable") else "❌ Strained"
                dscr_str = f"{dscr_val:.2f}x" if dscr_val else "N/A"
                st.metric("Post-Loan DSCR", dscr_str, delta=afford)

        # ── Conditions ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Recommended Conditions")
        for i, cond in enumerate(loan_rec.get("conditions", []), 1):
            st.markdown(f"**{i}.** {cond}")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — BENCHMARKING
# ══════════════════════════════════════════════════════════════════

with tabs[4]:
    if not st.session_state.analysis_done:
        no_analysis_msg()
    else:
        ratios = st.session_state.ratios
        industry = st.session_state.industry

        comparison = compare_to_benchmark(ratios, industry)
        summary = benchmark_summary(comparison)

        st.markdown(f"### {st.session_state.company_name} vs {industry} Industry Benchmarks")
        st.info(summary)

        # Summary cards
        cols = st.columns(4)
        for i, (metric, data) in enumerate(comparison.items()):
            with cols[i % 4]:
                comp_val = data["company"]
                bench_val = data["benchmark"]
                delta = data["delta_pct"]
                assessment = data["assessment"]
                label = data["label"]

                color = "#10B981" if "Above" in assessment else "#F97316" if "Average" in assessment else "#EF4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">{comp_val:.2f}</div>
                    <div class="metric-sub">Benchmark: {bench_val:.2f} · Delta: {delta:+.1f}%</div>
                    <div style="font-size:11px;margin-top:4px">{assessment}</div>
                </div>""", unsafe_allow_html=True)

        # Benchmark bar chart
        bench = get_benchmark(industry)
        company_for_chart = {
            "current_ratio": ratios["liquidity"].get("current_ratio"),
            "debt_equity":   ratios["leverage"].get("debt_to_equity"),
            "gross_margin":  ratios["profitability"].get("gross_profit_margin"),
            "net_margin":    ratios["profitability"].get("net_profit_margin"),
            "roe":           ratios["profitability"].get("roe"),
            "roa":           ratios["profitability"].get("roa"),
        }
        st.plotly_chart(
            benchmark_comparison_chart(company_for_chart, bench, industry),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════
# TAB 6 — PDF REPORT
# ══════════════════════════════════════════════════════════════════

with tabs[5]:
    if not st.session_state.analysis_done:
        no_analysis_msg()
    else:
        st.markdown("### 📄 Credit Assessment Report")
        st.info("Generate a professional, bank-style PDF credit memo ready for submission to credit committee.")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Report includes:**
            - ✅ Confidential cover header
            - ✅ Executive summary
            - ✅ Financial performance & key figures
            - ✅ Complete ratio analysis table
            - ✅ Risk assessment (Strengths / Weaknesses)
            - ✅ Five Cs of Credit
            - ✅ Credit rating with outlook
            - ✅ Loan recommendation with conditions
            - ✅ Loan simulation results
            - ✅ Analyst notes & disclaimer
            """)

        with col2:
            if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating professional credit memo..."):
                    try:
                        pdf_bytes = generate_pdf_report(
                            company_name=st.session_state.company_name,
                            industry=st.session_state.industry,
                            ratios=st.session_state.ratios,
                            credit_score=st.session_state.credit_score,
                            ai_analysis=st.session_state.ai_analysis,
                            five_cs=st.session_state.five_cs,
                            loan_rec=st.session_state.loan_rec,
                            bs=st.session_state.bs_data,
                            is_=st.session_state.is_data,
                            currency=st.session_state.get("currency", "AED"),
                        )
                        filename = f"Credit_Assessment_{st.session_state.company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.success("✅ PDF generated successfully!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════
# DEMO DATA LOADER
# ══════════════════════════════════════════════════════════════════

def _load_demo():
    """Load a realistic demo manufacturing company."""
    # Balance Sheet
    bs_rows = [
        ("Line Item", "2023"),
        # Current Assets
        ("Cash and Cash Equivalents", 12_500_000),
        ("Accounts Receivable", 38_000_000),
        ("Inventory", 22_000_000),
        ("Other Current Assets", 4_500_000),
        ("Total Current Assets", 77_000_000),
        # Non-current
        ("Property Plant and Equipment", 95_000_000),
        ("Intangible Assets", 5_000_000),
        ("Long-term Investments", 8_000_000),
        ("Total Non-Current Assets", 108_000_000),
        ("Total Assets", 185_000_000),
        # Current Liabilities
        ("Accounts Payable", 21_000_000),
        ("Short Term Debt", 15_000_000),
        ("Other Current Liabilities", 8_000_000),
        ("Total Current Liabilities", 44_000_000),
        # Non-current
        ("Long Term Debt", 65_000_000),
        ("Other Long Term Liabilities", 6_000_000),
        ("Total Non-Current Liabilities", 71_000_000),
        ("Total Liabilities", 115_000_000),
        # Equity
        ("Share Capital", 30_000_000),
        ("Retained Earnings", 38_000_000),
        ("Other Equity", 2_000_000),
        ("Total Equity", 70_000_000),
        ("Total Liabilities and Equity", 185_000_000),
    ]

    # Income Statement
    is_rows = [
        ("Line Item", "2023"),
        ("Revenue", 210_000_000),
        ("Cost of Goods Sold", 147_000_000),
        ("Gross Profit", 63_000_000),
        ("Operating Expenses", 28_000_000),
        ("EBITDA", 42_000_000),
        ("Depreciation and Amortization", 7_000_000),
        ("EBIT", 35_000_000),
        ("Interest Expense", 7_800_000),
        ("Earnings Before Tax", 27_200_000),
        ("Tax Expense", 3_000_000),
        ("Net Income", 24_200_000),
    ]

    # Cash Flow
    cf_rows = [
        ("Line Item", "2023"),
        ("Operating Cash Flow", 38_500_000),
        ("Capital Expenditure", (12_000_000)),
        ("Investing Cash Flow", (14_000_000)),
        ("Debt Issued", 10_000_000),
        ("Debt Repayment", (8_000_000)),
        ("Dividends Paid", (5_000_000)),
        ("Financing Cash Flow", (3_000_000)),
        ("Net Change in Cash", 21_500_000),
    ]

    bs_df = pd.DataFrame(bs_rows[1:], columns=bs_rows[0])
    is_df = pd.DataFrame(is_rows[1:], columns=is_rows[0])
    cf_df = pd.DataFrame(cf_rows[1:], columns=cf_rows[0])

    st.session_state.bs_df = bs_df
    st.session_state.is_df = is_df
    st.session_state.cf_df = cf_df
    st.session_state.company_name = "AlNaseem Manufacturing LLC"
    st.session_state.industry = "Manufacturing"

    # Run analysis immediately
    try:
        bs_data = map_balance_sheet(bs_df)
        is_data = map_income_statement(is_df)
        cf_data = map_cash_flow(cf_df)

        engine = RatioEngine(bs_data, is_data, cf_data)
        ratios = engine.calculate_all()
        credit_score = calculate_credit_score(ratios, "Manufacturing")
        five_cs = generate_five_cs_assessment(bs_data, is_data, cf_data, ratios)
        loan_rec = generate_loan_recommendation(ratios, credit_score,
                                                 loan_amount=20_000_000,
                                                 interest_rate=6.5,
                                                 tenure_years=5)
        ai_analysis = generate_ai_analysis(
            "AlNaseem Manufacturing LLC", ratios,
            bs_data, is_data, cf_data, credit_score, "Manufacturing",
            api_key=st.session_state.get("api_key", ""),
        )

        st.session_state.update({
            "bs_data": bs_data, "is_data": is_data, "cf_data": cf_data,
            "ratios": ratios, "credit_score": credit_score,
            "five_cs": five_cs, "loan_rec": loan_rec,
            "ai_analysis": ai_analysis, "analysis_done": True,
            "currency": "AED",
        })
    except Exception as e:
        st.error(f"Demo load error: {e}")
