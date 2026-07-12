"""
app.py
───────
AI Credit Analyst Platform — Streamlit entry point.

Phase 1: Document Upload System
    • Upload up to 3 financial statements (Income / Balance Sheet / Cash Flow)
    • Optional company & industry context
    • Auto-detect statement types
    • Display extraction summary + raw table preview

Run:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Credit Analyst Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import THEME, ALLOWED_EXTENSIONS
from modules.document_processor import DocumentProcessor, ProcessedDocument

# ── Injected CSS — Dark Fintech Theme ────────────────────────────────────────
def _inject_css() -> None:
    t = THEME
    st.markdown(f"""
    <style>
    /* ── Global ── */
    html, body, [class*="css"] {{
        background-color: {t['bg_primary']};
        color: {t['text_primary']};
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    .stApp {{ background-color: {t['bg_primary']}; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {t['bg_secondary']};
        border-right: 1px solid {t['border']};
    }}

    /* ── Upload Zone ── */
    [data-testid="stFileUploadDropzone"] {{
        background-color: {t['bg_card']};
        border: 1px dashed {t['border']};
        border-radius: 8px;
    }}

    /* ── Cards ── */
    .ca-card {{
        background: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }}
    .ca-card-title {{
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {t['text_muted']};
        margin-bottom: 0.4rem;
    }}
    .ca-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }}
    .badge-income    {{ background: rgba(88,166,255,0.15); color: {t['accent_blue']}; }}
    .badge-balance   {{ background: rgba(0,255,135,0.15);  color: {t['accent_green']}; }}
    .badge-cashflow  {{ background: rgba(240,180,41,0.15); color: {t['accent_amber']}; }}
    .badge-unknown   {{ background: rgba(255,64,64,0.15);  color: {t['accent_red']}; }}

    /* ── Section Headers ── */
    .ca-section-header {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {t['text_primary']};
        border-left: 3px solid {t['accent_green']};
        padding-left: 0.7rem;
        margin: 1.5rem 0 1rem;
    }}

    /* ── Metric blocks ── */
    [data-testid="stMetric"] {{
        background: {t['bg_card']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 0.8rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {t['text_muted']}; font-size: 0.75rem; }}
    [data-testid="stMetricValue"] {{ color: {t['accent_green']}; }}

    /* ── Buttons ── */
    .stButton>button {{
        background: {t['accent_green']};
        color: #000;
        border: none;
        font-weight: 700;
        border-radius: 6px;
    }}
    .stButton>button:hover {{
        background: #00cc6a;
        color: #000;
    }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{ border: 1px solid {t['border']}; border-radius: 8px; }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{
        background: {t['bg_card']};
        border-radius: 6px;
        color: {t['text_primary']};
    }}

    /* ── Divider ── */
    hr {{ border-color: {t['border']}; }}
    </style>
    """, unsafe_allow_html=True)


# ── Session State Bootstrap ───────────────────────────────────────────────────
def _init_session() -> None:
    defaults = {
        "documents":       {},   # {slot_key: ProcessedDocument}
        "company_name":    "",
        "company_context": "",
        "industry_context":"",
        "analysis_ready":  False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────
_BADGE_CLASS = {
    "income_statement": ("INCOME STMT",  "badge-income"),
    "balance_sheet":    ("BALANCE SHEET","badge-balance"),
    "cash_flow":        ("CASH FLOW",    "badge-cashflow"),
    "unknown":          ("UNKNOWN",      "badge-unknown"),
}

def _badge_html(stmt_type: str) -> str:
    label, css_class = _BADGE_CLASS.get(stmt_type, ("UNKNOWN", "badge-unknown"))
    return f'<span class="ca-badge {css_class}">{label}</span>'


def _render_document_card(slot: str, doc: ProcessedDocument) -> None:
    """Render a summary card for a successfully processed document."""
    table_count = len(doc.raw_tables)
    row_count   = sum(df.shape[0] for df in doc.raw_tables)
    col_count   = max((df.shape[1] for df in doc.raw_tables), default=0)

    st.markdown(f"""
    <div class="ca-card">
        <div class="ca-card-title">{slot.replace('_', ' ').title()}</div>
        <div style="margin-bottom:0.5rem">{_badge_html(doc.statement_type)}</div>
        <div style="font-size:0.85rem; color:#8B949E;">
            📄 {doc.filename} &nbsp;|&nbsp;
            🗂 {table_count} table(s) &nbsp;|&nbsp;
            📏 {row_count} rows &nbsp;|&nbsp;
            📐 {col_count} columns
        </div>
    </div>
    """, unsafe_allow_html=True)

    if doc.errors:
        with st.expander("⚠️ Processing Warnings", expanded=False):
            for err in doc.errors:
                st.warning(err)


def _render_table_preview(doc: ProcessedDocument) -> None:
    """Show an interactive preview of the primary extracted table."""
    table = doc.primary_table
    if table is None:
        st.info("No structured tables could be extracted from this file.")
        return

    st.markdown('<div class="ca-section-header">Raw Table Preview</div>',
                unsafe_allow_html=True)

    # Show first 30 rows; let user page through
    preview = table.head(30)
    st.dataframe(preview, use_container_width=True, height=320)
    st.caption(f"Showing up to 30 rows of {len(table)} total. "
               "Full data will be used in analysis.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/bar-chart.png",
            width=50,
        )
        st.markdown("## AI Credit Analyst")
        st.markdown(
            '<span style="font-size:0.75rem;color:#8B949E;">'
            'Professional Corporate Credit Analysis</span>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Navigation (phases 2+ will add more pages)
        st.markdown("### Navigation")
        pages = {
            "📤 Upload Documents":   "upload",
            "📊 Dashboard Overview": "dashboard",   # Phase 5
            "📈 Financial Analysis": "analysis",    # Phase 5
            "⚖️ Ratio Dashboard":    "ratios",      # Phase 3
            "🤖 AI Chat":            "chat",         # Phase 6
            "📄 Credit Memo":        "memo",         # Phase 6
        }
        for label, key in pages.items():
            disabled = key != "upload"  # only upload active in Phase 1
            if st.button(label, key=f"nav_{key}", disabled=disabled,
                         use_container_width=True):
                st.session_state["page"] = key

        st.divider()
        st.markdown(
            '<span style="font-size:0.7rem;color:#8B949E;">'
            'Phase 1 — Document Upload System<br>'
            'Phases 2–6 coming soon.</span>',
            unsafe_allow_html=True,
        )


# ── Upload Page ───────────────────────────────────────────────────────────────
def _render_upload_page() -> None:
    st.markdown(
        '<h1 style="font-size:1.8rem;font-weight:800;margin-bottom:0.2rem;">'
        '📤 Document Upload System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#8B949E;margin-top:0;">Upload financial statements to begin '
        'credit analysis. Supported formats: PDF, Excel (.xlsx/.xls), CSV.</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    processor = DocumentProcessor()
    ext_list  = ", ".join(sorted(ALLOWED_EXTENSIONS))

    # ── Company Context ───────────────────────────────────────────────────
    st.markdown('<div class="ca-section-header">Company Information</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state["company_name"] = st.text_input(
            "Company Name",
            value=st.session_state["company_name"],
            placeholder="e.g. Acme Corporation Ltd",
        )
    with col_b:
        st.session_state["company_context"] = st.text_input(
            "Business Description (optional)",
            value=st.session_state["company_context"],
            placeholder="e.g. Mid-size manufacturing company, B2B SaaS, retail chain…",
        )

    st.session_state["industry_context"] = st.text_area(
        "Industry & Market Context (optional)",
        value=st.session_state["industry_context"],
        placeholder=(
            "e.g. Operates in the UAE construction sector. "
            "Market growing at ~6% YoY. Key competitors: X, Y, Z."
        ),
        height=80,
    )

    st.divider()

    # ── Financial Statement Uploads ───────────────────────────────────────
    st.markdown('<div class="ca-section-header">Financial Statements</div>',
                unsafe_allow_html=True)

    upload_slots = {
        "income_statement": {
            "label":   "📈 Income Statement",
            "help":    "P&L / Profit & Loss statement",
            "hint":    "Revenue, EBITDA, Net Income…",
        },
        "balance_sheet": {
            "label":   "⚖️ Balance Sheet",
            "help":    "Statement of Financial Position",
            "hint":    "Assets, Liabilities, Equity…",
        },
        "cash_flow": {
            "label":   "💰 Cash Flow Statement",
            "help":    "Statement of Cash Flows",
            "hint":    "Operating, Investing, Financing cash flows…",
        },
    }

    for slot_key, meta in upload_slots.items():
        with st.expander(f"{meta['label']} — {meta['hint']}", expanded=True):
            uploaded = st.file_uploader(
                label=meta["label"],
                type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
                key=f"uploader_{slot_key}",
                help=meta["help"],
                label_visibility="collapsed",
            )

            if uploaded is not None:
                with st.spinner(f"Processing {uploaded.name}…"):
                    try:
                        doc = processor.process(uploaded, filename=uploaded.name)

                        # Override auto-detection with the slot's expected type
                        # if the model couldn't determine it confidently.
                        if doc.statement_type == "unknown":
                            doc.statement_type = slot_key
                            doc.errors.append(
                                "Statement type could not be auto-detected; "
                                f"defaulting to '{slot_key}' based on upload slot."
                            )

                        st.session_state["documents"][slot_key] = doc
                        _render_document_card(slot_key, doc)

                        with st.expander("🔍 Preview Extracted Data"):
                            _render_table_preview(doc)

                        with st.expander("📊 Detection Confidence"):
                            score_data = {
                                "Statement Type": list(doc.detection_scores.keys()),
                                "Keyword Matches": list(doc.detection_scores.values()),
                            }
                            st.dataframe(
                                pd.DataFrame(score_data)
                                  .sort_values("Keyword Matches", ascending=False),
                                hide_index=True,
                                use_container_width=True,
                            )

                    except Exception as exc:
                        st.error(f"❌ Failed to process **{uploaded.name}**: {exc}")

            elif slot_key in st.session_state["documents"]:
                # File was cleared — remove from state
                del st.session_state["documents"][slot_key]

    st.divider()

    # ── Upload Summary ────────────────────────────────────────────────────
    docs: dict = st.session_state["documents"]
    uploaded_count = len(docs)

    st.markdown('<div class="ca-section-header">Upload Summary</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Files Uploaded",  uploaded_count, f"of 3")
    col2.metric("Income Statement", "✅" if "income_statement" in docs else "—")
    col3.metric("Balance Sheet",    "✅" if "balance_sheet"    in docs else "—")
    col4.metric("Cash Flow",        "✅" if "cash_flow"        in docs else "—")

    st.divider()

    # ── Proceed Button ────────────────────────────────────────────────────
    can_proceed = uploaded_count >= 1  # at least one doc needed
    if can_proceed:
        st.success(
            f"✅  **{uploaded_count} document(s)** loaded. "
            "Ready for financial analysis once Phases 2–6 are complete."
        )
        if st.button("🚀  Run Credit Analysis", use_container_width=True):
            st.session_state["analysis_ready"] = True
            st.info(
                "📌 Analysis engine (Phase 2+) is not yet built. "
                "Your documents are staged and will be processed in the next phase."
            )
    else:
        st.info(
            "⬆️  Upload at least one financial statement above to continue."
        )

    # ── Uploaded-Doc Details ──────────────────────────────────────────────
    if docs:
        st.markdown('<div class="ca-section-header">Uploaded Documents</div>',
                    unsafe_allow_html=True)
        for slot_key, doc in docs.items():
            _render_document_card(slot_key, doc)


# ── Placeholder Page ──────────────────────────────────────────────────────────
def _render_placeholder(title: str) -> None:
    st.markdown(f"# {title}")
    st.info("This section will be available in a future phase.")


# ── App Router ────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()
    _init_session()
    _render_sidebar()

    page = st.session_state.get("page", "upload")

    if page == "upload":
        _render_upload_page()
    else:
        _render_placeholder(f"🚧 {page.replace('_', ' ').title()}")


if __name__ == "__main__":
    main()
