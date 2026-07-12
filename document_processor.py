"""
modules/document_processor.py
───────────────────────────────
Phase 1 — Document ingestion, type detection, and raw-data extraction.

Responsibilities
────────────────
1. Accept a file-like object (Streamlit UploadedFile or Path).
2. Dispatch to the correct parser (PDF / Excel / CSV).
3. Auto-detect which financial statement the file contains.
4. Return a structured ProcessedDocument with raw DataFrames ready
   for financial_extractor.py in Phase 2.

No financial logic lives here — this module is purely about I/O and
classification.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import pandas as pd
import pdfplumber
import fitz  # PyMuPDF — fallback PDF engine

from config.settings import (
    ALLOWED_EXTENSIONS,
    STATEMENT_KEYWORDS,
    STATEMENT_MATCH_THRESHOLD,
)
from utils.validators import validate_file_extension, validate_file_size

# ── Public Types ──────────────────────────────────────────────────────────────

StatementType = str  # "income_statement" | "balance_sheet" | "cash_flow" | "unknown"

STATEMENT_TYPES: list[StatementType] = [
    "income_statement",
    "balance_sheet",
    "cash_flow",
]


@dataclass
class ProcessedDocument:
    """Container returned by DocumentProcessor.process().

    Attributes
    ──────────
    filename        : Original file name.
    statement_type  : Auto-detected statement class or 'unknown'.
    raw_tables      : List of DataFrames extracted from the file
                      (one per sheet or PDF page-table).
    raw_text        : Full text dump (useful for LLM and keyword search).
    detection_scores: Keyword match counts per statement type.
    errors          : Non-fatal warnings collected during processing.
    """
    filename:         str
    statement_type:   StatementType
    raw_tables:       list[pd.DataFrame]     = field(default_factory=list)
    raw_text:         str                    = ""
    detection_scores: dict[str, int]         = field(default_factory=dict)
    errors:           list[str]              = field(default_factory=list)

    @property
    def primary_table(self) -> pd.DataFrame | None:
        """Return the largest extracted table, or None if no tables found."""
        if not self.raw_tables:
            return None
        return max(self.raw_tables, key=lambda df: df.size)


# ── Main Processor Class ──────────────────────────────────────────────────────

class DocumentProcessor:
    """Parse, classify, and surface raw data from uploaded financial files.

    Usage
    ─────
        processor = DocumentProcessor()
        doc = processor.process(uploaded_file)   # Streamlit UploadedFile
        doc = processor.process(Path("report.xlsx"))
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        source: Union[IO[bytes], Path, bytes],
        filename: str = "",
    ) -> ProcessedDocument:
        """Entry point.  Accepts a Streamlit UploadedFile, a Path, or raw bytes.

        Returns a ProcessedDocument regardless of success/failure;
        check doc.errors for non-fatal issues.
        """
        # ── Normalise input to (bytes, filename) ──────────────────────────
        file_bytes, filename = self._normalise_source(source, filename)

        # ── Validate ──────────────────────────────────────────────────────
        if not validate_file_extension(filename):
            suffix = Path(filename).suffix
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        if not validate_file_size(file_bytes):
            raise ValueError("File exceeds the 50 MB limit.")

        # ── Route to parser ───────────────────────────────────────────────
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            raw_tables, raw_text, errors = self._parse_pdf(file_bytes)
        elif suffix in {".xlsx", ".xls"}:
            raw_tables, raw_text, errors = self._parse_excel(file_bytes, suffix)
        elif suffix == ".csv":
            raw_tables, raw_text, errors = self._parse_csv(file_bytes)
        else:
            raise ValueError(f"Unhandled extension: {suffix}")

        # ── Classify statement type ───────────────────────────────────────
        combined_text = raw_text + " ".join(
            " ".join(str(c) for c in df.columns) + " ".join(
                " ".join(str(v) for v in df.iloc[:, 0].astype(str))
            )
            for df in raw_tables
        )
        statement_type, scores = self._detect_statement_type(combined_text)

        return ProcessedDocument(
            filename=filename,
            statement_type=statement_type,
            raw_tables=raw_tables,
            raw_text=raw_text,
            detection_scores=scores,
            errors=errors,
        )

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_pdf(
        self, file_bytes: bytes
    ) -> tuple[list[pd.DataFrame], str, list[str]]:
        """Extract tables and text from a PDF.

        Strategy
        ────────
        1. Try pdfplumber for table extraction (more accurate).
        2. Fall back to PyMuPDF for text if pdfplumber yields nothing.
        """
        tables: list[pd.DataFrame] = []
        full_text_parts: list[str] = []
        errors: list[str] = []

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # ── Text ──────────────────────────────────────────────
                    text = page.extract_text() or ""
                    full_text_parts.append(text)

                    # ── Tables ────────────────────────────────────────────
                    for tbl in page.extract_tables():
                        df = self._raw_table_to_df(tbl)
                        if df is not None:
                            tables.append(df)

        except Exception as exc:  # pdfplumber failed — try PyMuPDF
            errors.append(f"pdfplumber error: {exc}. Falling back to PyMuPDF.")
            tables, full_text_parts, mupdf_errors = self._parse_pdf_mupdf(file_bytes)
            errors.extend(mupdf_errors)

        return tables, "\n".join(full_text_parts), errors

    def _parse_pdf_mupdf(
        self, file_bytes: bytes
    ) -> tuple[list[pd.DataFrame], list[str], list[str]]:
        """PyMuPDF fallback — text extraction only (no table parsing)."""
        text_parts: list[str] = []
        errors: list[str] = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text_parts.append(page.get_text())
        except Exception as exc:
            errors.append(f"PyMuPDF error: {exc}")
        return [], text_parts, errors

    def _parse_excel(
        self, file_bytes: bytes, suffix: str
    ) -> tuple[list[pd.DataFrame], str, list[str]]:
        """Extract all sheets from an Excel workbook.

        Heuristics applied
        ──────────────────
        • Skip sheets shorter than 3 rows (likely cover / chart sheets).
        • Try to detect header row automatically when header=0 looks wrong.
        """
        tables: list[pd.DataFrame] = []
        text_parts: list[str] = []
        errors: list[str] = []

        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"

        try:
            xl = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
            for sheet_name in xl.sheet_names:
                try:
                    df_raw = xl.parse(sheet_name, header=None)
                    if df_raw.shape[0] < 3:
                        continue  # too small to be a financial table

                    df = self._detect_and_set_header(df_raw)
                    df = self._clean_dataframe(df)
                    if not df.empty:
                        tables.append(df)
                        text_parts.append(
                            f"Sheet: {sheet_name}\n" + df.to_string()
                        )
                except Exception as sheet_err:
                    errors.append(f"Sheet '{sheet_name}': {sheet_err}")

        except Exception as exc:
            errors.append(f"Excel parse error: {exc}")

        return tables, "\n".join(text_parts), errors

    def _parse_csv(
        self, file_bytes: bytes
    ) -> tuple[list[pd.DataFrame], str, list[str]]:
        """Parse a CSV file, auto-detecting delimiter."""
        errors: list[str] = []
        try:
            # Try common delimiters
            for sep in (",", ";", "\t", "|"):
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        sep=sep,
                        thousands=",",
                        encoding="utf-8",
                        on_bad_lines="skip",
                    )
                    if df.shape[1] >= 2:   # plausible table
                        df = self._clean_dataframe(df)
                        return [df], df.to_string(), errors
                except Exception:
                    continue
            errors.append("Could not parse CSV with any standard delimiter.")
        except Exception as exc:
            errors.append(f"CSV parse error: {exc}")
        return [], "", errors

    # ── Data Cleaning Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _raw_table_to_df(raw_table: list[list]) -> pd.DataFrame | None:
        """Convert pdfplumber's raw list-of-lists table to a clean DataFrame."""
        if not raw_table or len(raw_table) < 2:
            return None
        try:
            df = pd.DataFrame(raw_table[1:], columns=raw_table[0])
            return DocumentProcessor._clean_dataframe(df)
        except Exception:
            return None

    @staticmethod
    def _detect_and_set_header(df: pd.DataFrame) -> pd.DataFrame:
        """Scan the first 10 rows for the most likely header row.

        A header row is identified as the first row where the majority
        of cells are non-numeric strings (column labels, not values).
        """
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            non_numeric = sum(
                1 for v in row
                if isinstance(v, str) and not _looks_like_number(v)
            )
            if non_numeric >= max(2, len(row) // 2):
                new_df = df.iloc[i + 1:].copy()
                new_df.columns = df.iloc[i].astype(str).str.strip()
                return new_df
        # No clear header found — use default integer columns
        return df

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise a raw DataFrame for downstream use.

        Steps
        ─────
        • Strip whitespace from string columns.
        • Remove fully-empty rows and columns.
        • Coerce obvious numeric columns (drop currency symbols, commas).
        • Normalise column names to lowercase with underscores.
        """
        if df.empty:
            return df

        # Drop fully-empty rows / columns
        df = df.dropna(how="all").reset_index(drop=True)
        df = df.loc[:, df.notna().any()]

        # Strip strings
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        # Coerce numeric columns
        for col in df.columns:
            if df[col].dtype == object:
                cleaned = (
                    df[col]
                    .str.replace(r"[£$€¥,\s]", "", regex=True)
                    .str.replace(r"\((\d+\.?\d*)\)", r"-\1", regex=True)  # (123) → -123
                    .str.replace("%", "", regex=False)
                )
                numeric = pd.to_numeric(cleaned, errors="coerce")
                # Only replace column if more than half the values are numeric
                if numeric.notna().sum() > len(numeric) / 2:
                    df[col] = numeric

        return df

    # ── Statement-Type Detection ──────────────────────────────────────────────

    @staticmethod
    def _detect_statement_type(text: str) -> tuple[StatementType, dict[str, int]]:
        """Score the extracted text against keyword lists for each statement type.

        Returns the winning type (or 'unknown') and the full score dict.
        """
        normalised = re.sub(r"\s+", " ", text.lower())
        scores: dict[str, int] = {}

        for stmt_type, keywords in STATEMENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in normalised)
            scores[stmt_type] = score

        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        if best_score < STATEMENT_MATCH_THRESHOLD:
            return "unknown", scores

        return best_type, scores

    # ── Internal Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_source(
        source: Union[IO[bytes], Path, bytes], filename: str
    ) -> tuple[bytes, str]:
        """Return (file_bytes, filename) from any supported input type."""
        if isinstance(source, bytes):
            return source, filename

        if isinstance(source, Path):
            return source.read_bytes(), filename or source.name

        # Streamlit UploadedFile or any file-like object
        name = filename or getattr(source, "name", "unknown_file")
        if hasattr(source, "read"):
            data = source.read()
            # Streamlit objects may return str — encode if so
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data, name

        raise TypeError(f"Unsupported source type: {type(source)}")


# ── Private Utilities ─────────────────────────────────────────────────────────

def _looks_like_number(value: str) -> bool:
    """Heuristic: does this string represent a financial number?"""
    cleaned = re.sub(r"[£$€¥,\s()%]", "", value.strip())
    try:
        float(cleaned)
        return True
    except ValueError:
        return False
