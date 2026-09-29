"""
ingest_excel.py — Batch ingestion of all Excel/XLS files from the DATA folder.

For each sheet in each file it does TWO things:
  1. Saves a clean CSV  → data/uploads/<table_name>.csv
     (so sql_engine.py can run SQL queries against it)
  2. Embeds a text chunk → Qdrant
     (so LOOKUP/RAG queries can describe the data)

Smart header detection:
  Many institutional sheets have 3-5 merged title rows before the real
  column headers.  This script scans the first 10 rows and picks the one
  with the most non-null string values as the header row.

Run:
    cd App
    source .venv/bin/activate   (or venv/bin/activate)
    python ingest_excel.py
"""

import os, re, warnings, sys
import pandas as pd
warnings.filterwarnings("ignore")

# ── locate the App directory ──────────────────────────────────────────────────
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(APP_DIR, "..", "DATA")
UPLOAD_DIR= os.path.join(APP_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── add App to path so services are importable ────────────────────────────────
sys.path.insert(0, APP_DIR)
from services.embeddings import generate_query_embedding
from services.vectordb   import store_embeddings

# ── all Excel files to ingest ─────────────────────────────────────────────────
EXCEL_FILES = [
    # (relative path from DATA_ROOT,                        friendly_prefix)
    ("STUDENT/2 year.xls",                                  "student_2yr"),
    ("STUDENT/3 year data.xls",                             "student_3yr"),
    ("STUDENT/IT - 4 th year data.xlsx",                    "student_4yr"),
    ("FACUTLY/Telephone Directory.xlsx",                    "faculty_tel"),
    ("ADMINISTRATION/PT_Lee_CNCET_Academic_Data_RAG.xlsx",  "admin_academic"),
    ("ACADEMICS/IT 2023 2027/III IT  BATCH 2023-2027 RA-2025 ODD 5th-SEM.xlsx",  "marks_5sem_ra"),
    ("ACADEMICS/IT 2023 2027/IT batch 2023 2027 5 th sem attendance.xlsx",         "attendance_5sem"),
    ("ACADEMICS/IT 2023 2027/IT batch 2023-2027  3thsem result analysis.xlsx",     "result_3sem"),
    ("ACADEMICS/IT 2023 2027/4 th sem IAT - 2 Mark Sheet.xlsx",                   "marks_4sem_iat2"),
    ("ACADEMICS/IT 2023 2027/6th sem IAT-2 Mark sheet  MARCH 2026.xlsx",          "marks_6sem_iat2"),
    ("ACADEMICS/IT 2023 2027/6th semIAT-1 Mark sheet  FEB 2026.xlsx",             "marks_6sem_iat1"),
    ("ACADEMICS/IT 2023 2027/IAT-2  second sem.xlsx",                              "marks_2sem_iat2"),
    ("ACADEMICS/IT 2023 2027/Model Exam_Mark sheet   4 sem   (1).xlsx",           "marks_4sem_model"),
]

# ── column names for 4th-year sheet (no header row) ──────────────────────────
STUDENT_4YR_COLS = [
    "Photo_File", "Photo_Alt", "Batch", "Student_Name", "Reg_No",
    "Father_Name", "DOB", "Blood_Group", "Department", "Aadhaar",
    "Phone", "Community", "College_Email", "Personal_Email", "Address",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Turn any string into a safe table/filename token."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(name)).strip("_")[:60]


def detect_header_row(df_raw: pd.DataFrame, max_scan: int = 10) -> int:
    """
    Find the row index that looks most like a real header:
    - highest count of unique, non-null, string-valued cells.
    Returns 0 if sheet already has a clean header.
    """
    best_row, best_score = 0, -1
    for i in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[i]
        score = sum(
            1 for v in row
            if isinstance(v, str) and v.strip()
            and "unnamed" not in v.lower()
            and len(v.strip()) < 80   # skip long merged title cells
        )
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def load_sheet(xl: pd.ExcelFile, sheet: str, prefix: str) -> pd.DataFrame | None:
    """
    Load one sheet into a clean DataFrame with proper column names.
    Returns None if the sheet is empty after cleaning.
    """
    # 1. Read raw (no header) to detect layout
    raw = xl.parse(sheet, header=None, dtype=str)
    raw.dropna(how="all", inplace=True)
    raw.dropna(axis=1, how="all", inplace=True)
    if raw.empty:
        return None

    # 2. Find the real header row
    hrow = detect_header_row(raw)

    # 3. Re-read with the correct header row
    df = xl.parse(sheet, header=hrow, dtype=str)
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    if df.empty:
        return None

    # 4. Clean column names
    cols = []
    seen = {}
    for c in df.columns:
        c_str = sanitize(str(c)) if not str(c).lower().startswith("unnamed") else ""
        if not c_str:
            c_str = f"col_{len(cols)}"
        seen[c_str] = seen.get(c_str, 0) + 1
        cols.append(f"{c_str}_{seen[c_str]}" if seen[c_str] > 1 else c_str)
    df.columns = cols

    # 5. Drop rows that are all NaN/empty after parsing
    df = df[df.apply(lambda r: r.str.strip().ne("").any() if r.dtype == object else r.notna().any(), axis=1)]
    df.reset_index(drop=True, inplace=True)

    return df if not df.empty else None


def df_to_text_chunks(df: pd.DataFrame, table_name: str, rows_per_chunk: int = 20) -> list[dict]:
    """
    Convert a DataFrame into text chunks for Qdrant embedding.
    Each chunk is rows_per_chunk rows expressed as key:value prose.
    """
    chunks = []
    cols = list(df.columns)
    for start in range(0, len(df), rows_per_chunk):
        block = df.iloc[start:start + rows_per_chunk]
        lines = [f"Table: {table_name} | Rows {start+1}–{start+len(block)}"]
        for _, row in block.iterrows():
            pairs = ", ".join(
                f"{c}={v}" for c, v in row.items()
                if pd.notna(v) and str(v).strip() not in ("", "nan")
            )
            if pairs:
                lines.append(pairs)
        text = "\n".join(lines)
        chunks.append({"text": text, "page": start // rows_per_chunk + 1})
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    total_tables   = 0
    total_rows_csv = 0
    total_chunks   = 0
    skipped        = []

    for rel_path, prefix in EXCEL_FILES:
        abs_path = os.path.join(DATA_ROOT, rel_path)
        if not os.path.exists(abs_path):
            print(f"[SKIP] Not found: {abs_path}")

        skipped.append(rel_path)
        continue

    engine = "xlrd" if abs_path.endswith(".xls") else "openpyxl"
    try:
        xl = pd.ExcelFile(abs_path, engine=engine)
    except Exception as e:
        print(f"[ERROR] Cannot open {rel_path}: {e}")
        skipped.append(rel_path)
        continue

    print(f"\n{'='*60}")
    print(f"  File   : {os.path.basename(abs_path)}")
    print(f"  Prefix : {prefix}")
    print(f"  Sheets : {xl.sheet_names}")

    for sheet in xl.sheet_names:
        table_name = f"{prefix}__{sanitize(sheet)}"

        # ── Special case: 4th-year student sheet has no header row ──────────
        if prefix == "student_4yr":
            try:
                df = xl.parse(sheet, header=None, dtype=str)
                df.dropna(how="all", inplace=True)
                df.dropna(axis=1, how="all", inplace=True)
                ncols = len(df.columns)
                df.columns = STUDENT_4YR_COLS[:ncols] + [f"extra_{i}" for i in range(ncols - len(STUDENT_4YR_COLS))]
            except Exception as e:
                print(f"  [SKIP] {sheet}: {e}")
                continue
        else:
            df = load_sheet(xl, sheet, prefix)
            if df is None:
                print(f"  [SKIP] {sheet} — empty after cleaning")
                continue

        rows = len(df)
        print(f"\n  ── Sheet: {sheet}")
        print(f"     Table : {table_name}")
        print(f"     Cols  : {list(df.columns)[:8]}{'...' if len(df.columns) > 8 else ''}")
        print(f"     Rows  : {rows}")

        # ── 1. Save CSV ──────────────────────────────────────────────────────
        csv_path = os.path.join(UPLOAD_DIR, f"{table_name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"     CSV   : {csv_path}")
        total_rows_csv += rows

        # ── 2. Embed into Qdrant ─────────────────────────────────────────────
        text_chunks = df_to_text_chunks(df, table_name)
        try:
            embeddings = [generate_query_embedding(c["text"]) for c in text_chunks]
            store_embeddings(
                chunks=text_chunks,
                embeddings=embeddings,
                filename=f"{table_name}.csv",
            )
            print(f"     Qdrant: {len(text_chunks)} chunks embedded ✓")
            total_chunks += len(text_chunks)
        except Exception as e:
            print(f"     Qdrant: FAILED — {e}")

        total_tables += 1

    # ─────────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE")
    print(f"  Tables created : {total_tables}")
    print(f"  CSV rows total : {total_rows_csv}")
    print(f"  Qdrant chunks  : {total_chunks}")
    if skipped:
        print(f"  Skipped files  : {skipped}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

