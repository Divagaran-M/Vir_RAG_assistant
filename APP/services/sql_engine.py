"""
SQL Engine -- The COMPUTE branch of the RAG pipeline.

Executes natural-language questions against the unified SQLite database:
  App/data/app.db (Consolidated modular schema + views)

Key Features:
  1. Ultra-Compact schema_master loader: Only object_name and column_name (no datatypes, no samples, no descriptions)
  2. Zero-Token Direct Fast-Path: Deterministic queries (e.g. 12-digit register numbers, student names) execute instantly
  3. Self-Correction Loop (Up to 5 Retries): If generated SQL errors, feed error message back to the LLM to auto-correct
"""

import os
import re
import glob
import sqlite3
import csv as csv_module

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, UPLOAD_FOLDER


def _get_client():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. Please add it to your .env file."
        )
    return Groq(api_key=GROQ_API_KEY)


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(APP_DIR, "data", "app.db")


# ---------------------------------------------------------------------------
# Schema Master Loader -- Ultra-Compact (Column Names Only)
# ---------------------------------------------------------------------------

def load_schema_context(conn: sqlite3.Connection, question: str = "") -> str:
    """
    Query schema_master to build an ultra-compact schema string containing ONLY
    table/view names and their column names.
    No datatypes, no sample values, no descriptions.
    Size: ~700-900 chars (< 250 tokens).
    """
    try:
        c = conn.cursor()
        objects = c.execute("""
            SELECT DISTINCT object_name, object_type
            FROM schema_master
            ORDER BY object_type DESC, object_name
        """).fetchall()

        if not objects:
            return _introspect_schema(conn)

        parts = []
        for obj_name, obj_type in objects:
            cols = c.execute("""
                SELECT column_name
                FROM schema_master
                WHERE object_name = ?
                ORDER BY id
            """, (obj_name,)).fetchall()

            col_names = [col[0] for col in cols]
            parts.append(f"{'TABLE' if obj_type == 'table' else 'VIEW'} {obj_name} ({', '.join(col_names)})")

        return "\n".join(parts)

    except Exception as e:
        print(f"[SQLEngine] schema_master query failed ({e}), falling back to introspection")
        return _introspect_schema(conn)


def _introspect_schema(conn: sqlite3.Connection) -> str:
    """Fallback: build compact schema from sqlite_master if schema_master is missing."""
    c = conn.cursor()
    objects = c.execute("""
        SELECT name, type FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT IN ('sqlite_sequence', 'schema_master')
        ORDER BY type DESC, name
    """).fetchall()

    parts = []
    for name, obj_type in objects:
        cols = c.execute(f"PRAGMA table_info({name})").fetchall()
        col_names = [col[1] for col in cols]
        parts.append(f"{'TABLE' if obj_type == 'table' else 'VIEW'} {name} ({', '.join(col_names)})")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_table_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    if re.match(r"^[0-9a-fA-F]{32}_", base):
        base = base[33:]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = f"t_{name}"
    return name or "data_table"


def _load_csvs_into_sqlite(csv_paths: list) -> tuple:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema_parts = []

    for path in csv_paths:
        table_name = _sanitize_table_name(path)
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue
                columns = list(rows[0].keys())

            cols_sql = ", ".join(f'"{c}" TEXT' for c in columns)
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_sql})')

            placeholders = ", ".join("?" for _ in columns)
            col_list = ", ".join(f'"{c}"' for c in columns)
            conn.executemany(
                f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})',
                [tuple(row.get(c, "") for c in columns) for row in rows],
            )
            conn.commit()
            schema_parts.append(f'TABLE "{table_name}" ({", ".join(columns)})')
        except Exception as e:
            print(f"[SQLEngine] Skipping {path}: {e}")

    schema_text = "\n".join(schema_parts) if schema_parts else ""
    return conn, schema_text


def _extract_sql(text: str) -> str:
    code_block = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block:
        sql = code_block.group(1).strip()
    else:
        select_match = re.search(r"(SELECT\b[\s\S]+)", text, re.IGNORECASE)
        sql = select_match.group(1).strip() if select_match else text.strip()

    sql = re.sub(r";\s*$", "", sql).strip()

    lines = []
    for line in sql.split("\n"):
        stripped = line.strip()
        if stripped.startswith("--") or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if stripped:
            lines.append(line)

    return "\n".join(lines).strip()


def _try_direct_sql(question: str) -> str:
    """
    Fast, zero-LLM SQL generator for unambiguous deterministic lookups.
    Covers:
      - 12-digit registration numbers → students table
      - Faculty name / Prof. / Dr. lookups → faculty table
      - Designation lookups (HOD, Principal, Director, etc.) → faculty table
      - Department-wise faculty list → faculty table
      - 'who is <Name>' → students OR faculty table
    """
    q = question.strip()
    q_lower = q.lower()

    # 1. 12-digit Register Number lookup
    reg_match = re.search(r"\b(5\d{11})\b", q)
    if reg_match:
        reg_no = reg_match.group(1)
        if any(w in q_lower for w in ["subject", "course", "enrolled"]):
            return f"SELECT course_code, course_title, semester, exam_type FROM student_assessments WHERE reg_no = '{reg_no}' GROUP BY course_code"
        if any(w in q_lower for w in ["mark", "score", "grade", "iat", "exam"]):
            return f"SELECT course_code, course_title, exam_type, score_numeric, grade, is_arrear FROM student_assessments WHERE reg_no = '{reg_no}'"
        if any(w in q_lower for w in ["attendance", "present", "absent", "eligible"]):
            return f"SELECT course_code, course_title, total_classes_conducted, classes_attended, attendance_percentage, exam_eligibility_status FROM attendance WHERE reg_no = '{reg_no}'"
        return f"SELECT * FROM students WHERE reg_no = '{reg_no}'"

    # ── Faculty fast-paths ──────────────────────────────────────────────────────

    _DESIGNATIONS = [
        "hod", "head of department", "principal", "director", "dean",
        "warden", "librarian", "counsellor", "counselor",
        "associate professor", "assistant professor", "professor",
        "placement officer", "lab in-charge", "lab incharge",
        "class incharge", "class advisor", "coordinator",
    ]
    designation_found = next((d for d in _DESIGNATIONS if d in q_lower), None)

    # 2. Designation-based lookup: "who is the HOD of CSE", "contact of principal",
    #    "phone number of the director", "find the warden"
    desig_match = re.search(
        r"\b(who is|find|get|show|tell me about|contact of|phone of|phone number of|"
        r"email of|cabin of|room of|details of|number of|contact)\b"
        r".{0,50}"
        r"\b(hod|head of department|principal|director|dean|warden|librarian|counsellor|counselor|"
        r"professor|associate professor|assistant professor|placement officer|class advisor|coordinator)\b",
        q_lower
    )
    if desig_match and designation_found:
        dept_match = re.search(r"\b(cse|it|ece|eee|mech|civil|ai|aids)\b", q_lower)
        if dept_match:
            dept = dept_match.group(1).upper()
            return (
                f"SELECT faculty_name, designation, department, phone_primary, email, room_cabin_no "
                f"FROM faculty WHERE LOWER(designation) LIKE LOWER('%{designation_found}%') "
                f"AND LOWER(department) LIKE LOWER('%{dept}%')"
            )
        return (
            f"SELECT faculty_name, designation, department, phone_primary, email, room_cabin_no "
            f"FROM faculty WHERE LOWER(designation) LIKE LOWER('%{designation_found}%')"
        )

    # 3. "list all faculty/staff in <dept>" / "show professors in ECE" / "show all ECE faculty"
    #    Two patterns: dept AFTER keyword ("faculty in ECE") OR dept BEFORE ("ECE faculty")
    _DEPT_RE = r"\b(cse|it|ece|eee|mech|civil|ai|aids)\b"
    _FAC_WORD = r"\b(faculty|staff|professors?|teachers?)\b"
    _LIST_VERB = r"\b(list|show|give|get|find|all)\b"
    list_faculty_match = (
        re.search(rf"{_LIST_VERB}.{{0,20}}{_FAC_WORD}.{{0,30}}{_DEPT_RE}", q_lower)
        or re.search(rf"{_LIST_VERB}.{{0,20}}{_DEPT_RE}.{{0,20}}{_FAC_WORD}", q_lower)
        or re.search(rf"{_DEPT_RE}.{{0,20}}{_FAC_WORD}", q_lower)  # bare "ECE faculty" / "IT staff"
    )
    # Only trigger if there's actually a dept keyword present
    dept_m = re.search(_DEPT_RE, q_lower)
    if list_faculty_match and dept_m:
        dept = dept_m.group(1).upper()
        return (
            f"SELECT faculty_name, designation, department, phone_primary, email, room_cabin_no "
            f"FROM faculty WHERE LOWER(department) LIKE LOWER('%{dept}%') ORDER BY designation"
        )

    # 4. Faculty name lookup with Prof./Dr. prefix: "find Dr. Kumar", "who is Prof. Priya"
    #    Guard: skip if query contains list/all/dept keywords (already handled above)
    _has_list_signal = any(w in q_lower for w in ["all ", "list ", "show all", "every"])
    faculty_name_match = re.search(
        r"\b(who is|find|search|get|show|fetch|contact of|phone of|email of|cabin of|room of|details of|info of|profile of)\b"
        r"\s+(?:prof\.?|dr\.?|mr\.?|mrs\.?|ms\.?)?\s*([A-Za-z][A-Za-z\s\.]{2,40}?)(?:\s+from|\s+in|\s+of|\s*$)",
        q, re.IGNORECASE
    )
    if (
        faculty_name_match
        and not _has_list_signal
        and any(w in q_lower for w in [
            "prof", "dr.", "dr ", "faculty", "staff", "teacher", "professor", "hod", "sir", "madam"
        ])
    ):
        name_query = faculty_name_match.group(2).strip().rstrip(".")
        # Don't use as name if it looks like a dept word
        bad_words = {"all", "faculty", "staff", "cse", "it", "ece", "eee", "mech", "civil", "ai"}
        if len(name_query) >= 2 and name_query.lower() not in bad_words:
            return (
                f"SELECT faculty_name, designation, department, qualification, phone_primary, email, room_cabin_no "
                f"FROM faculty WHERE LOWER(faculty_name) LIKE LOWER('%{name_query}%')"
            )

    # 5. Simple 'who is <Name>' — try faculty first if Prof/Dr is mentioned, else students
    who_match = re.search(r"^who is\s+([A-Za-z\s\.]+?)(?:\s+from|\s+in|\s*$)", q, re.IGNORECASE)
    if who_match:
        name_query = who_match.group(1).strip()
        if len(name_query) >= 3 and not any(w in name_query.lower() for w in ["the", "this", "that", "faculty", "student"]):
            if any(p in q_lower for p in ["prof", "dr.", "dr ", "professor"]):
                return (
                    f"SELECT faculty_name, designation, department, phone_primary, email, room_cabin_no "
                    f"FROM faculty WHERE LOWER(faculty_name) LIKE LOWER('%{name_query}%')"
                )
            return f"SELECT * FROM students WHERE LOWER(student_name) LIKE LOWER('%{name_query}%')"

    return ""


def _generate_sql(schema_text: str, question: str, previous_error: str = None, previous_sql: str = None) -> str:
    """
    Generate SQLite query from schema (column names only).
    If previous_error is provided, the prompt instructs the model to self-correct.
    """
    retry_context = ""
    if previous_error and previous_sql:
        retry_context = (
            f"\nATTENTION: Your previous query failed with this SQLite error:\n"
            f"Failed SQL: {previous_sql}\n"
            f"Error message: {previous_error}\n"
            f"Please fix the error and generate a corrected SQL query.\n"
        )

    prompt = (
        "You are an expert SQLite query generator for a college campus assistant.\n\n"
        "Generate a valid SQLite SELECT query to answer the question using ONLY the provided tables, views, and column names.\n\n"
        "DATABASE SCHEMA (Tables, Views & Columns only):\n"
        f"{schema_text}\n"
        f"{retry_context}\n"
        "RULES:\n"
        "1. Output ONLY the raw SQL query. No markdown formatting, no explanations, no trailing semicolon.\n"
        "2. Use column names EXACTLY as listed in the schema.\n"
        "3. Use case-insensitive LIKE or LOWER() for names and departments (e.g. LOWER(student_name) LIKE '%aathi%').\n"
        "4. Available views for convenience: view_student_performance_summary, view_exam_subject_analytics, view_student_complete_profile.\n"
        "5. If question cannot be answered, output: SELECT 'NO_DATA'\n"
        "6. FACULTY RULES — the 'faculty' table stores all staff/professors/admin:\n"
        "   - Columns: faculty_id, faculty_name, qualification, designation, department, phone_primary, phone_secondary, email, room_cabin_no, class_incharge_role\n"
        "   - For 'who is the HOD of X': WHERE LOWER(designation) LIKE '%hod%' AND LOWER(department) LIKE '%X%'\n"
        "   - For 'who is the principal/director': WHERE LOWER(designation) LIKE '%principal%' (or '%director%')\n"
        "   - For faculty by name (Prof. X / Dr. X): WHERE LOWER(faculty_name) LIKE '%X%'\n"
        "   - For contact/phone/email/cabin: SELECT faculty_name, designation, phone_primary, email, room_cabin_no FROM faculty WHERE ...\n\n"
        f"QUESTION: {question}\n\n"
        "SQL:"
    )

    try:
        resp = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )
        content = resp.choices[0].message.content or ""
        return _extract_sql(content)
    except Exception as e:
        raise RuntimeError(f"LLM SQL generation failed: {e}")



# Regex that catches any destructive keyword anywhere in the SQL (blocks bypass tricks
# like "WITH t AS (DELETE ...) SELECT ...").
_DESTRUCTIVE_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE"
    r"|ATTACH|DETACH|VACUUM|REINDEX|ANALYZE"
    r"|PRAGMA\s+(?!table_info|index_list|index_info|foreign_key_list|table_xinfo))\b",
    re.IGNORECASE,
)



# Regex that catches any destructive keyword anywhere in the SQL (blocks bypass tricks
# like "WITH t AS (DELETE ...) SELECT ...").
_DESTRUCTIVE_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE"
    r"|ATTACH|DETACH|VACUUM|REINDEX|ANALYZE"
    r"|PRAGMA\s+(?!table_info|index_list|index_info|foreign_key_list|table_xinfo))\b",
    re.IGNORECASE,
)


def _safe_execute(conn: sqlite3.Connection, sql: str) -> dict:
    """
    Safely execute a SELECT or WITH … SELECT query against SQLite.

    Guards (layered):
      1. First token must be SELECT or WITH.
      2. Full SQL is scanned for any destructive keyword (blocks CTE bypass tricks).
      3. Connection is put into query_only mode before execution.
    """
    stripped = sql.strip()

    # Guard 1 — first token
    first_token = stripped.split()[0].upper() if stripped else ""
    if first_token not in ("SELECT", "WITH"):
        return {
            "sql": sql,
            "columns": [],
            "rows": [],
            "error": "Only SELECT queries are permitted.",
        }

    # Guard 2 — destructive keyword anywhere in the statement
    if _DESTRUCTIVE_SQL_RE.search(stripped):
        return {
            "sql": sql,
            "columns": [],
            "rows": [],
            "error": "Query contains a disallowed keyword and was blocked.",
        }

    try:
        # Guard 3 — set read-only mode at the connection level
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.execute(stripped)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"sql": stripped, "columns": columns, "rows": rows, "error": None}
    except sqlite3.Error as e:
        return {"sql": stripped, "columns": [], "rows": [], "error": str(e)}



# ---------------------------------------------------------------------------
# Public API with 5-Iteration Self-Correction Retry Loop
# ---------------------------------------------------------------------------

def run_sql(question: str, filename: str = None, max_retries: int = 5) -> dict:
    """
    Execute a natural-language COMPUTE question against the SQLite database.
    Includes a self-correction loop of up to `max_retries` (default 5 iterations)
    if SQLite returns an execution error.
    """
    print("\n========== SQL ENGINE ==========")
    print(f"Question : {question}")
    print(f"Filename : {filename or 'Consolidated SQLite (app.db)'}")

    # ?? Fast Check: Direct Zero-Token Regex ???????????????????????????????????
    direct_sql = _try_direct_sql(question)
    if direct_sql and not filename:
        print(f"[SQLEngine] Direct SQL match: {direct_sql}")
        if not os.path.exists(DB_PATH):
            import subprocess
            subprocess.run(["python3", os.path.join(APP_DIR, "ingest_sqlite.py")], check=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            result = _safe_execute(conn, direct_sql)
        finally:
            conn.close()
        result["schema"] = "(fast-path)"
        print(f"Result rows  : {len(result.get('rows', []))}")
        print("================================\n")
        return result

    # ?? Case A: Custom CSV filename provided (legacy upload mode) ?????????????
    if filename and not filename.endswith(".db"):
        pattern = os.path.join(UPLOAD_FOLDER, f"*_{filename}")
        matches = glob.glob(pattern)
        if not matches:
            direct = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(direct):
                matches = [direct]
        csv_paths = [m for m in matches if m.endswith(".csv")]

        if csv_paths:
            csv_conn, schema_text = _load_csvs_into_sqlite(csv_paths)
            last_error = None
            last_sql = None
            result = {"sql": "", "columns": [], "rows": [], "error": "No query generated"}

            try:
                for attempt in range(1, max_retries + 1):
                    sql = _generate_sql(schema_text, question, previous_error=last_error, previous_sql=last_sql)
                    print(f"Generated SQL (CSV attempt {attempt}/{max_retries}): {sql}")
                    result = _safe_execute(csv_conn, sql)
                    if not result.get("error"):
                        break
                    last_error = result["error"]
                    last_sql = sql
                    print(f"[SQLEngine] SQL Error on attempt {attempt}: {last_error} -> Retrying...")
            finally:
                csv_conn.close()

            result["schema"] = schema_text
            print(f"Result rows  : {len(result.get('rows', []))}")
            print("================================\n")
            return result

    # ?? Case B: Primary Unified Database (app.db) ?????????????????????????????
    if not os.path.exists(DB_PATH):
        import subprocess
        print("[SQLEngine] app.db not found, building via ingest_sqlite.py...")
        subprocess.run(["python3", os.path.join(APP_DIR, "ingest_sqlite.py")], check=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    schema_text = load_schema_context(conn, question)
    print(f"[SQLEngine] Ultra-compact schema context loaded ({len(schema_text)} chars)")

    last_error = None
    last_sql = None
    result = {"sql": "", "columns": [], "rows": [], "error": "No query generated"}

    try:
        # 5-Loop Self-Correction Execution
        for attempt in range(1, max_retries + 1):
            sql = _generate_sql(schema_text, question, previous_error=last_error, previous_sql=last_sql)
            print(f"Generated SQL (Attempt {attempt}/{max_retries}): {sql}")
            
            result = _safe_execute(conn, sql)
            
            # If query succeeded without error, exit loop
            if not result.get("error"):
                break
                
            last_error = result["error"]
            last_sql = sql
            print(f"[SQLEngine] SQL Error on attempt {attempt}: {last_error} -> Retrying with error context...")

    except Exception as e:
        result = {"sql": "", "columns": [], "rows": [], "error": str(e)}
    finally:
        conn.close()

    result["schema"] = schema_text
    print(f"Result rows  : {len(result.get('rows', []))}")
    print("================================\n")

    return result

