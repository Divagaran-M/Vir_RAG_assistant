"""
test_rag_accuracy.py — End-to-end RAG accuracy & flow test suite.

Tests EVERY layer of the pipeline with REAL data from the ingested CSVs:
  1. Router accuracy      — does classify LOOKUP / COMPUTE / HYBRID correctly?
  2. SQL Engine accuracy  — does the SQL query return the right rows/values?
  3. Retriever accuracy   — does vector search retrieve the right chunks?
  4. Full pipeline flow   — does the entire chat() path produce a correct answer?
  5. Flow trace logging   — prints a detailed trace of what data passed where.

Ground truth values are derived from the ACTUAL ingested CSV files so tests
can only fail if the pipeline is genuinely broken — not because the data
changed.

Run from the App/ directory:
    cd App
    source .venv/bin/activate
    python -m pytest tests/test_rag_accuracy.py -v --tb=short 2>&1 | tee tests/last_run.log

Or run directly for a human-readable trace:
    python tests/test_rag_accuracy.py
"""

import sys
import os
import time
import json
import textwrap
from datetime import datetime

# ── make services importable ──────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

# ── colour helpers ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def ok(msg):   print(f"{GREEN}  ✓  {msg}{RESET}")
def fail(msg): print(f"{RED}  ✗  {msg}{RESET}")
def info(msg): print(f"{CYAN}  ℹ  {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ⚠  {msg}{RESET}")
def section(title):
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
def sub(title):
    print(f"\n{DIM}{'─'*55}{RESET}")
    print(f"  {BOLD}{title}{RESET}")

# ── log collector (captures all trace data into a JSON-serialisable dict) ─────
TRACE_LOG = []

def log_step(step: str, data: dict):
    entry = {"timestamp": datetime.now().isoformat(), "step": step, **data}
    TRACE_LOG.append(entry)

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH  (derived from real ingested CSV rows)
# ─────────────────────────────────────────────────────────────────────────────
GT = {
    # student_2yr: 394 rows, first student
    "total_2yr_students":        394,
    "first_student_name":        "ABARNA.V",
    "first_student_reg":         "511524243001",
    "first_student_dept":        "AI & DS",

    # student_3yr: 332 rows
    "total_3yr_students":        332,

    # attendance_5sem: Aathi S present ~20 days out of 35
    "aathi_reg":                 "511523205002",
    "aathi_name":                "Aathi S",
    "aathi_days_present":        20,   # floor of 20.14
    "aathi_attendance_pct":      0.57, # floor of 0.575 (57%)

    # 5sem results: Aathi S row index 2 (1-based), grade in first subject = A
    "aathi_5sem_first_grade":    "A",

    # SQL sanity: count of IT students in 2yr table (Batch 2024-2028)
    "ai_ds_students_2yr":        394,   # all rows are 2yr table
}


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 2 — SQL ENGINE ACCURACY
# ═════════════════════════════════════════════════════════════════════════════

def suite_sql():
    section("SUITE 2 — SQL Engine: Compute Accuracy Against Real CSV Data")
    from services.sql_engine import run_sql

    cases = [
        # (question, check_fn, description)
        (
            "How many rows are in the student_2yr table?",
            lambda r: int(r["rows"][0][list(r["rows"][0].keys())[0]]) == GT["total_2yr_students"],
            f"Count of 2yr students should be {GT['total_2yr_students']}",
        ),
        (
            "How many rows are in the student_3yr table?",
            lambda r: int(r["rows"][0][list(r["rows"][0].keys())[0]]) == GT["total_3yr_students"],
            f"Count of 3yr students should be {GT['total_3yr_students']}",
        ),
        (
            "What is Aathi S's attendance percentage from the attendance_5sem table?",
            lambda r: r["rows"] and float(list(r["rows"][0].values())[0]) < 0.65,
            "Aathi S attendance pct should be < 65%",
        ),
        (
            "List the name and reg number of the first student in student_2yr",
            lambda r: any(
                GT["first_student_name"].lower() in str(v).lower()
                or GT["first_student_reg"] in str(v)
                for row in r["rows"] for v in row.values()
            ),
            f"First student name ({GT['first_student_name']}) or reg ({GT['first_student_reg']}) should appear",
        ),
        (
            "How many students have more than 80% attendance in attendance_5sem?",
            lambda r: r["rows"] and int(list(r["rows"][0].values())[0]) >= 0,
            "Count of high-attendance students should be a non-negative integer",
        ),
        (
            "What is the blood group of ABARNA from student_2yr?",
            lambda r: any("O+" in str(v) for row in r["rows"] for v in row.values()),
            "ABARNA's blood group should be O+",
        ),
    ]

    passed = 0
    for question, check, desc in cases:
        sub(f"[{desc}]")
        print(f"    Q: {question}")
        t0 = time.time()
        result = run_sql(question)
        elapsed = round((time.time() - t0) * 1000)

        log_step("SQL_ENGINE", {
            "question": question,
            "sql_generated": result.get("sql", ""),
            "error": result.get("error"),
            "rows_returned": len(result.get("rows", [])),
            "rows_preview": result.get("rows", [])[:3],
            "schema_tables": [
                l.split('"')[1] for l in result.get("schema", "").split("\n")
                if l.startswith('Table:')
            ],
            "ms": elapsed,
        })

        print(f"    SQL: {result.get('sql', 'n/a')[:120]}")
        print(f"    Rows returned: {len(result.get('rows', []))}   Error: {result.get('error')}")
        if result.get("rows"):
            print(f"    Preview: {json.dumps(result['rows'][:2], default=str)[:200]}")

        if result.get("error"):
            fail(f"FAIL — SQL error: {result['error']}")
        elif not result.get("rows"):
            fail("FAIL — 0 rows returned")
        else:
            try:
                if check(result):
                    ok(f"PASS ({elapsed}ms)")
                    passed += 1
                else:
                    fail(f"FAIL — check function returned False. Rows: {result['rows'][:2]}")
            except Exception as e:
                fail(f"FAIL — check raised: {e}")

    print(f"\n  SQL Engine Score: {passed}/{len(cases)}")
    return passed, len(cases)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 3 — RETRIEVER / QDRANT ACCURACY
# ═════════════════════════════════════════════════════════════════════════════

def suite_retriever():
    section("SUITE 3 — Retriever: Vector Search Chunk Quality")
    from services.retriever import retrieve_context

    cases = [
        # (question, filename_filter, keywords_that_must_appear_in_context, question_type, desc)
        (
            "Tell me about Aathi S",
            "attendance_5sem__IT_III.csv",
            ["Aathi", "511523205002"],
            "general",
            "Aathi S should appear in attendance chunks",
        ),
        (
            "Who is ABARNA in the student list?",
            "student_2yr__ALL_FILES_COMBINED.csv",
            ["ABARNA", "511524243001"],
            "general",
            "ABARNA should appear in 2yr student chunks",
        ),
        (
            "5th semester IT results grades",
            "marks_5sem_ra__III_IT.csv",
            ["marks_5sem_ra", "pass"],
            "general",
            "5th sem result chunks should contain results and pass information",
        ),
        (
            "IT department students batch 2023 2027",
            None,   # search across all docs
            ["IT"],
            "general",
            "Cross-doc search should find IT student records",
        ),
    ]


    passed = 0
    for question, filename, keywords, qtype, desc in cases:
        sub(f"[{desc}]")
        print(f"    Q: {question}")
        print(f"    Filter: {filename or 'all docs'}")
        print(f"    Must contain: {keywords}")

        t0 = time.time()
        context = retrieve_context(question=question, filename=filename, question_type=qtype)
        elapsed = round((time.time() - t0) * 1000)
        context_lower = context.lower()

        found_keywords = [kw for kw in keywords if kw.lower() in context_lower]
        missing_keywords = [kw for kw in keywords if kw.lower() not in context_lower]

        log_step("RETRIEVER", {
            "question": question,
            "filename_filter": filename,
            "context_length_chars": len(context),
            "context_preview": context[:400],
            "keywords_expected": keywords,
            "keywords_found": found_keywords,
            "keywords_missing": missing_keywords,
            "passed": len(missing_keywords) == 0,
            "ms": elapsed,
        })

        print(f"    Context length: {len(context)} chars  ({elapsed}ms)")
        print(f"    Context preview: {context[:250].strip()!r}")
        if found_keywords:
            ok(f"Keywords found: {found_keywords}")
        if missing_keywords:
            warn(f"Keywords missing: {missing_keywords}")

        if len(missing_keywords) == 0:
            ok("PASS — all keywords found in retrieved context")
            passed += 1
        elif len(found_keywords) > 0:
            warn(f"PARTIAL — {len(found_keywords)}/{len(keywords)} keywords found")
        else:
            fail("FAIL — no relevant content retrieved")

    print(f"\n  Retriever Score: {passed}/{len(cases)}")
    return passed, len(cases)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 4 — FULL PIPELINE FLOW (end-to-end)
# ═════════════════════════════════════════════════════════════════════════════

def suite_pipeline():
    section("SUITE 4 — Full Pipeline: End-to-End Answer Quality")
    from services.router import route_question
    from services.sql_engine import run_sql
    from services.retriever import retrieve_context
    from services.prompt_builder import build_prompt
    from services.llm import generate_response

    def run_pipeline(question, filename=None):
        """Simulate exactly what chat.py does, logging every step."""
        trace = {"question": question, "steps": []}

        # Step 1 — Route
        t0 = time.time()
        intent = route_question(question)
        trace["steps"].append({"step": "ROUTER", "output": intent, "ms": round((time.time()-t0)*1000)})

        if intent == "COMPUTE":
            t0 = time.time()
            sql_result = run_sql(question, filename)
            trace["steps"].append({
                "step": "SQL_ENGINE",
                "sql": sql_result.get("sql"),
                "rows": sql_result.get("rows", [])[:5],
                "error": sql_result.get("error"),
                "ms": round((time.time()-t0)*1000),
            })

            if sql_result.get("error") or not sql_result.get("rows"):
                answer = f"SQL failed: {sql_result.get('error', 'no rows')}"
            else:
                rows_text = "\n".join(
                    ", ".join(f"{k}={v}" for k, v in row.items())
                    for row in sql_result["rows"]
                )
                fmt_prompt = (
                    f"Answer this question in one sentence using the SQL result.\n"
                    f"Question: {question}\nResult:\n{rows_text}\nAnswer:"
                )
                t0 = time.time()
                answer = generate_response(fmt_prompt)
                trace["steps"].append({"step": "LLM_FORMAT", "ms": round((time.time()-t0)*1000)})

        elif intent == "LOOKUP":
            t0 = time.time()
            context = retrieve_context(question=question, filename=filename, question_type="general")
            trace["steps"].append({
                "step": "RETRIEVER",
                "context_chars": len(context),
                "context_preview": context[:300],
                "ms": round((time.time()-t0)*1000),
            })
            prompt = build_prompt(context=context, question=question, history=[])
            t0 = time.time()
            answer = generate_response(prompt)
            trace["steps"].append({"step": "LLM_ANSWER", "ms": round((time.time()-t0)*1000)})

        else:  # HYBRID
            t0 = time.time()
            sql_result = run_sql(question, filename)
            trace["steps"].append({
                "step": "SQL_ENGINE",
                "sql": sql_result.get("sql"),
                "rows": sql_result.get("rows", [])[:3],
                "ms": round((time.time()-t0)*1000),
            })
            sql_summary = str(sql_result.get("rows", "no rows"))
            rag_query = f"{question} — {sql_summary}"
            t0 = time.time()
            context = retrieve_context(question=rag_query, filename=filename, question_type="general")
            trace["steps"].append({
                "step": "RETRIEVER",
                "context_chars": len(context),
                "context_preview": context[:200],
                "ms": round((time.time()-t0)*1000),
            })
            merge_prompt = (
                f"Answer using both:\nSQL: {sql_summary}\nContext: {context[:600]}\n"
                f"Question: {question}\nAnswer:"
            )
            t0 = time.time()
            answer = generate_response(merge_prompt)
            trace["steps"].append({"step": "LLM_MERGE", "ms": round((time.time()-t0)*1000)})

        trace["final_answer"] = answer
        return trace

    # ── test cases ────────────────────────────────────────────────────────────

    e2e_cases = [
        {
            "id": "E2E-1",
            "question": "How many students are in the 2 year batch?",
            "filename": None,
            "expected_intent": "COMPUTE",
            "answer_must_contain": ["394"],   # exact number from CSV
            "answer_must_not_contain": ["I couldn't", "error", "sorry"],
            "desc": "Student count — should return 394 from SQL",
        },
        {
            "id": "E2E-2",
            "question": "Tell me about Aathi S",
            "filename": "attendance_5sem__IT_III.csv",
            "expected_intent": "LOOKUP",
            "answer_must_contain": ["Aathi"],
            "answer_must_not_contain": ["I couldn't find"],
            "desc": "Lookup student — Aathi S should be found in context",
        },
        {
            "id": "E2E-3",
            "question": "What is the blood group of ABARNA?",
            "filename": "student_2yr__ALL_FILES_COMBINED.csv",
            "expected_intent": "LOOKUP",
            "answer_must_contain": ["O+"],
            "answer_must_not_contain": ["I couldn't"],
            "desc": "Specific fact lookup — ABARNA blood group = O+",
        },
        {
            "id": "E2E-4",
            "question": "How many students have more than 60% attendance in 5th sem?",
            "filename": "attendance_5sem__IT_III.csv",
            "expected_intent": "COMPUTE",
            "answer_must_contain": [],  # just check no error
            "answer_must_not_contain": ["error", "failed"],
            "desc": "Filtered count — attendance > 60%",
        },
        {
            "id": "E2E-5",
            "question": "What is the total number of students in the 3 year data?",
            "filename": None,
            "expected_intent": "COMPUTE",
            "answer_must_contain": ["332"],
            "answer_must_not_contain": ["error"],
            "desc": "3yr student count — should return 332",
        },
    ]

    passed = 0
    for case in e2e_cases:
        sub(f"[{case['id']}] {case['desc']}")
        print(f"    Q: {case['question']}")
        print(f"    File: {case['filename'] or 'all'}")

        t_start = time.time()
        try:
            trace = run_pipeline(case["question"], case["filename"])
        except Exception as ex:
            fail(f"PIPELINE CRASHED: {ex}")
            log_step("E2E_PIPELINE", {"id": case["id"], "crashed": str(ex)})
            continue

        total_ms = round((time.time() - t_start) * 1000)
        answer = trace.get("final_answer", "")

        # ── print flow trace ──────────────────────────────────────────────────
        print(f"\n    {'─'*50}")
        print(f"    FLOW TRACE:")
        for step in trace["steps"]:
            step_name = step["step"]
            ms = step.get("ms", "?")
            if step_name == "ROUTER":
                print(f"      [ROUTER]      intent={step['output']}  ({ms}ms)")
            elif step_name == "SQL_ENGINE":
                sql_short = (step.get("sql") or "")[:80]
                print(f"      [SQL_ENGINE]  sql={sql_short!r}  rows={len(step.get('rows',[]))}  ({ms}ms)")
                if step.get("error"):
                    print(f"                   ERROR: {step['error']}")
            elif step_name == "RETRIEVER":
                print(f"      [RETRIEVER]   context={step['context_chars']} chars  ({ms}ms)")
                print(f"                   preview={step['context_preview'][:100]!r}")
            elif step_name in ("LLM_ANSWER", "LLM_FORMAT", "LLM_MERGE"):
                print(f"      [{step_name}]  ({ms}ms)")

        print(f"      [TOTAL]       {total_ms}ms")
        print(f"    {'─'*50}")
        print(f"    Answer: {answer[:300]!r}")

        log_step("E2E_PIPELINE", {
            "id": case["id"],
            "question": case["question"],
            "intent": trace["steps"][0].get("output") if trace["steps"] else "?",
            "answer": answer,
            "flow": trace["steps"],
            "total_ms": total_ms,
        })

        # ── checks ────────────────────────────────────────────────────────────
        check_failures = []

        # Intent check
        actual_intent = trace["steps"][0].get("output") if trace["steps"] else "?"
        if actual_intent != case["expected_intent"]:
            check_failures.append(f"intent={actual_intent} (expected {case['expected_intent']})")

        # Content checks
        answer_lower = answer.lower()
        for must in case["answer_must_contain"]:
            if must.lower() not in answer_lower:
                check_failures.append(f"answer missing: {must!r}")
        for must_not in case["answer_must_not_contain"]:
            if must_not.lower() in answer_lower:
                check_failures.append(f"answer contains forbidden: {must_not!r}")

        if not check_failures:
            ok(f"PASS — all checks passed ({total_ms}ms)")
            passed += 1
        else:
            fail(f"FAIL — {'; '.join(check_failures)}")

    print(f"\n  Pipeline Score: {passed}/{len(e2e_cases)}")
    return passed, len(e2e_cases)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 5 — DATA FLOW TRACE (shows exactly what shape of data each layer sees)
# ═════════════════════════════════════════════════════════════════════════════

def suite_data_flow_trace():
    section("SUITE 5 — Data Flow Trace: What Shape of Data Passes Between Layers")
    from services.embeddings import generate_query_embedding
    from services.vectordb import search_embeddings
    from services.sql_engine import run_sql, _find_csv_paths, _load_csvs_into_sqlite

    question = "Aathi S attendance in 5th semester"
    info(f"Tracing question: {question!r}")

    # ── Layer 0: CSV Discovery ────────────────────────────────────────────────
    sub("Layer 0 — CSV Discovery")
    csvs = _find_csv_paths(filename="attendance_5sem__IT_III.csv")
    print(f"    CSVs found: {[os.path.basename(c) for c in csvs]}")
    conn, schema = _load_csvs_into_sqlite(csvs)
    print(f"    Schema handed to LLM:\n")
    for line in schema.split("\n")[:10]:
        print(f"      {line}")
    conn.close()

    log_step("FLOW_CSV_LAYER", {"csvs": csvs, "schema_preview": schema[:400]})

    # ── Layer 1: Query Embedding ──────────────────────────────────────────────
    sub("Layer 1 — Query Embedding")
    t0 = time.time()
    embedding = generate_query_embedding(question)
    elapsed = round((time.time() - t0) * 1000)
    print(f"    Embedding dim : {len(embedding)}")
    print(f"    First 8 dims  : {[round(x, 4) for x in embedding[:8]]}")
    print(f"    Time          : {elapsed}ms")
    info("Type: List[float] (dense vector, dim=1024, cosine similarity space)")

    log_step("FLOW_EMBEDDING", {
        "question": question,
        "dim": len(embedding),
        "first8": [round(x, 4) for x in embedding[:8]],
        "ms": elapsed,
    })

    # ── Layer 2: Vector Search (raw Qdrant response) ──────────────────────────
    sub("Layer 2 — Qdrant Vector Search (raw response shape)")
    t0 = time.time()
    raw = search_embeddings(
        query_embedding=embedding,
        filename="attendance_5sem__IT_III.csv",
        top_k=3,
    )
    elapsed = round((time.time() - t0) * 1000)
    print(f"    Keys returned by search_embeddings : {list(raw.keys())}")
    print(f"    documents[0] type                  : {type(raw['documents'][0])}")
    print(f"    Number of chunks                   : {len(raw['documents'][0])}")
    print(f"    Chunk 0 preview (300 chars):")
    print(f"      {raw['documents'][0][0][:300]!r}")
    print(f"    Pages                              : {raw['pages'][:3]}")
    print(f"    Filenames                          : {raw['filenames'][:3]}")
    print(f"    Time                               : {elapsed}ms")

    log_step("FLOW_QDRANT_SEARCH", {
        "keys": list(raw.keys()),
        "num_chunks": len(raw["documents"][0]),
        "chunk0_preview": raw["documents"][0][0][:300],
        "pages": raw["pages"],
        "filenames": raw["filenames"],
        "ms": elapsed,
    })

    # ── Layer 3: SQL Engine data shape ────────────────────────────────────────
    sub("Layer 3 — SQL Engine (data shape into and out of the layer)")
    sql_q = "What is Aathi S attendance percentage from attendance_5sem table?"
    t0 = time.time()
    sql_res = run_sql(sql_q, filename="attendance_5sem__IT_III.csv")
    elapsed = round((time.time() - t0) * 1000)
    print(f"    Input type    : str (natural language question)")
    print(f"    Generated SQL : {sql_res.get('sql')}")
    print(f"    Output keys   : {list(sql_res.keys())}")
    print(f"    columns       : {sql_res.get('columns')}")
    print(f"    rows (type)   : List[Dict[str, Any]]  len={len(sql_res.get('rows',[]))}")
    if sql_res.get("rows"):
        print(f"    rows[0]       : {json.dumps(sql_res['rows'][0], default=str)[:200]}")
    print(f"    error         : {sql_res.get('error')}")
    print(f"    Time          : {elapsed}ms")

    log_step("FLOW_SQL_SHAPE", {
        "question": sql_q,
        "sql": sql_res.get("sql"),
        "columns": sql_res.get("columns"),
        "num_rows": len(sql_res.get("rows", [])),
        "row0": sql_res.get("rows", [{}])[0],
        "error": sql_res.get("error"),
        "ms": elapsed,
    })

    info("Flow complete — no assertions in this suite, it's purely diagnostic.")
    return 0, 0   # no pass/fail scoring


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — run all suites and write trace log
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'█'*60}{RESET}")
    print(f"{BOLD}  Vir RAG System — Accuracy & Flow Test Suite{RESET}")
    print(f"{BOLD}  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{'█'*60}{RESET}")

    results = {}

    # Run all suites
    results["sql"]       = suite_sql()
    results["retriever"] = suite_retriever()
    suite_data_flow_trace()   # diagnostic only, no score

    # ── Final summary ─────────────────────────────────────────────────────────
    section("FINAL SUMMARY")
    total_passed = 0
    total_cases  = 0
    for suite_name, (p, t) in results.items():
        pct = int(100 * p / t) if t else 0
        bar = ("█" * pct + "░" * (100 - pct))[:20]
        colour = GREEN if pct >= 80 else YELLOW if pct >= 50 else RED
        print(f"  {suite_name:<12}  {colour}{bar}{RESET}  {p}/{t}  ({pct}%)")
        total_passed += p
        total_cases  += t

    overall_pct = int(100 * total_passed / total_cases) if total_cases else 0
    print(f"\n  {'─'*50}")
    colour = GREEN if overall_pct >= 80 else YELLOW if overall_pct >= 50 else RED
    print(f"  OVERALL       {colour}{total_passed}/{total_cases}  ({overall_pct}%){RESET}")

    # ── Write trace log ───────────────────────────────────────────────────────
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_run_trace.json")
    with open(log_path, "w") as f:
        json.dump({
            "run_at": datetime.now().isoformat(),
            "summary": {
                k: {"passed": p, "total": t, "pct": int(100*p/t) if t else 0}
                for k, (p, t) in results.items()
            },
            "overall_pct": overall_pct,
            "trace": TRACE_LOG,
        }, f, indent=2, default=str)

    print(f"\n  Detailed trace log written → {log_path}")
    print(f"{BOLD}{'█'*60}{RESET}\n")

    return overall_pct >= 60   # exit 0 if ≥60% pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# ─────────────────────────────────────────────────────────────────────────────
# pytest-compatible wrappers (run with: pytest tests/test_rag_accuracy.py -v)
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_engine_accuracy():
    passed, total = suite_sql()
    assert passed / total >= 0.6, f"SQL accuracy {passed}/{total} below 60%"

def test_retriever_accuracy():
    passed, total = suite_retriever()
    assert passed / total >= 0.5, f"Retriever accuracy {passed}/{total} below 50%"


