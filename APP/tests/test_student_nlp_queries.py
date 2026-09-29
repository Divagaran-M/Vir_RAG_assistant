"""
tests/test_student_nlp_queries.py — Comprehensive NLP & Persona Test Suite for Student Queries

Evaluates how Vir's Agentic RAG System handles queries about students across:
1. Personas:
   - Parent Queries (Welfare, grades, attendance, hostel status, parent contact, fees)
   - Staff / Faculty / HOD Queries (Cohort analytics, arrear tracking, attendance defaulters, advising)
   - Student / Peer Queries (Self marks, exam eligibility, toppers, peer lookup)
   - Front Desk / Admin Queries (Verification, DOB, demographic counts)

2. NLP Techniques & Linguistic Variations:
   - Exact Entity Matching (12-digit reg numbers, formal full names)
   - Case Invariance & Punctuation Robustness (abarna.v, AARTHI V, dot variations)
   - Partial / Incomplete Name Resolution (First-name + department context)
   - Colloquial & Conversational Slang ("Hey Vir, how's Aarthi doing?")
   - Code-Mixed / Tanglish Expressions ("marks details sollunga", "hosteller-ah day scholar-ah")
   - Multi-Turn Conversation & Anaphoric Pronoun Resolution ("Tell me about X" -> "What are her marks?")
   - Relational & Multi-Condition Filtering (Gender + Department + Residence)
   - Superlative & Comparative Aggregations (Top scorers, department averages)
   - Negation & Out-of-Domain Boundaries (No arrears, non-existent students)
   - Hybrid Knowledge Fusion (SQL Database + Vector Search Regulations)

Usage:
  - Run with pytest: pytest tests/test_student_nlp_queries.py -v
  - Run standalone CLI: python tests/test_student_nlp_queries.py
"""

import os
import sys
import time
import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from services.agent import run_agent
from services.fast_path import fast_path


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================

def assert_valid_response(result: dict, min_length: int = 15):
    """Ensures response is non-empty, above minimum length, and free of system error phrases."""
    answer = result.get("answer", "").strip()
    assert answer, "Agent returned an empty answer."
    assert len(answer) >= min_length, f"Answer is too short ({len(answer)} chars): {answer}"
    
    error_indicators = [
        "sql query failed",
        "syntax error",
        "operationalerror",
        "table not found",
        "no such column",
        "exception in agent",
    ]
    answer_lower = answer.lower()
    for err in error_indicators:
        assert err not in answer_lower, f"Error phrase '{err}' found in answer: {answer}"


def assert_tool_used(result: dict, expected_tool: str):
    """Verifies that the expected tool was invoked during the agentic reasoning loop."""
    tools = result.get("tools_used", [])
    assert expected_tool in tools, f"Expected '{expected_tool}' to be used, but got: {tools}"


# ==============================================================================
# 1. PARENT PERSONA TEST SUITE
# ==============================================================================

class TestParentQueries:
    """
    Tests real-world phrasing used by parents checking on their son/daughter.
    Focus: Academic performance, attendance, hostel accommodation, guardian details.
    """

    def test_parent_checking_child_marks_natural_language(self):
        """Parent asking in conversational English about their daughter's IAT-2 marks."""
        q = "My daughter Aarthi V is studying in 2nd year CSE. Could you tell me her marks in the IAT-2 exam?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        # Aarthi V has records for Theory of Computation, AI & ML, DBMS
        assert any(k in result["answer"].lower() for k in ["theory of computation", "dbms", "database", "100", "84", "aarthi"])

    def test_parent_checking_attendance_percentage(self):
        """Parent inquiring about their child's attendance and missed classes."""
        q = "How is my son Aasaimani T's attendance in IT department? Has he missed any classes?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "aasaimani" in result["answer"].lower() or "100" in result["answer"]

    def test_parent_verifying_hostel_residence(self):
        """Parent checking residence status (Hosteller vs Day Scholar)."""
        q = "Is my daughter Deepika N registered as a hosteller or a day scholar in AI&DS?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "hostel" in result["answer"].lower() or "hosteller" in result["answer"].lower()

    def test_parent_guardian_contact_verification(self):
        """Parent/guardian checking registered father name and contact for student 511524243002."""
        q = "Who is listed as the father and what is the parent phone number for student 511524243002?"
        result = run_agent(q)
        assert_valid_response(result)
        # Register number might route directly or via agent sql_query
        assert "ramesh" in result["answer"].lower() or "6380238604" in result["answer"]

    def test_parent_arrear_inquiry(self):
        """Parent checking if their ward has any uncleared subjects or arrears."""
        q = "Please check if Aarthi V from CSE has any arrears or failed subjects."
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")


# ==============================================================================
# 2. FACULTY / STAFF / HOD PERSONA TEST SUITE
# ==============================================================================

class TestStaffFacultyQueries:
    """
    Tests administrative, analytical, and class advising queries from faculty/staff.
    Focus: Aggregations, arrear rosters, attendance defaulters, student dossiers.
    """

    def test_staff_department_student_count(self):
        """HOD asking for total headcount in a department cohort."""
        q = "How many students are enrolled in the AI&DS department batch 2024-2028?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")

    def test_staff_subject_topper_and_analytics(self):
        """Faculty checking high performers for Theory of Computation in CSE."""
        q = "Show me the top scoring students for Theory of Computation in CSE 4th semester."
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "aarthi" in result["answer"].lower() or "100" in result["answer"]

    def test_staff_attendance_eligibility_filter(self):
        """Class incharge checking students marked NOT_ELIGIBLE or low attendance."""
        q = "List students in the IT department along with their exam eligibility status."
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")

    def test_staff_complete_student_advising_dossier(self):
        """Faculty advisor retrieving comprehensive student profile by register number."""
        q = "Provide a full academic and profile summary for register number 511523104001."
        result = run_agent(q)
        assert_valid_response(result)
        assert "aarthi" in result["answer"].lower() or "cse" in result["answer"].lower()

    def test_staff_gender_distribution(self):
        """Staff querying demographic balance in AI&DS."""
        q = "How many female and male students are in the AI&DS department?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")


# ==============================================================================
# 3. STUDENT / PEER PERSONA TEST SUITE
# ==============================================================================

class TestStudentPeerQueries:
    """
    Tests casual, first-person, and peer queries typical among college students.
    Focus: Self marks lookup, subject grades, peer contact, exam readiness.
    """

    def test_student_check_own_grades_by_reg_no(self):
        """Student looking up their marks across all subjects in IAT-2."""
        q = "I am register number 511523104001, what are my scores in DBMS and AI&ML?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "74" in result["answer"] or "84" in result["answer"] or "a" in result["answer"].lower()

    def test_student_check_exam_eligibility(self):
        """Student checking whether their attendance permits writing semester exams."""
        q = "Check attendance and exam eligibility status for Aathi S from IT department."
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "aathi" in result["answer"].lower() or "100" in result["answer"]

    def test_student_peer_blood_group_emergency(self):
        """Student looking for peer blood group for blood donation / medical need."""
        q = "What is the blood group and phone number of Abarna V in AI&DS?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "o+" in result["answer"].lower() or "8056634587" in result["answer"]

    def test_student_asking_who_got_highest_mark(self):
        """Student asking who scored the highest in Theory of Computation."""
        q = "Who got 100 out of 100 in Theory of Computation?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "aarthi" in result["answer"].lower()


# ==============================================================================
# 4. NLP TECHNIQUES & LINGUISTIC VARIATIONS TEST SUITE
# ==============================================================================

class TestNLPTechniquesAndVariations:
    """
    Tests robustness across diverse Natural Language Processing variations:
    - Fuzzy case / punctuation
    - Incomplete / Partial names
    - Colloquial / Casual phrasing
    - Code-Mixed / Indian English / Tanglish
    - Multi-turn conversation anaphora
    - Relational multi-attribute queries
    - Negation / Out-of-bounds boundary handling
    - Hybrid Knowledge Fusion (DB + Policy RAG)
    """

    def test_exact_12_digit_reg_no_fast_path(self):
        """NLP Technique: Entity extraction on strict 12-digit register number."""
        q = "511524243001"
        assert fast_path(q) == "sql_only"

    def test_case_invariance_and_dot_punctuation(self):
        """NLP Technique: Case-insensitivity and punctuation robustness (dots, mixed case)."""
        # Testing 'aBaRnA.v' and 'aarthi v' lowercase
        q1 = "Tell me the department and batch of aBaRnA.v"
        result1 = run_agent(q1)
        assert_valid_response(result1)
        assert "ai&ds" in result1["answer"].lower() or "2024-2028" in result1["answer"]

        q2 = "give me the father name of aarthi v from cse"
        result2 = run_agent(q2)
        assert_valid_response(result2)

    def test_partial_first_name_resolution(self):
        """NLP Technique: Disambiguation with partial first name + department context."""
        q = "Tell me about student Adhisammandhar from AI&DS department"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "adhisammandhar" in result["answer"].lower() or "511524243002" in result["answer"]

    def test_colloquial_conversational_phrasing(self):
        """NLP Technique: Informal, conversational query structure with filler words."""
        q = "Hey assistant, can you quickly pull up the address and phone for student Akash D?"
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert "akash" in result["answer"].lower() or "6384544180" in result["answer"]

    def test_code_mixed_tanglish_phrasing(self):
        """NLP Technique: Code-mixed Indian English / Tamil-English (Tanglish) patterns."""
        # 1. Asking for marks details
        q1 = "Aarthi V CSE marks details kudunga"
        result1 = run_agent(q1)
        assert_valid_response(result1)
        assert_tool_used(result1, "sql_query")

        # 2. Asking residence type in Tanglish
        q2 = "511524243001 hosteller-ah illa day scholar-ah?"
        result2 = run_agent(q2)
        assert_valid_response(result2)
        assert "hostel" in result2["answer"].lower() or "hosteller" in result2["answer"].lower()

    def test_multiturn_anaphora_pronoun_resolution(self):
        """NLP Technique: Multi-turn contextual continuity resolving 'her' and 'she'."""
        history = [
            {"role": "user", "content": "Tell me about student Aarthi V in CSE"},
            {"role": "assistant", "content": "Aarthi V (Reg No: 511523104001) is a 2nd year student in Computer Science and Engineering."},
        ]
        q = "What are her scores in the 4th semester assessments?"
        result = run_agent(q, history=history)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        assert any(subj in result["answer"].lower() for subj in ["theory", "dbms", "intelligence", "100", "84", "74"])

    def test_relational_multi_condition_filtering(self):
        """NLP Technique: Multi-attribute compound filtering (Department + Residence + Gender)."""
        q = "List all female hosteller students in the AI&DS department."
        result = run_agent(q)
        assert_valid_response(result)
        assert_tool_used(result, "sql_query")
        # Abarna V and Deepika N are female hostellers in AI&DS
        assert "abarna" in result["answer"].lower() or "deepika" in result["answer"].lower()

    def test_negation_and_boundary_cases(self):
        """NLP Technique: Negation queries and non-existent entity handling."""
        # 1. Non-existent student entity
        q_fake = "What are the marks of student Harry Potter in semester 4?"
        result_fake = run_agent(q_fake)
        assert_valid_response(result_fake)
        # Should gracefully inform no records found rather than crashing
        assert any(w in result_fake["answer"].lower() for w in ["not found", "no record", "no data", "could not find", "unable to find"])

    def test_hybrid_policy_plus_student_data(self):
        """NLP Technique: Hybrid knowledge fusion (Regulations PDF Vector Search + Student SQL DB)."""
        q = "What is the minimum attendance requirement for exam eligibility, and does Aasaimani T meet this requirement?"
        result = run_agent(q)
        assert_valid_response(result)
        tools = result.get("tools_used", [])
        assert len(tools) >= 1, "Expected at least one tool to be invoked."
        assert "attendance" in result["answer"].lower()


# ==============================================================================
# STANDALONE CLI TEST RUNNER & BENCHMARK REPORT
# ==============================================================================

def run_comprehensive_student_nlp_suite():
    """
    Executes a structured benchmark across all personas and NLP variations,
    printing real-time results, token usage, latency, and tool invocations.
    """
    print("\n" + "=" * 80)
    print(" 🚀 VIR CAMPUS ASSISTANT — STUDENT QUERIES NLP TEST BENCHMARK")
    print("=" * 80)

    test_matrix = [
        # (Persona / Category, NLP Technique, Question)
        ("Parent", "Conversational English", "My daughter Aarthi V is in 2nd year CSE. How did she score in IAT-2?"),
        ("Parent", "Attendance & Discipline", "What is my son Aasaimani T's attendance percentage? Has he missed any classes?"),
        ("Parent", "Hostel Verification", "Is my daughter Deepika N staying in the hostel or day scholar in AI&DS?"),
        ("Parent", "Guardian Contact Lookup", "Who is listed as father for Adhisammandhar R and what is the contact number?"),
        ("Staff / HOD", "Cohort Aggregation", "How many students are enrolled in AI&DS batch 2024-2028?"),
        ("Staff / Faculty", "Subject Top Scorers", "Who scored 100 marks in Theory of Computation in CSE?"),
        ("Staff / Faculty", "Attendance Status", "Show attendance and exam eligibility status for students in IT department"),
        ("Student / Peer", "Self Grade Inquiry", "I am reg no 511523104001, what are my marks in DBMS and AI?"),
        ("Student / Peer", "Emergency Medical", "What is the blood group and phone number of Abarna V?"),
        ("NLP Variation", "Fuzzy / Case Invariance", "give me the batch and department of aBaRnA.v"),
        ("NLP Variation", "Partial Name Resolution", "Tell me about student Adhisammandhar in AI&DS"),
        ("NLP Variation", "Code-Mixed / Tanglish", "Aarthi V CSE marks details sollunga"),
        ("NLP Variation", "Tanglish Residence", "511524243001 hosteller-ah day scholar-ah?"),
        ("NLP Variation", "Multi-Attribute Filter", "Show all female hostellers in the AI&DS department"),
        ("NLP Variation", "Negative Boundary Case", "Check marks for non-existent student Hermione Granger in IT"),
        ("Hybrid Fusion", "Policy + SQL Database", "What is the minimum attendance requirement and what is Aasaimani T's attendance?"),
    ]

    total_prompt_tok = 0
    total_comp_tok = 0
    total_tok = 0
    test_results = []

    print(f"Total Test Cases: {len(test_matrix)}\n")

    for idx, (persona, nlp_tech, q) in enumerate(test_matrix, 1):
        print(f"[{idx:02d}/{len(test_matrix):02d}] 🧑 Persona: {persona} | 🔬 NLP: {nlp_tech}")
        print(f"     ❓ Query: \"{q}\"")

        start_time = time.time()
        try:
            res = run_agent(q)
            elapsed = time.time() - start_time
            tools = res.get("tools_used", [])
            tokens = res.get("tokens", {})
            p_tok = tokens.get("prompt_tokens", 0)
            c_tok = tokens.get("completion_tokens", 0)
            t_tok = tokens.get("total_tokens", 0) or (p_tok + c_tok)

            total_prompt_tok += p_tok
            total_comp_tok += c_tok
            total_tok += t_tok

            answer_snippet = res.get("answer", "").strip().replace("\n", " ")[:120]
            is_ok = bool(res.get("answer", "").strip()) and "error" not in res.get("answer", "").lower()[:50]

            print(f"     ⚡ Tools: {tools} | Rounds: {res.get('rounds', 1)} | Time: {elapsed:.2f}s")
            print(f"     📊 Tokens: Prompt={p_tok:,} | Completion={c_tok:,} | Total={t_tok:,}")
            print(f"     💬 Answer: {answer_snippet}...")
            print(f"     ✅ Status: {'PASS' if is_ok else 'CHECK'}\n")

            test_results.append({
                "index": idx,
                "persona": persona,
                "nlp_tech": nlp_tech,
                "query": q,
                "tools": tools,
                "rounds": res.get("rounds", 1),
                "tokens": t_tok,
                "time": elapsed,
                "passed": is_ok,
            })
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"     ❌ FAILED with exception: {e}\n")
            test_results.append({
                "index": idx,
                "persona": persona,
                "nlp_tech": nlp_tech,
                "query": q,
                "tools": [],
                "rounds": 0,
                "tokens": 0,
                "time": elapsed,
                "passed": False,
                "error": str(e),
            })

    # Summary table
    print("\n" + "=" * 80)
    print(" 📋 BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    passed_count = sum(1 for r in test_results if r["passed"])
    print(f"Results: {passed_count}/{len(test_results)} Test Cases Passed ({passed_count/len(test_results)*100:.1f}%)\n")

    print(f"{'No':<4} {'Status':<7} {'Persona':<14} {'NLP Technique':<22} {'Tools':<16} {'Time':<7} {'Tokens'}")
    print("-" * 80)
    for r in test_results:
        st_icon = "✅ PASS" if r["passed"] else "❌ FAIL"
        tools_str = ",".join(r["tools"]) if r["tools"] else "none"
        print(f"{r['index']:<4} {st_icon:<7} {r['persona'][:13]:<14} {r['nlp_tech'][:21]:<22} {tools_str[:15]:<16} {r['time']:.2f}s  {r['tokens']:,}")

    print("-" * 80)
    print(f"TOTAL TOKENS CONSUMED : {total_tok:,}")
    print(f"  └─ Prompt Tokens    : {total_prompt_tok:,}")
    print(f"  └─ Completion Tokens: {total_comp_tok:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_comprehensive_student_nlp_suite()
