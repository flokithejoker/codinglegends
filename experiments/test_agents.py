"""Simple test to verify agent classes work end-to-end."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.analyse_agent import AnalyseAgent
from src.agents.locate_agent import LocateAgent
from src.agents.assign_agent import AssignAgent
from src.agents.verify_agent import VerifyAgent

# Example clinical note
CLINICAL_NOTE = """
CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS:
65-year-old male presents to the ED with substernal chest pain radiating to the left arm,
associated with diaphoresis and shortness of breath for the past 2 hours. Patient has a
history of hypertension and type 2 diabetes mellitus. He reports compliance with his
medications including metformin and lisinopril.

PHYSICAL EXAMINATION:
- BP: 165/95 mmHg
- HR: 102 bpm
- SpO2: 94% on room air
- Lungs: Bilateral crackles at bases
- Heart: Regular rhythm, no murmurs

ASSESSMENT/PLAN:
1. Acute ST-elevation myocardial infarction (STEMI) - cardiology consulted for emergent cath
2. Hypertensive urgency - continue lisinopril, monitor
3. Type 2 diabetes mellitus - hold metformin, monitor glucose
4. Acute hypoxic respiratory failure - supplemental O2, monitor
"""

# Mock data for locate agent (normally from RAG)
MOCK_TERMS = [
    {"path": "Infarction, myocardium, myocardial (acute)"},
    {"path": "Hypertension, hypertensive"},
    {"path": "Diabetes, diabetic, type 2"},
    {"path": "Failure, respiratory, acute"},
    {"path": "Pain, chest"},
    {"path": "Fracture, arm"},  # irrelevant
]

# Mock data for assign agent (normally from index lookup)
MOCK_CODES = [
    {"code": "I21.3", "description": "ST elevation myocardial infarction of unspecified site", "path": "Infarction, myocardium"},
    {"code": "I10", "description": "Essential (primary) hypertension", "path": "Hypertension"},
    {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "path": "Diabetes, type 2"},
    {"code": "J96.01", "description": "Acute respiratory failure with hypoxia", "path": "Failure, respiratory"},
    {"code": "S52.90", "description": "Unspecified fracture of unspecified forearm", "path": "Fracture, arm"},  # irrelevant
]

# Mock data for verify agent (normally from guidelines DB)
MOCK_GUIDELINES = [
    {"content": "Code I21.- for STEMI requires documentation of ST elevation on ECG or troponin elevation."},
    {"content": "Hypertension should be coded when documented as diagnosis, not just elevated BP reading."},
]

MOCK_INSTRUCTIONAL_NOTES = [
    {
        "name": "I21.3",
        "assignable": True,
        "includes": ["ST elevation myocardial infarction", "Transmural infarction"],
        "excludes": ["Old myocardial infarction", "Chronic ischemic heart disease"],
    },
    {
        "name": "E11.9",
        "assignable": True,
        "includes": ["Type 2 diabetes NOS"],
        "use_additional": ["Code for any associated complications"],
    },
]

# Mock codes for verify (slightly different format)
MOCK_VERIFY_CODES = [
    {"name": "I21.3", "description": "ST elevation myocardial infarction of unspecified site"},
    {"name": "I10", "description": "Essential (primary) hypertension"},
    {"name": "E11.9", "description": "Type 2 diabetes mellitus without complications"},
    {"name": "J96.01", "description": "Acute respiratory failure with hypoxia"},
]


def test_analyse():
    print("\n" + "=" * 50)
    print("Testing AnalyseAgent")
    print("=" * 50)

    agent = AnalyseAgent("openai", "gpt-4o-mini")
    result = agent.run_single(note=CLINICAL_NOTE)

    print(f"Extracted terms: {result.terms}")
    return result.terms


def test_locate(terms_from_analyse=None):
    print("\n" + "=" * 50)
    print("Testing LocateAgent")
    print("=" * 50)

    agent = LocateAgent("openai", "gpt-4o-mini")
    result = agent.run_single(note=CLINICAL_NOTE, terms=MOCK_TERMS)

    print(f"Selected term IDs: {result.term_ids}")
    return result.term_ids


def test_assign():
    print("\n" + "=" * 50)
    print("Testing AssignAgent")
    print("=" * 50)

    agent = AssignAgent("openai", "gpt-4o-mini")
    result = agent.run_single(note=CLINICAL_NOTE, codes=MOCK_CODES)

    print(f"Assigned code IDs: {result.code_ids}")
    return result.code_ids


def test_verify():
    print("\n" + "=" * 50)
    print("Testing VerifyAgent")
    print("=" * 50)

    agent = VerifyAgent("openai", "gpt-4o-mini")
    result = agent.run_single(
        note=CLINICAL_NOTE,
        guidelines=MOCK_GUIDELINES,
        instructional_notes=MOCK_INSTRUCTIONAL_NOTES,
        codes=MOCK_VERIFY_CODES,
    )

    print(f"Verified code IDs: {result.code_ids}")
    return result.code_ids


def run_pipeline():
    print("\n" + "#" * 50)
    print("Running full pipeline test")
    print("#" * 50)

    # Step 1: Analyse
    terms = test_analyse()

    # Step 2: Locate (using mock terms since no RAG yet)
    term_ids = test_locate()

    # Step 3: Assign
    code_ids = test_assign()

    # Step 4: Verify
    verified_ids = test_verify()

    print("\n" + "#" * 50)
    print("Pipeline complete!")
    print("#" * 50)
    print(f"Analyse -> {len(terms)} terms extracted")
    print(f"Locate -> selected IDs: {term_ids}")
    print(f"Assign -> assigned IDs: {code_ids}")
    print(f"Verify -> verified IDs: {verified_ids}")


if __name__ == "__main__":
    run_pipeline()
