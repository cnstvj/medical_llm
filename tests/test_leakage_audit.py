"""
Unit and integration tests for the LeakageAuditor module.
"""

import os
import json
import pytest
import pandas as pd
from medical_llm.utils.audit import LeakageAuditor


def test_audit_patient_splits_pass(tmp_path):
    manifest_data = {
        "subject_id": [1, 2, 3, 4, 5, 6],
        "hadm_id": [10, 20, 30, 40, 50, 60],
        "partition": ["dev", "dev", "dev", "dev", "holdout", "holdout"],
        "fold": [1, 1, 2, 2, 0, 0],
    }
    df = pd.DataFrame(manifest_data)
    auditor = LeakageAuditor(output_dir=str(tmp_path))

    passed = auditor.audit_patient_splits(df)
    assert passed is True
    assert auditor.results["patient_isolation"]["status"] == "PASS"


def test_audit_patient_splits_fail_outer(tmp_path):
    manifest_data = {
        "subject_id": [1, 2, 3, 4, 4, 6],
        "hadm_id": [10, 20, 30, 40, 50, 60],
        "partition": ["dev", "dev", "dev", "dev", "holdout", "holdout"],
        "fold": [1, 1, 2, 2, 0, 0],
    }
    df = pd.DataFrame(manifest_data)
    auditor = LeakageAuditor(output_dir=str(tmp_path))

    passed = auditor.audit_patient_splits(df)
    assert passed is False
    assert auditor.results["patient_outer_overlap"]["status"] == "FAIL"


def test_audit_temporal_windows_pass(tmp_path):
    cohort_data = {
        "hadm_id": [101, 102],
        "admittime": ["2020-01-01 00:00:00", "2020-01-02 00:00:00"],
        "dischtime": ["2020-01-05 00:00:00", "2020-01-06 00:00:00"],
    }
    labs_data = {
        "hadm_id": [101, 102],
        "charttime": ["2020-01-01 12:00:00", "2020-01-02 08:00:00"],
    }
    auditor = LeakageAuditor(output_dir=str(tmp_path))
    passed = auditor.audit_temporal_windows(pd.DataFrame(labs_data), pd.DataFrame(cohort_data), prediction_horizon_hours=24.0)
    assert passed is True
    assert auditor.results["temporal_boundary"]["status"] == "PASS"


def test_audit_target_and_identifiers(tmp_path):
    vocab = {"CREATININE_Q0": 0, "GLUCOSE_Q1": 1, "POTASSIUM_Q2": 2}
    cohort_df = pd.DataFrame({
        "subject_id": [12345, 67890],
        "hadm_id": [99999, 88888],
    })
    auditor = LeakageAuditor(output_dir=str(tmp_path))
    assert auditor.audit_target_and_identifiers(vocab, cohort_df) is True

    bad_vocab = {"CREATININE_Q0": 0, "12345": 1}
    assert auditor.audit_target_and_identifiers(bad_vocab, cohort_df) is False


def test_save_report(tmp_path):
    auditor = LeakageAuditor(output_dir=str(tmp_path))
    auditor.audit_single_output_directory(str(tmp_path))
    report_file = tmp_path / "audit_report.json"
    report = auditor.save_report(str(report_file))

    assert os.path.exists(report_file)
    assert report["overall_audit_status"] == "PASS"
