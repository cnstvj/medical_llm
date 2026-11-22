"""
rerun_pipeline.py
=================
Pipeline orchestrator script to clean output artifacts and rerun preprocessing,
sequence building, and 5-fold cross-validation.

Usage:
  python rerun_pipeline.py                         # Full production run
  python rerun_pipeline.py --dummy                 # Rapid synthetic test run
  python rerun_pipeline.py --dummy --unimodal      # Fast CPU dummy test without ClinicalBERT
  python rerun_pipeline.py --label "Feature test"  # Tag experiment in logs
"""

import os
import json
import shutil
import sys

# Ensure script runs from its own directory and src is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import DEFAULT_CONFIG, parse_pipeline_args
from medical_llm.data.preprocessing import run_preprocessing
from medical_llm.data.sequence_builder import run_sequence_builder
from medical_llm.pipeline.cross_validator import run_cross_validation
from medical_llm.utils.logging import log_event
from synthetic.generate_dummy_data import generate_dummy_datasets


def clean_artifacts(config=DEFAULT_CONFIG) -> None:
    """Removes intermediate pipeline files and checkpoint models."""
    generated_files = [
        config.processed_admissions_file,
        config.processed_labevents_file,
        config.processed_reports_file,
        config.sequence_dataset_file,
        config.quantiles_file,
    ]
    model_files = [
        f for f in os.listdir(".") if (f.startswith("best_fold_") or f == "best_model.pth") and f.endswith(".pth")
    ]
    for file in generated_files + model_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")

    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__", ignore_errors=True)


def main():
    config, dummy_mode, run_label = parse_pipeline_args()

    print("\n===== CLEANING OLD OUTPUTS =====\n")
    clean_artifacts(config)

    if dummy_mode:
        print("\n[DUMMY MODE] Generating synthetic dummy CSV files...\n")
        generate_dummy_datasets(verbose=True)

    print("\n===== RUNNING PREPROCESSING =====\n")
    run_preprocessing(config=config, dummy_mode=dummy_mode)

    if not os.path.exists(config.quantiles_file):
        raise RuntimeError(f"{config.quantiles_file} was not generated.")
    print(f"[OK] {config.quantiles_file} created")

    print("\n===== RUNNING SEQUENCE BUILDER =====\n")
    run_sequence_builder(config=config)

    print("\n===== VERIFYING DATASET =====\n")
    with open(config.sequence_dataset_file, "r") as f:
        data = json.load(f)
        total = len(data["labels"])
        deaths = sum(data["labels"])
        alive = total - deaths

        print(f"Sequences : {total}")
        print(f"Deaths    : {deaths}")
        print(f"Alive     : {alive}")

        if not dummy_mode:
            if total != 129:
                print("\nWARNING:\nExpected roughly 129 admissions.")
            if deaths != 40:
                print("\nWARNING:\nExpected roughly 40 deaths.")
        else:
            print("[DUMMY MODE] Synthetic dataset verified.")

    print("\n===== RUNNING 5-FOLD CROSS VALIDATION =====\n")
    if run_label is None:
        if sys.stdin.isatty():
            try:
                user_input = input("Enter a label / note for this run (e.g. 'Consolidated pipeline'): ").strip()
                run_label = user_input if user_input else "Standard rerun"
            except (EOFError, KeyboardInterrupt):
                run_label = "Standard rerun"
        else:
            run_label = "Standard rerun"

    results = run_cross_validation(config=config, dummy_mode=dummy_mode, run_label=run_label)

    print("\n===== PIPELINE COMPLETE =====\n")
    saved_models = [
        f for f in os.listdir(".") if f.startswith("best_fold_") and f.endswith(".pth")
    ]
    print(f"Saved fold models: {len(saved_models)}")
    for model in sorted(saved_models):
        print(f"  {model}")

    print(f"\n[OK] Results logged to {config.log_file} with label: '{run_label}'")
    return results


if __name__ == "__main__":
    main()
