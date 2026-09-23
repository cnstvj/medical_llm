"""
rerun_pipeline.py
=================
Pipeline orchestrator script to clean output artifacts and rerun preprocessing, sequence building, and cross-validation.
"""

import os
import json
import shutil
import sys

# Ensure script runs from its own directory and src is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import DEFAULT_CONFIG
from medical_llm.data.preprocessing import run_preprocessing
from medical_llm.data.sequence_builder import run_sequence_builder
from medical_llm.pipeline.cross_validator import run_cross_validation
from medical_llm.utils.logging import log_event

DUMMY_MODE = "--dummy" in sys.argv

if DUMMY_MODE:
    print("[DUMMY MODE] Generating dummy CSV files...")
    import subprocess
    subprocess.run([sys.executable, "generate_dummy_data.py"], check=True)

GENERATED_FILES = [
    DEFAULT_CONFIG.processed_admissions_file,
    DEFAULT_CONFIG.processed_labevents_file,
    DEFAULT_CONFIG.processed_reports_file,
    DEFAULT_CONFIG.sequence_dataset_file,
    DEFAULT_CONFIG.quantiles_file,
]

print("\n===== CLEANING OLD OUTPUTS =====\n")

MODEL_FILES = [
    f for f in os.listdir(".") if (f.startswith("best_fold_") or f == "best_model.pth") and f.endswith(".pth")
]

for file in GENERATED_FILES + MODEL_FILES:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")

if os.path.exists("__pycache__"):
    shutil.rmtree("__pycache__")
    print("Deleted __pycache__")

print("\n===== RUNNING PREPROCESSING =====\n")

run_preprocessing(dummy_mode=DUMMY_MODE)

if not os.path.exists(DEFAULT_CONFIG.quantiles_file):
    raise RuntimeError(f"{DEFAULT_CONFIG.quantiles_file} was not generated.")

print(f"[OK] {DEFAULT_CONFIG.quantiles_file} created")

print("\n===== RUNNING SEQUENCE BUILDER =====\n")

run_sequence_builder()

print("\n===== VERIFYING DATASET =====\n")

with open(DEFAULT_CONFIG.sequence_dataset_file, "r") as f:
    data = json.load(f)
    total = len(data["labels"])
    deaths = sum(data["labels"])
    alive = total - deaths

    print(f"Sequences : {total}")
    print(f"Deaths    : {deaths}")
    print(f"Alive     : {alive}")

    if not DUMMY_MODE:
        if total != 129:
            print("\nWARNING:\nExpected roughly 129 admissions.")
        if deaths != 40:
            print("\nWARNING:\nExpected roughly 40 deaths.")
    else:
        print("[DUMMY MODE] Skipping production dataset size checks.")

print("\n===== RUNNING 5-FOLD CROSS VALIDATION =====\n")

# Allow CLI --label or --tag, or prompt user interactively
run_label = None
for i, arg in enumerate(sys.argv):
    if arg in ("--label", "--tag") and i + 1 < len(sys.argv):
        run_label = sys.argv[i + 1]

if run_label is None:
    try:
        user_input = input("Enter a label / note for this run (e.g. 'Added temporal cutoff fix'): ").strip()
        run_label = user_input if user_input else "Standard rerun"
    except (EOFError, KeyboardInterrupt):
        run_label = "Standard rerun"

results = run_cross_validation(config=DEFAULT_CONFIG, dummy_mode=DUMMY_MODE, run_label=run_label)

print("\n===== PIPELINE COMPLETE =====\n")

saved_models = [
    f for f in os.listdir(".") if f.startswith("best_fold_") and f.endswith(".pth")
]

print(f"Saved fold models: {len(saved_models)}")
for model in sorted(saved_models):
    print(f"  {model}")

print(f"\n[OK] Results logged to {DEFAULT_CONFIG.log_file} with label: '{run_label}'")
