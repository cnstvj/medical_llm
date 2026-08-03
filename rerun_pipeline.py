# rerun_pipeline.py

import os
import json
import shutil
import subprocess
import sys

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DUMMY_MODE = "--dummy" in sys.argv

if DUMMY_MODE:
    print("[DUMMY MODE] Generating dummy CSV files...")
    subprocess.run([sys.executable, "generate_dummy_data.py"], check=True)

GENERATED_FILES = [
"processed_admissions.csv",
"processed_labevents.csv",
"processed_reports.csv",
"sequence_dataset.json",
"quantiles.pkl",
]

MODEL_FILES = [
f for f in os.listdir(".")
if f.startswith("best_fold_") and f.endswith(".pth")
]

print("\n===== CLEANING OLD OUTPUTS =====\n")

for file in GENERATED_FILES + MODEL_FILES:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")

if os.path.exists("__pycache__"):
    shutil.rmtree("__pycache__")
    print("Deleted __pycache__")

print("\n===== RUNNING PREPROCESSING =====\n")

preprocess_env = os.environ.copy()
if DUMMY_MODE:
    preprocess_env["DUMMY_MODE"] = "1"

subprocess.run(
[sys.executable, "preprocessing.py"],
env=preprocess_env,
check=True
)

if not os.path.exists("quantiles.pkl"):
    raise RuntimeError(
    "quantiles.pkl was not generated."
    )

print("[OK] quantiles.pkl created")

print("\n===== RUNNING SEQUENCE BUILDER =====\n")

subprocess.run(
[sys.executable, "sequence_builder.py"],
check=True
)

print("\n===== VERIFYING DATASET =====\n")

with open("sequence_dataset.json", "r") as f:
    data = json.load(f)

    total = len(data["labels"])
    deaths = sum(data["labels"])
    alive = total - deaths

    print(f"Sequences : {total}")
    print(f"Deaths    : {deaths}")
    print(f"Alive     : {alive}")

    if not DUMMY_MODE:
        if total != 129:
            print(
            "\nWARNING:"
            "\nExpected roughly 129 admissions."
            )

        if deaths != 40:
            print(
            "\nWARNING:"
            "\nExpected roughly 40 deaths."
            )
    else:
        print("[DUMMY MODE] Skipping production dataset size checks.")

print("\n===== RUNNING 5-FOLD CROSS VALIDATION =====\n")

subprocess.run(
[sys.executable, "cross_validate.py"],
check=True
)

print("\n===== PIPELINE COMPLETE =====\n")

saved_models = [
f for f in os.listdir(".")
if f.startswith("best_fold_")
and f.endswith(".pth")
]

print(
f"Saved fold models: "
f"{len(saved_models)}"
)

for model in sorted(saved_models):
    print(f"  {model}")
