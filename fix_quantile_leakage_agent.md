# Fix Quantile Leakage --- Minimal Project-Safe Patch

## Goal

Fix **quantile leakage** in the existing MIMIC/medical Transformer
project **without breaking the current project architecture, dataset
format, model, training code, or ClinicalBERT-related work**.

The current issue is:

``` text
ALL MIMIC DATA
      ↓
Calculate quantile boundaries
      ↓
Create quantile tokens
      ↓
Split into Train / Validation / Test
```

This leaks information from validation/test data into preprocessing.

The required pipeline is:

``` text
ALL DATA
   ↓
Patient/admission-level Train / Validation / Test split
   ↓
Fit quantile boundaries ONLY on TRAIN
   ↓
Save quantile boundaries
   ↓
Apply the SAME boundaries to Train / Validation / Test
   ↓
Create quantile tokens
   ↓
Train / Validation / Test
```

------------------------------------------------------------------------

## Critical requirements

### 1. Do NOT rewrite the whole project

Make the **smallest safe change possible**.

Do NOT: - replace the existing Transformer - replace the tokenizer -
replace the dataset format - change model architecture unnecessarily -
remove existing preprocessing steps - change training objectives -
introduce ClinicalBERT unless it is already part of the current code -
change unrelated files - rename public functions/classes unless
absolutely necessary

Preserve existing behavior everywhere except the quantile preprocessing.

------------------------------------------------------------------------

## 2. Identify the current leakage

The current code reportedly contains logic similar to:

``` python
quantile_map = fit_quantiles(labevents)
```

and:

``` python
for itemid, group in labevents.groupby("itemid"):
    vals = group["valuenum"].dropna()
    boundaries = np.quantile(
        vals,
        np.linspace(0, 1, 11)
    )
```

This is problematic if `labevents` contains records from the entire
dataset before the train/validation/test split.

First inspect the actual repository and verify the exact implementation
before modifying anything.

Do NOT assume filenames or function signatures if the repository
differs.

------------------------------------------------------------------------

# Required implementation

## Step 1 --- Find the split point

Determine exactly where the project creates:

-   training data
-   validation data
-   test data

The split must be done at the appropriate **patient/admission level**,
not randomly across individual clinical events if that would allow the
same patient's information to appear in multiple splits.

Preserve the project's existing split strategy if it is already correct.

------------------------------------------------------------------------

## Step 2 --- Fit quantiles ONLY on training data

Create or reuse a function such as:

``` python
quantile_map = fit_quantiles(train_labevents)
```

The function must use only training records.

For each lab `itemid`:

``` python
vals = train_group["valuenum"].dropna()

boundaries = np.quantile(
    vals,
    np.linspace(0, 1, 11)
)
```

The exact existing number of quantile bins must be preserved.

If the project currently uses 10 quantile bins, keep 10.

Do not change the binning scheme unless required to fix an actual bug.

------------------------------------------------------------------------

## Step 3 --- Apply the same quantiles everywhere

After fitting:

``` python
quantile_map = fit_quantiles(train_labevents)
```

use the same `quantile_map` for:

``` text
TRAIN
VALIDATION
TEST
```

Conceptually:

``` python
train_tokens = apply_quantiles(train_labevents, quantile_map)
val_tokens   = apply_quantiles(val_labevents, quantile_map)
test_tokens  = apply_quantiles(test_labevents, quantile_map)
```

There must be **NO**:

``` python
fit_quantiles(val_labevents)
```

or:

``` python
fit_quantiles(test_labevents)
```

------------------------------------------------------------------------

# Important edge cases

Handle these safely without changing unrelated behavior.

## Missing values

Preserve the project's current missing-value handling.

Do not silently convert missing values into arbitrary quantile bins.

## Lab item with insufficient training values

If an `itemid` does not have enough valid training values:

-   preserve the existing fallback behavior if one exists
-   otherwise implement the smallest safe fallback
-   do not calculate its boundaries using validation/test data

## Duplicate quantile boundaries

Medical lab values can contain many repeated values.

For example:

``` text
[0, 0, 0, 0, 1, 1, 1, 2, 2]
```

can produce repeated quantile boundaries.

Do not redesign the project's tokenization just because of this.

Preserve the current behavior unless it causes a runtime failure.

If `np.digitize` or equivalent requires special handling, fix only that
issue.

------------------------------------------------------------------------

# Step 4 --- Prevent accidental leakage in future runs

Prefer making the preprocessing API explicit.

For example:

``` python
quantile_map = fit_quantiles(train_labevents)

train_labevents = apply_quantiles(
    train_labevents,
    quantile_map
)

val_labevents = apply_quantiles(
    val_labevents,
    quantile_map
)

test_labevents = apply_quantiles(
    test_labevents,
    quantile_map
)
```

This is preferable to a function that silently calculates quantiles
internally from whatever dataframe it receives.

------------------------------------------------------------------------

# Step 5 --- Persist quantile boundaries if appropriate

If the current project saves preprocessing artifacts, add the quantile
map to that mechanism.

For example:

``` python
import json

with open("quantile_map.json", "w") as f:
    json.dump(quantile_map, f)
```

However, **do not add a new persistence system if the project already
has one**.

Reuse the existing artifact/config/checkpoint mechanism.

The important rule is:

> Quantile boundaries must be fitted from training data and reused
> unchanged during validation, testing, and inference.

------------------------------------------------------------------------

# Step 6 --- Avoid inference-time refitting

The production/inference path must NEVER do:

``` python
fit_quantiles(new_patient_data)
```

It must load/use the training quantile boundaries:

``` text
Saved training quantiles
        ↓
New patient
        ↓
Apply existing quantiles
        ↓
Model
```

------------------------------------------------------------------------

# Validation checklist

After making the patch, verify all of the following.

### Leakage

-   [ ] Quantiles are fitted only from training data.
-   [ ] Validation data never contributes to quantile boundaries.
-   [ ] Test data never contributes to quantile boundaries.
-   [ ] Inference data never contributes to quantile boundaries.
-   [ ] The same quantile boundaries are used for train/validation/test.

### Data splitting

-   [ ] Existing patient/admission split is preserved.
-   [ ] No patient leakage is introduced.
-   [ ] Dataset sizes remain approximately/identically what they were
    before.

### Project compatibility

-   [ ] Existing model architecture is unchanged.
-   [ ] Existing tokenizer behavior is unchanged.
-   [ ] Existing sequence format remains compatible.
-   [ ] Existing training script still runs.
-   [ ] Existing validation/evaluation still runs.
-   [ ] Existing test/evaluation still runs.
-   [ ] No unrelated functionality is changed.

------------------------------------------------------------------------

# Add tests

Add small tests specifically for the leakage fix.

## Test 1 --- Validation cannot affect quantiles

Create a synthetic example:

``` python
train = [1, 2, 3, 4, 5]
validation = [100000, 200000, 300000]
```

Fit quantiles using train only.

Then verify that changing validation values does **not** change the
fitted quantile map.

For example:

``` python
q1 = fit_quantiles(train)

validation_changed = [9999999, 8888888, 7777777]

q2 = fit_quantiles(train)

assert q1 == q2
```

The more important architectural test is that the preprocessing code
never calls `fit_quantiles()` on validation/test data.

------------------------------------------------------------------------

## Test 2 --- Test cannot affect quantiles

Use the same idea with test data.

The fitted quantile boundaries must depend only on training data.

------------------------------------------------------------------------

## Test 3 --- Same boundaries are applied

Verify that:

``` python
apply_quantiles(train, qmap)
apply_quantiles(val, qmap)
apply_quantiles(test, qmap)
```

all use the exact same `qmap`.

------------------------------------------------------------------------

# Recommended architecture

The final preprocessing flow should look like:

``` text
                    RAW MIMIC DATA
                          │
                          ↓
              Patient/admission split
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
            TRAIN         VAL        TEST
              │
              ↓
       Fit quantile map
              │
              ↓
       quantile_map.json
              │
       ┌──────┼───────────────┐
       ↓      ↓               ↓
     TRAIN    VAL             TEST
       │      │               │
       └──────┼───────────────┘
              ↓
       Apply same quantiles
              ↓
       Quantile lab tokens
              ↓
       Existing token/sequence pipeline
              ↓
       Existing Transformer
              ↓
       Existing evaluation
```

------------------------------------------------------------------------

# Do not confuse this with ClinicalBERT

This task is ONLY to fix quantile leakage.

Do not modify ClinicalBERT integration unless the existing code requires
a compatibility change caused directly by the preprocessing fix.

ClinicalBERT and quantile preprocessing are separate concerns:

``` text
Clinical notes ──→ ClinicalBERT/text encoder
                         │
Lab values ───────→ quantile preprocessing
                         │
                         ↓
                  Existing model
```

Do not replace the current Transformer with ClinicalBERT as part of this
task.

------------------------------------------------------------------------

# Final deliverables

After modifying the project, report:

1.  Which files were changed.
2.  Exactly where quantile fitting previously happened.
3.  Exactly where it happens now.
4.  Confirmation that quantiles are fitted ONLY on training data.
5.  Confirmation that validation/test use the training quantile
    boundaries.
6.  Tests executed and their results.
7.  Any compatibility concerns.

Keep the patch minimal and project-safe.

## Success criterion

The project must behave the same as before **except that quantile
boundaries are no longer influenced by validation/test data**.

The core rule is:

> **SPLIT FIRST → FIT QUANTILES ON TRAIN ONLY → APPLY THE SAME QUANTILES
> EVERYWHERE ELSE.**
