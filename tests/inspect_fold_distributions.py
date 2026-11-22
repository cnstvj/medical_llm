"""
Script to inspect synthetic labels, fold sizes, class balance, and Fold 3 behavior.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from medical_llm.config import PipelineConfig
from medical_llm.data.preprocessing import load_raw_tables

def inspect():
    config = PipelineConfig()
    print("=== 1. LOADING DUMMY DATA TABLES ===")
    admissions, labevents, reports = load_raw_tables(config, dummy_mode=True)
    
    print(f"Total admissions: {len(admissions)}")
    print(f"Total unique patients (subject_id): {admissions['subject_id'].nunique()}")
    print(f"Total deaths (target=1): {admissions['target'].sum()}")
    print(f"Total survivors (target=0): {(admissions['target'] == 0).sum()}")
    print(f"Overall mortality rate: {admissions['target'].mean():.4f}")
    print()

    # Outer split
    subject_ids = admissions["subject_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.random_seed)
    outer_train_idx, outer_val_idx = next(gss.split(admissions, admissions["target"], groups=subject_ids))
    outer_train_admissions = admissions.iloc[outer_train_idx].copy()
    outer_val_admissions = admissions.iloc[outer_val_idx].copy()

    print("=== OUTER SPLIT (80% train, 20% val) ===")
    print(f"Outer Train: {len(outer_train_admissions)} adm, {outer_train_admissions['subject_id'].nunique()} patients, "
          f"Deaths={outer_train_admissions['target'].sum()}, Survivors={(outer_train_admissions['target'] == 0).sum()}")
    print(f"Outer Val:   {len(outer_val_admissions)} adm, {outer_val_admissions['subject_id'].nunique()} patients, "
          f"Deaths={outer_val_admissions['target'].sum()}, Survivors={(outer_val_admissions['target'] == 0).sum()}")
    print()

    # Inner 5-fold CV
    inner_subject_ids = outer_train_admissions["subject_id"].values
    inner_labels = outer_train_admissions["target"].values
    sgkf = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)

    print("=== 2. 5-FOLD CV SPLITS INSPECTION ===")
    fold_splits = list(sgkf.split(outer_train_admissions, inner_labels, groups=inner_subject_ids))
    for fold, (train_idx, test_idx) in enumerate(fold_splits, start=1):
        tr_adm = outer_train_admissions.iloc[train_idx]
        te_adm = outer_train_admissions.iloc[test_idx]
        
        tr_deaths = int(tr_adm["target"].sum())
        tr_surv = int((tr_adm["target"] == 0).sum())
        te_deaths = int(te_adm["target"].sum())
        te_surv = int((te_adm["target"] == 0).sum())

        print(f"Fold {fold}:")
        print(f"  Train: total={len(tr_adm):2d} (patients={tr_adm['subject_id'].nunique():2d}) | Deaths={tr_deaths:2d}, Survivors={tr_surv:2d} (Mortality={tr_deaths/len(tr_adm):.1%})")
        print(f"  Test : total={len(te_adm):2d} (patients={te_adm['subject_id'].nunique():2d}) | Deaths={te_deaths:2d}, Survivors={te_surv:2d} (Mortality={te_deaths/len(te_adm):.1%})")
        
        if fold == 3:
            print(f"  --> Fold 3 test subject_ids: {te_adm['subject_id'].tolist()}")
            print(f"  --> Fold 3 test hadm_ids: {te_adm['hadm_id'].tolist()}")
            print(f"  --> Fold 3 test targets: {te_adm['target'].tolist()}")

if __name__ == "__main__":
    inspect()
