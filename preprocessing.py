import pandas as pd
import numpy as np
import re

def clean_text(text):
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split by spaces and join to remove multiple spaces
    return ' '.join(text.split())

def main():
    print("Loading data...")
    admissions = pd.read_csv('ADMISSIONS.csv')
    patients = pd.read_csv('PATIENTS.csv')
    labevents = pd.read_csv('LABEVENTS.csv')
    d_labitems = pd.read_csv('D_LABITEMS.csv')
    reports = pd.read_csv('structured_medical_records.csv')
    
    print("Merging admissions and patients...")
    adm_pat = pd.merge(admissions, patients, on='subject_id', how='inner')
    adm_pat['admittime'] = pd.to_datetime(adm_pat['admittime'])
    adm_pat['dob'] = pd.to_datetime(adm_pat['dob'])
    adm_pat['age'] = (adm_pat['admittime'] - adm_pat['dob']).dt.days / 365.25
    adm_pat.loc[adm_pat['age'] > 89, 'age'] = 90
    adm_pat['target'] = adm_pat['hospital_expire_flag']
    admissions_processed = adm_pat[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'age', 'gender', 'admission_type', 'target']]
    
    print("Processing lab events (Continuous Bucketing)...")
    labevents = labevents.dropna(subset=['hadm_id'])
    labevents = pd.merge(labevents, d_labitems[['itemid', 'label']], on='itemid', how='left')
    
    from quantile_utils import (
        fit_quantiles,
        apply_quantile,
        save_quantiles
    )

    quantile_map = fit_quantiles(
        labevents
    )

    save_quantiles(
        quantile_map
    )

    qbuckets = []

    for itemid, value in zip(
            labevents["itemid"],
            labevents["valuenum"]):

        if itemid in quantile_map:

            q = apply_quantile(
                value,
                quantile_map[itemid]
            )

        else:

            q = None

        qbuckets.append(q)

    labevents["qbucket"] = qbuckets
    
    labevents['flag'] = labevents['flag'].fillna('NORMAL')
    
    tokens = []
    for q, f, item in zip(labevents['qbucket'], labevents['flag'], labevents['itemid']):
        if pd.notnull(q):
            tokens.append(f"LAB_{item}_Q{int(q)}")
        else:
            tokens.append(f"LAB_{item}_{str(f).upper()}")
            
    labevents['event_token'] = tokens
    labevents = labevents[['subject_id', 'hadm_id', 'charttime', 'event_token', 'valuenum']]
    
    print("Processing medical reports (NLP)...")
    reports['clean_text'] = reports['medical_report'].apply(clean_text)

    # One report row per admission
    reports = (
        reports
        .groupby('hadm_id', as_index=False)
        .agg({
            'clean_text': lambda x: ' '.join(
                str(v) for v in x
                if pd.notnull(v) and str(v).strip()
            )
        })
    )
    
    print("Saving processed data...")
    admissions_processed.to_csv('processed_admissions.csv', index=False)
    labevents.to_csv('processed_labevents.csv', index=False)
    reports[['hadm_id', 'clean_text']].to_csv('processed_reports.csv', index=False)
    print("Preprocessing completed.")

if __name__ == '__main__':
    main()
