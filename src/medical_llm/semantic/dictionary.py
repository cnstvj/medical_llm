"""
Clinical Knowledge Dictionary and Ontology Mapper for MIMIC-III Technical Tokens.
Maps technical laboratory item IDs, quantile tokens, demographic tokens, and temporal gap
tokens into clinical concepts while strictly separating statistical quantiles from clinical abnormality.
"""

import os
from typing import Dict, Any, Optional, Tuple
import pandas as pd


# Known curated clinical profiles for primary ICU lab analytes
LAB_CLINICAL_PROFILES: Dict[int, Dict[str, Any]] = {
    50800: {
        "name": "Specimen Type / Blood Gas Panel",
        "system": "Arterial/Venous Blood Gas",
        "description": "Baseline blood gas specimen evaluation",
        "high_interpretation": "elevated panel parameter",
        "low_interpretation": "depressed panel parameter",
    },
    50809: {
        "name": "Glucose",
        "system": "Endocrine / Metabolism",
        "description": "Circulating blood glucose level",
        "high_interpretation": "elevated glucose (cohort top percentile, may indicate physiological stress response)",
        "low_interpretation": "low glucose (cohort bottom percentile, hypoglycemia risk)",
    },
    50813: {
        "name": "Lactate",
        "system": "Tissue Perfusion / Cellular Respiration",
        "description": "End-product of anaerobic metabolism",
        "high_interpretation": "elevated lactate (indicates potential anaerobic metabolism or cellular hypoperfusion)",
        "low_interpretation": "low/baseline lactate",
    },
    50818: {
        "name": "pCO2",
        "system": "Respiratory / Acid-Base",
        "description": "Partial pressure of carbon dioxide in blood",
        "high_interpretation": "elevated pCO2 (indicates potential hypercapnia or hypoventilation)",
        "low_interpretation": "reduced pCO2 (indicates hyperventilation or respiratory alkalosis)",
    },
    50820: {
        "name": "pH",
        "system": "Acid-Base Homeostasis",
        "description": "Serum/blood hydrogen ion concentration",
        "high_interpretation": "alkalemic pH range (relative to cohort)",
        "low_interpretation": "acidemic pH range (relative to cohort, indicating potential metabolic or respiratory acidosis)",
    },
    50821: {
        "name": "pO2",
        "system": "Oxygenation",
        "description": "Partial pressure of oxygen in blood",
        "high_interpretation": "high pO2 (hyperoxia)",
        "low_interpretation": "depressed pO2 (hypoxemia risk)",
    },
    50868: {
        "name": "Anion Gap",
        "system": "Acid-Base / Electrolytes",
        "description": "Difference between measured cations and anions",
        "high_interpretation": "elevated anion gap (indicates unmeasured organic anions)",
        "low_interpretation": "low anion gap",
    },
    50882: {
        "name": "Bicarbonate",
        "system": "Renal / Acid-Base Buffer",
        "description": "Serum bicarbonate buffer capacity",
        "high_interpretation": "elevated bicarbonate (metabolic alkalosis or compensatory retention)",
        "low_interpretation": "low bicarbonate (metabolic acidosis consumption or loss)",
    },
    50893: {
        "name": "Calcium, Total",
        "system": "Mineral / Neuromuscular",
        "description": "Total serum calcium level",
        "high_interpretation": "elevated calcium (hypercalcemia)",
        "low_interpretation": "reduced calcium (hypocalcemia)",
    },
    50902: {
        "name": "Chloride",
        "system": "Electrolytes",
        "description": "Serum chloride electrolyte level",
        "high_interpretation": "elevated chloride (hyperchloremia)",
        "low_interpretation": "low chloride (hypochloremia)",
    },
    50912: {
        "name": "Creatinine",
        "system": "Renal Function",
        "description": "Byproduct of muscle metabolism cleared by kidneys",
        "high_interpretation": "elevated creatinine (indicates reduced glomerular filtration or acute renal stress)",
        "low_interpretation": "low creatinine",
    },
    50931: {
        "name": "Glucose (Serum)",
        "system": "Endocrine / Metabolism",
        "description": "Serum chemistry glucose measurement",
        "high_interpretation": "elevated serum glucose (cohort top percentile)",
        "low_interpretation": "low serum glucose",
    },
    50960: {
        "name": "Magnesium",
        "system": "Electrolytes / Cardiac",
        "description": "Serum magnesium concentration",
        "high_interpretation": "elevated magnesium",
        "low_interpretation": "low magnesium",
    },
    50970: {
        "name": "Phosphate",
        "system": "Metabolic / Renal",
        "description": "Inorganic serum phosphorus level",
        "high_interpretation": "elevated phosphate",
        "low_interpretation": "low phosphate",
    },
    50971: {
        "name": "Potassium",
        "system": "Electrolytes / Cardiac Conduction",
        "description": "Serum potassium concentration",
        "high_interpretation": "elevated potassium (hyperkalemia risk)",
        "low_interpretation": "low potassium (hypokalemia risk)",
    },
    50983: {
        "name": "Sodium",
        "system": "Electrolytes / Fluid Osmolarity",
        "description": "Primary extracellular cation governing osmolarity",
        "high_interpretation": "elevated sodium (hypernatremia)",
        "low_interpretation": "low sodium (hyponatremia)",
    },
    51006: {
        "name": "Blood Urea Nitrogen (BUN)",
        "system": "Renal / Protein Catabolism",
        "description": "Nitrogenous breakdown product filtered by kidneys",
        "high_interpretation": "elevated urea nitrogen (indicates renal hypoperfusion or impaired clearance)",
        "low_interpretation": "low urea nitrogen",
    },
    51221: {
        "name": "Hematocrit",
        "system": "Hematology / Red Cell Volume",
        "description": "Percentage volume of red blood cells",
        "high_interpretation": "elevated hematocrit (hemoconcentration or erythrocytosis)",
        "low_interpretation": "reduced hematocrit (anemia or hemodilution)",
    },
    51222: {
        "name": "Hemoglobin",
        "system": "Hematology / Oxygen Transport",
        "description": "Oxygen-carrying protein in erythrocytes",
        "high_interpretation": "elevated hemoglobin",
        "low_interpretation": "depressed hemoglobin (anemia)",
    },
    51237: {
        "name": "INR (PT)",
        "system": "Coagulation",
        "description": "International normalized ratio for extrinsic clotting pathway",
        "high_interpretation": "elevated INR (coagulopathy or prolonged clotting time)",
        "low_interpretation": "normal/low INR",
    },
    51244: {
        "name": "Lymphocytes",
        "system": "Hematology / Immune",
        "description": "Circulating lymphocyte differential",
        "high_interpretation": "elevated lymphocytes",
        "low_interpretation": "depressed lymphocytes (lymphopenia)",
    },
    51248: {
        "name": "MCH",
        "system": "Hematology / Red Cell Indices",
        "description": "Mean corpuscular hemoglobin",
        "high_interpretation": "elevated MCH",
        "low_interpretation": "low MCH",
    },
    51249: {
        "name": "MCHC",
        "system": "Hematology / Red Cell Indices",
        "description": "Mean corpuscular hemoglobin concentration",
        "high_interpretation": "elevated MCHC",
        "low_interpretation": "low MCHC",
    },
    51250: {
        "name": "MCV",
        "system": "Hematology / Red Cell Indices",
        "description": "Mean corpuscular volume (average red blood cell size)",
        "high_interpretation": "macrocytic red cell index",
        "low_interpretation": "microcytic red cell index",
    },
    51265: {
        "name": "Platelet Count",
        "system": "Hematology / Hemostasis",
        "description": "Circulating thrombocyte count essential for clotting",
        "high_interpretation": "elevated platelet count (thrombocytosis)",
        "low_interpretation": "low platelet count (thrombocytopenia, bleeding vulnerability)",
    },
    51274: {
        "name": "PT",
        "system": "Coagulation",
        "description": "Prothrombin time in seconds",
        "high_interpretation": "prolonged prothrombin time",
        "low_interpretation": "normal/short prothrombin time",
    },
    51275: {
        "name": "PTT",
        "system": "Coagulation",
        "description": "Partial thromboplastin time evaluating intrinsic clotting pathway",
        "high_interpretation": "prolonged PTT (intrinsic pathway coagulopathy)",
        "low_interpretation": "normal/short PTT",
    },
    51277: {
        "name": "RDW",
        "system": "Hematology / Anisocytosis",
        "description": "Red cell distribution width measuring size variation",
        "high_interpretation": "elevated RDW (high anisocytosis, often associated with physiological stress)",
        "low_interpretation": "normal RDW",
    },
    51279: {
        "name": "Red Blood Cells",
        "system": "Hematology",
        "description": "Erythrocyte count",
        "high_interpretation": "elevated RBC count",
        "low_interpretation": "low RBC count",
    },
    51301: {
        "name": "White Blood Cells (WBC)",
        "system": "Immunology / Inflammation",
        "description": "Total circulating leukocyte count",
        "high_interpretation": "elevated white blood cell count (leukocytosis, potential inflammatory response)",
        "low_interpretation": "depressed white blood cell count (leukopenia)",
    },
}

TEMPORAL_GAP_MAP: Dict[str, Dict[str, Any]] = {
    "TIME_LT1H": {"label": "< 1 hour", "category": "immediate", "hours_max": 1.0, "description": "within less than 1 hour"},
    "TIME_1TO6H": {"label": "1 to 6 hours", "category": "acute", "hours_max": 6.0, "description": "within a 1 to 6 hour interval"},
    "TIME_6TO24H": {"label": "6 to 24 hours", "category": "intermediate", "hours_max": 24.0, "description": "within a 6 to 24 hour interval"},
    "TIME_1TO3D": {"label": "1 to 3 days", "category": "extended", "hours_max": 72.0, "description": "spaced over 1 to 3 days"},
    "TIME_GT3D": {"label": "> 3 days", "category": "long-term", "hours_max": 999.0, "description": "spaced over more than 3 days"},
}


class ClinicalDictionary:
    """Clinical dictionary mapping technical tokens to medically accurate ontological concepts."""

    def __init__(self, d_labitems_csv_path: Optional[str] = None):
        self.d_labitems: Dict[int, Dict[str, Any]] = {}
        self._load_d_labitems(d_labitems_csv_path)

    def _load_d_labitems(self, csv_path: Optional[str]):
        search_paths = [
            csv_path,
            "D_LABITEMS.csv",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "D_LABITEMS.csv"),
            "d:/gitclone/medical_llm/D_LABITEMS.csv",
        ]
        loaded = False
        for p in search_paths:
            if p and os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    for _, row in df.iterrows():
                        itemid = int(row["itemid"])
                        self.d_labitems[itemid] = {
                            "label": str(row.get("label", "")),
                            "fluid": str(row.get("fluid", "")),
                            "category": str(row.get("category", "")),
                            "loinc_code": str(row.get("loinc_code", "")) if pd.notnull(row.get("loinc_code")) else "",
                        }
                    loaded = True
                    break
                except Exception:
                    pass

        # Fallback to curated profiles if file load failed
        if not loaded:
            for itemid, prof in LAB_CLINICAL_PROFILES.items():
                self.d_labitems[itemid] = {
                    "label": prof["name"],
                    "fluid": "Blood",
                    "category": prof["system"],
                    "loinc_code": "",
                }

    def decode_lab_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decodes a technical lab token like 'LAB_50912_Q9' or 'LAB_50813_NORMAL' into a structured clinical concept.
        Distinguishes cohort statistical quantile from clinical reference abnormality.
        """
        if not token.startswith("LAB_"):
            return None

        parts = token.split("_")
        if len(parts) < 3:
            return None

        try:
            itemid = int(parts[1])
        except ValueError:
            return None

        status = "_".join(parts[2:]).upper()

        # Retrieve metadata
        meta = self.d_labitems.get(itemid, {})
        curated = LAB_CLINICAL_PROFILES.get(itemid, {})

        lab_name = meta.get("label") or curated.get("name") or f"Laboratory Test {itemid}"
        system = curated.get("system") or meta.get("category") or "General Chemistry"

        # Dissect status: deciles Q0..Q9 or qualitative NORMAL/ABNORMAL
        direction = "normal"
        statistical_bucket = status
        is_statistical = False
        clinical_interp = "within baseline range"
        significance = "neutral"

        if status.startswith("Q") and status[1:].isdigit():
            is_statistical = True
            q_num = int(status[1:])
            statistical_bucket = f"Q{q_num}"

            if q_num >= 8:
                direction = "high"
                significance = "acute_high"
                clinical_interp = curated.get("high_interpretation", f"elevated ({statistical_bucket}, cohort top decile)")
            elif q_num <= 1:
                direction = "low"
                significance = "acute_low"
                clinical_interp = curated.get("low_interpretation", f"low ({statistical_bucket}, cohort bottom decile)")
            else:
                direction = "baseline"
                significance = "central"
                clinical_interp = f"central cohort distribution ({statistical_bucket})"

        elif status in ("ABNORMAL", "HIGH"):
            direction = "abnormal"
            significance = "flagged_abnormal"
            clinical_interp = curated.get("high_interpretation", "flagged as abnormal by laboratory reference")
        elif status == "NORMAL":
            direction = "normal"
            significance = "normal"
            clinical_interp = "within laboratory reference normal limits"

        distinction_note = (
            f"Statistical quantile {statistical_bucket} denotes position within the cohort distribution, "
            "not a confirmed diagnostic reference range." if is_statistical else "Laboratory qualitative flag."
        )

        return {
            "token": token,
            "itemid": itemid,
            "lab_name": lab_name,
            "system": system,
            "direction": direction,
            "statistical_bucket": statistical_bucket,
            "is_statistical_decile": is_statistical,
            "clinical_interpretation": clinical_interp,
            "significance": significance,
            "distinction_note": distinction_note,
        }

    def decode_temporal_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decodes temporal gap token (e.g. TIME_LT1H) into structured temporal metadata."""
        if token in TEMPORAL_GAP_MAP:
            info = TEMPORAL_GAP_MAP[token]
            return {
                "token": token,
                "label": info["label"],
                "category": info["category"],
                "hours_max": info["hours_max"],
                "description": info["description"],
            }
        return None

    def decode_demographic_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decodes demographic tokens like AGE_70, GENDER_F, ADM_EMERGENCY."""
        if token.startswith("AGE_"):
            val = token.replace("AGE_", "")
            return {"token": token, "type": "age", "value": val, "description": f"patient age group ~{val}s"}
        elif token.startswith("GENDER_"):
            g = token.replace("GENDER_", "")
            label = "Female" if g == "F" else "Male" if g == "M" else g
            return {"token": token, "type": "gender", "value": g, "description": f"{label} gender"}
        elif token.startswith("ADM_"):
            adm = token.replace("ADM_", "").replace("_", " ")
            return {"token": token, "type": "admission_type", "value": adm, "description": f"{adm} admission"}
        return None


_DEFAULT_DICT: Optional[ClinicalDictionary] = None


def get_default_dictionary() -> ClinicalDictionary:
    global _DEFAULT_DICT
    if _DEFAULT_DICT is None:
        _DEFAULT_DICT = ClinicalDictionary()
    return _DEFAULT_DICT
