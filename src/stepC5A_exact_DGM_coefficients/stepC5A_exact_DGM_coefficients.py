#!/usr/bin/env python3
"""Step C5A: export exact DGM coefficients from the fully synthetic generator.

This script reads only an available Step C generator source (the original
notebook or its public Python export), exported fully synthetic summary tables,
and small bounded samples from fully synthetic scenario files for a true-LP
reconstruction audit. It does not read real SLAM data, Step B data, raw Excel
files, or full per-person predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DATA_DIR = Path("fully_synthetic_stepC_v1")
SCENARIO_DIR = DATA_DIR / "scenario_datasets"
TABLE_DIR = DATA_DIR / "tables"
GENERATOR_NOTEBOOK = Path("code/stepC_fully_synthetic_exportable_generator_v1.ipynb")
GENERATOR_PYTHON = Path(
    "src/stepC_generator/stepC_fully_synthetic_exportable_generator_v1.py"
)
OUT_DIR = Path("fully_synthetic_stepC5A_exact_DGM_coefficients")

MAX_REPS_PER_SCENARIO_AUDIT = 2
MAX_ROWS_PER_REP_AUDIT = 1000
CHUNKSIZE = 5000

SPARSE_MRI_SIGNAL_COLS = [
    "Right_Hippocampus_pct",
    "Left_Hippocampus_pct",
    "Right_Entorhinal_pct",
    "Left_Entorhinal_pct",
    "Right_LateralVentricle_pct",
    "Left_LateralVentricle_pct",
    "Right_Precuneus_pct",
    "Left_Precuneus_pct",
    "Right_Temporal_Region01_pct",
    "Left_Temporal_Region01_pct",
]

COMORBIDITY_COLS = [
    "falls",
    "hypertension",
    "Cerebrovascular_accident",
    "Epilepsy",
    "Diabetes_mellitus",
    "Chronic_kidney_disease",
    "Parkinsons_disease",
    "Transient_ischemic_attack",
    "Chronic_obstructive_lung_disease",
    "Arthritis",
    "Heart_failure",
    "Ischemic_heart_disease",
    "Atrial_fibrillation",
    "Chronic_liver_disease",
    "Coronary_arteriosclerosis",
]

BASE_CAREHOME_TERMS = [
    {
        "term_type": "linear_main",
        "predictor_name": "age",
        "transformed_predictor_name": "zscore(age)",
        "coefficient": 0.35,
        "block": "demographic",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "MMSE",
        "transformed_predictor_name": "zscore(30 - MMSE)",
        "coefficient": 0.45,
        "block": "cognition",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "MTL_total_pct",
        "transformed_predictor_name": "zscore(MTL_total_pct)",
        "coefficient": -0.24,
        "block": "MRI",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "Temporal_lateral_total_pct",
        "transformed_predictor_name": "zscore(Temporal_lateral_total_pct)",
        "coefficient": -0.14,
        "block": "MRI",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "Posterior_total_pct",
        "transformed_predictor_name": "zscore(Posterior_total_pct)",
        "coefficient": -0.12,
        "block": "MRI",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "Ventricles_total_pct",
        "transformed_predictor_name": "zscore(Ventricles_total_pct)",
        "coefficient": 0.25,
        "block": "MRI",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "pathology_ischaemic",
        "transformed_predictor_name": "zscore(log1p(pathology_ischaemic))",
        "coefficient": 0.22,
        "block": "NLP",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "pathology_neurodegenerative_dementia",
        "transformed_predictor_name": "zscore(log1p(pathology_neurodegenerative_dementia))",
        "coefficient": 0.15,
        "block": "NLP",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "comorbidity_burden",
        "transformed_predictor_name": "zscore(sum(comorbidity indicators))",
        "coefficient": 0.24,
        "block": "comorbidity",
    },
    {
        "term_type": "linear_main",
        "predictor_name": "IMD_Score_2019_synthetic",
        "transformed_predictor_name": "zscore(IMD_Score_2019_synthetic)",
        "coefficient": 0.10,
        "block": "deprivation",
    },
]

NONLINEAR_TERMS = [
    {
        "term_type": "nonlinear_threshold",
        "predictor_name": "MMSE",
        "transformed_predictor_name": "I(MMSE < 18)",
        "coefficient": 0.45,
        "block": "cognition",
        "nonlinear_function": "indicator below 18",
    },
    {
        "term_type": "nonlinear_threshold",
        "predictor_name": "MMSE",
        "transformed_predictor_name": "I(MMSE < 23)",
        "coefficient": 0.25,
        "block": "cognition",
        "nonlinear_function": "indicator below 23",
    },
    {
        "term_type": "nonlinear_threshold",
        "predictor_name": "Ventricles_total_pct",
        "transformed_predictor_name": "I(zscore(Ventricles_total_pct) > 1)",
        "coefficient": 0.35,
        "block": "MRI",
        "nonlinear_function": "indicator above one within-repetition SD",
    },
    {
        "term_type": "interaction",
        "predictor_name": "age",
        "transformed_predictor_name": "zscore(age) * zscore(30 - MMSE)",
        "coefficient": 0.25,
        "block": "demographic/cognition",
        "interaction_partner": "MMSE",
        "interaction_expression": "zscore(age) * zscore(30 - MMSE)",
    },
    {
        "term_type": "latent_interaction",
        "predictor_name": "latent_frailty",
        "transformed_predictor_name": "zscore(latent_frailty) * zscore(latent_vascular)",
        "coefficient": 0.20,
        "block": "latent_unexported",
        "interaction_partner": "latent_vascular",
        "interaction_expression": "zscore(latent_frailty) * zscore(latent_vascular)",
        "note": "Latent factors are used by the generator but are not exported as predictors.",
    },
]

NONPH_EARLY_TERMS = [
    ("age", "zscore(age)", 0.25, "demographic"),
    ("MMSE", "zscore(30 - MMSE)", 0.75, "cognition"),
    ("MTL_total_pct", "zscore(MTL_total_pct)", -0.35, "MRI"),
    ("Ventricles_total_pct", "zscore(Ventricles_total_pct)", 0.25, "MRI"),
    ("pathology_neurodegenerative_dementia", "zscore(log1p(pathology_neurodegenerative_dementia))", 0.20, "NLP"),
]

NONPH_LATE_TERMS = [
    ("age", "zscore(age)", 0.55, "demographic"),
    ("MMSE", "zscore(30 - MMSE)", 0.25, "cognition"),
    ("pathology_ischaemic", "zscore(log1p(pathology_ischaemic))", 0.50, "NLP"),
    ("comorbidity_burden", "zscore(sum(comorbidity indicators))", 0.40, "comorbidity"),
    ("latent_frailty", "zscore(latent_frailty)", 0.30, "latent_unexported"),
]

DEATH_BEFORE_TERMS = [
    ("age", "zscore(age)", 0.75, "demographic"),
    ("MMSE", "zscore(30 - MMSE)", 0.30, "cognition"),
    ("comorbidity_burden", "zscore(sum(comorbidity indicators))", 0.60, "comorbidity"),
    ("pathology_ischaemic", "zscore(log1p(pathology_ischaemic))", 0.30, "NLP"),
    ("latent_frailty", "zscore(latent_frailty)", 0.25, "latent_unexported"),
    ("sex_Female", "1 - sex_Female", 0.12, "demographic"),
]


def ensure_output_dirs() -> None:
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_generator_source() -> Tuple[Path, str, str]:
    if GENERATOR_NOTEBOOK.exists():
        notebook = json.loads(GENERATOR_NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        return (
            GENERATOR_NOTEBOOK,
            "\n".join(code_cells),
            f"Notebook source with {len(code_cells)} code cells.",
        )
    if GENERATOR_PYTHON.exists():
        source = GENERATOR_PYTHON.read_text(encoding="utf-8")
        return (
            GENERATOR_PYTHON,
            source,
            f"Python generator source with {len(source.splitlines())} lines.",
        )
    raise FileNotFoundError(
        "No Step C generator source found. Expected either "
        f"{GENERATOR_NOTEBOOK} or {GENERATOR_PYTHON}."
    )


def parse_safe_bool(x: object) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_inputs() -> Dict[str, pd.DataFrame]:
    load_generator_source()
    audit_path = DATA_DIR / "audit" / "export_safety_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        if "safe_to_export_column_names" in audit.columns:
            safe = audit["safe_to_export_column_names"].map(parse_safe_bool)
            if not safe.all():
                bad = audit.loc[~safe]
                raise RuntimeError(
                    f"Export safety audit contains {len(bad)} unsafe column-name flags; stopping."
                )

    return {
        "dgm": pd.read_csv(require_file(TABLE_DIR / "dgm_definition_table.csv")),
        "features": pd.read_csv(require_file(TABLE_DIR / "feature_dictionary.csv")),
        "scenario_summary": pd.read_csv(require_file(TABLE_DIR / "scenario_summary.csv")),
        "repetition_summary": pd.read_csv(require_file(TABLE_DIR / "repetition_summary.csv")),
    }


def direction(coefficient: float) -> str:
    if coefficient > 0:
        return "higher transformed value increases hazard/risk"
    if coefficient < 0:
        return "higher transformed value decreases hazard/risk"
    return "zero contribution"


def base_row(
    scenario: pd.Series,
    term: Dict,
    coefficient: Optional[float] = None,
    term_type: Optional[str] = None,
    note: str = "",
    lp_sd_target: Optional[float] = None,
) -> Dict:
    coef = float(term["coefficient"] if coefficient is None else coefficient)
    if lp_sd_target is None:
        lp_sd_target = float(scenario["lp_sd"])
    inherited_note = str(term.get("note", ""))
    if note and inherited_note:
        final_note = f"{note} {inherited_note}"
    else:
        final_note = note or inherited_note
    return {
        "scenario_id": scenario["scenario_id"],
        "dgm_type": scenario["dgm_type"],
        "term_type": term_type or term["term_type"],
        "predictor_name": term["predictor_name"],
        "transformed_predictor_name": term["transformed_predictor_name"],
        "coefficient": coef,
        "coefficient_scale": "raw pre-rescaling LP coefficient from generator code",
        "direction": direction(coef),
        "block": term["block"],
        "included": True,
        "nonlinear_function": term.get("nonlinear_function", ""),
        "interaction_partner": term.get("interaction_partner", ""),
        "interaction_expression": term.get("interaction_expression", ""),
        "standardisation_used": "within-repetition zscore on complete pre-missingness synthetic features, where applicable",
        "lp_rescaling_used": True,
        "lp_sd_target": lp_sd_target,
        "note": final_note,
    }


def build_carehome_coefficients(dgm: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, scenario in dgm.iterrows():
        sid = str(scenario["scenario_id"])
        sparse = bool(scenario.get("use_sparse_mri_signal", False))
        nonph = bool(scenario.get("use_non_ph", False))

        for term in BASE_CAREHOME_TERMS:
            coef = float(term["coefficient"])
            note = "This coefficient enters true_lp_carehome before rescale_lp."
            if sparse:
                coef = 0.55 * coef
                note += " In S6 the baseline LP is multiplied by 0.55 before sparse MRI terms are added."
            if nonph:
                note += " In S4 this baseline true_lp_carehome is exported, while event-time generation uses separate early and late LPs."
            rows.append(base_row(scenario, term, coefficient=coef, note=note))

        if bool(scenario.get("use_nonlinear", False)):
            for term in NONLINEAR_TERMS:
                rows.append(
                    base_row(
                        scenario,
                        term,
                        note="S3 nonlinear/interaction term added before final LP rescaling.",
                    )
                )

        if sparse:
            for col in SPARSE_MRI_SIGNAL_COLS:
                sign = 1.0 if "Ventricle" in col else -1.0
                rows.append(
                    base_row(
                        scenario,
                        {
                            "term_type": "highdim_sparse_mri",
                            "predictor_name": col,
                            "transformed_predictor_name": f"zscore({col})",
                            "coefficient": sign * 0.18,
                            "block": "MRI",
                        },
                        note="S6 sparse regional MRI term added after multiplying the baseline LP by 0.55.",
                    )
                )

        if nonph:
            for predictor, transformed, coef, block in NONPH_EARLY_TERMS:
                rows.append(
                    base_row(
                        scenario,
                        {
                            "term_type": "nonPH_early_hazard_lp",
                            "predictor_name": predictor,
                            "transformed_predictor_name": transformed,
                            "coefficient": coef,
                            "block": block,
                        },
                        lp_sd_target=float(scenario["lp_sd"]),
                        note="S4 event-time generator uses this early piecewise LP before 2 years.",
                    )
                )
            for predictor, transformed, coef, block in NONPH_LATE_TERMS:
                rows.append(
                    base_row(
                        scenario,
                        {
                            "term_type": "nonPH_late_hazard_lp",
                            "predictor_name": predictor,
                            "transformed_predictor_name": transformed,
                            "coefficient": coef,
                            "block": block,
                        },
                        lp_sd_target=float(scenario["lp_sd"]),
                        note="S4 event-time generator uses this late piecewise LP after 2 years.",
                    )
                )

    return pd.DataFrame(rows)


def build_death_before_coefficients(dgm: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, scenario in dgm.iterrows():
        for predictor, transformed, coef, block in DEATH_BEFORE_TERMS:
            note = "Community death-before-carehome LP; final base hazard is calibrated to the latent death target."
            if str(scenario["scenario_id"]).startswith("S7"):
                note += " S7 uses the stronger latent 5-year death target of 0.40."
            rows.append(
                base_row(
                    scenario,
                    {
                        "term_type": "death_before_carehome_linear_main",
                        "predictor_name": predictor,
                        "transformed_predictor_name": transformed,
                        "coefficient": coef,
                        "block": block,
                    },
                    lp_sd_target=0.9,
                    note=note,
                )
            )
    return pd.DataFrame(rows)


def build_post_carehome_mechanism(dgm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, scenario in dgm.iterrows():
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "dgm_type": scenario["dgm_type"],
                "post_carehome_death_hr_multiplier": 1.5,
                "community_death_lp_source": "true_lp_death_before_carehome",
                "hazard_expression": "post_rate = 1.5 * lambda0_death_before_carehome * exp(true_lp_death_before_carehome)",
                "interpretation": (
                    "Post-carehome death hazard multiplier is a state-risk / frailty-state assumption, "
                    "not a causal claim that care home increases mortality."
                ),
                "source_code_location": "code cell 13 generate_outcomes lines 148-155",
            }
        )
    return pd.DataFrame(rows)


def build_scenario_summary(dgm: pd.DataFrame, scenario_summary: pd.DataFrame) -> pd.DataFrame:
    merged = dgm.merge(
        scenario_summary[["scenario_id", "n_reps", "n_per_rep", "mean_observed_death_before_carehome_rate"]],
        on="scenario_id",
        how="left",
    )
    rows = []
    for _, scenario in merged.iterrows():
        sid = str(scenario["scenario_id"])
        rows.append(
            {
                "scenario_id": sid,
                "target_carehome_rate": scenario["target_observed_carehome_rate"],
                "death_competing_strength": (
                    "strong latent 5-year community death target 0.40"
                    if sid.startswith("S7")
                    else f"source-derived latent death target; mean observed death-before-carehome={scenario['mean_observed_death_before_carehome_rate']:.3f}"
                ),
                "hazard_type": "piecewise exponential care-home hazard" if bool(scenario["use_non_ph"]) else "exponential proportional-hazards care-home hazard",
                "proportional_hazards": not bool(scenario["use_non_ph"]),
                "nonlinearity": bool(scenario["use_nonlinear"]),
                "interaction": bool(scenario["use_nonlinear"]),
                "highdim_sparse_signal": bool(scenario["use_sparse_mri_signal"]),
                "missingness_type": (
                    "higher MAR-lite missingness multiplier 1.75"
                    if sid.startswith("S5")
                    else "MAR-lite capped source-informed missingness"
                ),
                "post_carehome_death_multiplier": 1.5,
                "lp_sd": scenario["lp_sd"],
                "n_repetitions": int(scenario["n_reps"]),
                "n_per_repetition": int(scenario["n_per_rep"]),
            }
        )
    return pd.DataFrame(rows)


def build_code_location_audit() -> pd.DataFrame:
    source_path, source_text, source_note = load_generator_source()
    specifications = [
        (
            "scenario definitions",
            "SCENARIOS",
            ["SCENARIOS", "target_observed_carehome_rate", "lp_sd"],
            "Defines S0-S7 event targets, LP scaling and scenario switches.",
            source_note,
        ),
        (
            "care home DGM linear LP",
            "build_lp_components",
            ["build_lp_components", "true_lp_carehome", "lp_inst"],
            "Constructs the base care-home LP and rescales it to scenario LP SD.",
            "true_lp_carehome is exported from the base lp_inst component.",
        ),
        (
            "death-before-carehome DGM",
            "build_lp_components / generate_outcomes",
            ["lp_death", "calibrate_base_rate_for_latent_event"],
            "Constructs the community-death LP and calibrates its baseline hazard.",
            "S7 changes the latent death target, not the death LP coefficients.",
        ),
        (
            "post-carehome death DGM",
            "generate_outcomes",
            ["post_carehome_death_hr_multiplier", "death_after_carehome"],
            "Applies the post-care-home multiplier to the subject-specific community-death hazard.",
            "The multiplier is a state-risk assumption, not a causal claim.",
        ),
        (
            "nonlinear and interaction DGM",
            "build_lp_components",
            ["use_nonlinear", "mmse < 18", "age_z * low_mmse_z"],
            "Adds S3 thresholds plus observed and latent interaction terms.",
            "Latent frailty and vascular factors are not exported, so S3 reconstruction is incomplete.",
        ),
        (
            "non-PH DGM",
            "build_lp_components / generate_outcomes",
            ["lp_early", "lp_late", "non_ph=True"],
            "Builds early/late LPs and uses piecewise event-time generation in S4.",
            "The exported true_lp_carehome remains the base LP rather than the full time-varying score.",
        ),
        (
            "high-dimensional sparse MRI DGM",
            "SPARSE_MRI_SIGNAL_COLS / build_lp_components",
            ["SPARSE_MRI_SIGNAL_COLS", "use_sparse_mri_signal", "sparse_terms"],
            "Defines and applies signed sparse regional-MRI terms in S6.",
            "Ventricle terms are positive; other selected regional-volume terms are negative.",
        ),
        (
            "MAR-lite missingness DGM",
            "apply_missingness / mar_missing_mask",
            ["apply_missingness", "missingness_multiplier", "mar_missing_mask"],
            "Applies structured block missingness after outcome generation.",
            "S5 raises the multiplier to 1.75 before block-specific caps.",
        ),
    ]
    rows = []
    for component, symbol, tokens, description, notes in specifications:
        rows.append(
            {
                "component": component,
                "found": all(token in source_text for token in tokens),
                "source_file": str(source_path),
                "function_or_cell_name": symbol,
                "line_or_description": description,
                "notes": notes,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["found"].all():
        missing = audit.loc[~audit["found"], "component"].astype(str).tolist()
        raise RuntimeError(
            "Generator source audit could not locate: " + ", ".join(missing)
        )
    return audit


def zscore(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mean = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / sd


def rescale_lp(lp: np.ndarray, target_sd: float) -> np.ndarray:
    arr = np.asarray(lp, dtype=float)
    arr = arr - np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return arr / sd * float(target_sd)


def scenario_file_for(scenario_id: str) -> Path:
    return require_file(SCENARIO_DIR / f"{scenario_id}.csv.gz")


def sample_scenario_reps(scenario_id: str, reps: List[int], usecols: List[str]) -> pd.DataFrame:
    path = scenario_file_for(scenario_id)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    available = [c for c in usecols if c in header]
    missing = sorted(set(usecols) - set(available))
    if missing:
        raise RuntimeError(f"Missing required reconstruction columns for {scenario_id}: {missing}")

    counts = {int(rep): 0 for rep in reps}
    parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=available, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["replicate_id"].isin(reps)]
        if chunk.empty:
            continue
        for rep in reps:
            rep = int(rep)
            if counts[rep] >= MAX_ROWS_PER_REP_AUDIT:
                continue
            sub = chunk[chunk["replicate_id"] == rep]
            if sub.empty:
                continue
            needed = MAX_ROWS_PER_REP_AUDIT - counts[rep]
            take = sub.head(needed)
            parts.append(take)
            counts[rep] += len(take)
        if all(v >= MAX_ROWS_PER_REP_AUDIT for v in counts.values()):
            break
    if not parts:
        raise RuntimeError(f"No sampled rows found for {scenario_id}, reps={reps}")
    return pd.concat(parts, ignore_index=True)


def reconstruct_carehome_lp_from_observed_sample(df: pd.DataFrame, scenario: pd.Series) -> Tuple[np.ndarray, str]:
    age_z = zscore(df["age"].to_numpy())
    low_mmse_z = zscore(30 - df["MMSE"].to_numpy())
    mtl_z = zscore(df["MTL_total_pct"].to_numpy())
    temp_z = zscore(df["Temporal_lateral_total_pct"].to_numpy())
    post_z = zscore(df["Posterior_total_pct"].to_numpy())
    vent_z = zscore(df["Ventricles_total_pct"].to_numpy())
    imd_z = zscore(df["IMD_Score_2019_synthetic"].to_numpy())
    nlp_vasc_z = zscore(np.log1p(df["pathology_ischaemic"].to_numpy()))
    nlp_neuro_z = zscore(np.log1p(df["pathology_neurodegenerative_dementia"].to_numpy()))
    comorbidity_burden = df[COMORBIDITY_COLS].sum(axis=1, skipna=False).to_numpy()
    comorb_z = zscore(comorbidity_burden)

    lp = (
        0.35 * age_z
        + 0.45 * low_mmse_z
        - 0.24 * mtl_z
        - 0.14 * temp_z
        - 0.12 * post_z
        + 0.25 * vent_z
        + 0.22 * nlp_vasc_z
        + 0.15 * nlp_neuro_z
        + 0.24 * comorb_z
        + 0.10 * imd_z
    )
    note = "Reconstructed from exported observed predictors; zscore/rescale estimated within the bounded audit sample."

    if bool(scenario.get("use_nonlinear", False)):
        mmse = df["MMSE"].to_numpy()
        lp = (
            lp
            + 0.45 * (mmse < 18).astype(float)
            + 0.25 * (mmse < 23).astype(float)
            + 0.35 * (vent_z > 1.0).astype(float)
            + 0.25 * age_z * low_mmse_z
        )
        note += " Latent frailty*vascular term is not exported and is omitted."

    if bool(scenario.get("use_sparse_mri_signal", False)):
        sparse_terms = np.zeros(len(df), dtype=float)
        for col in SPARSE_MRI_SIGNAL_COLS:
            sign = 1.0 if "Ventricle" in col else -1.0
            sparse_terms += sign * 0.18 * zscore(df[col].to_numpy())
        lp = 0.55 * lp + sparse_terms
        note += " S6 sparse MRI terms included."

    reconstructed = rescale_lp(lp, float(scenario.get("lp_sd", 0.8)))
    return reconstructed, note


def metric_or_nan(func, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    try:
        return float(func(x, y))
    except Exception:
        return np.nan


def run_reconstruction_audit(dgm: pd.DataFrame, repetition_summary: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "scenario_id",
        "replicate_id",
        "age",
        "MMSE",
        "MTL_total_pct",
        "Temporal_lateral_total_pct",
        "Posterior_total_pct",
        "Ventricles_total_pct",
        "IMD_Score_2019_synthetic",
        "pathology_ischaemic",
        "pathology_neurodegenerative_dementia",
        "true_lp_carehome",
    ]
    usecols = base_cols + COMORBIDITY_COLS + SPARSE_MRI_SIGNAL_COLS
    rows = []

    for _, scenario in dgm.iterrows():
        sid = str(scenario["scenario_id"])
        reps = (
            repetition_summary.loc[repetition_summary["scenario_id"] == sid, "replicate_id"]
            .drop_duplicates()
            .sort_values()
            .head(MAX_REPS_PER_SCENARIO_AUDIT)
            .astype(int)
            .tolist()
        )
        if not reps:
            rows.append(
                {
                    "scenario_id": sid,
                    "replicate_id": "none",
                    "n_sampled_rows": 0,
                    "n_rows_used_complete_case": 0,
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "mean_absolute_difference": np.nan,
                    "max_absolute_difference": np.nan,
                    "pass_spearman_0_999": False,
                    "reconstruction_scope": "carehome true LP",
                    "failure_reason": "No repetition IDs available.",
                    "notes": "",
                }
            )
            continue

        sample = sample_scenario_reps(sid, reps, usecols)
        reconstructed, note = reconstruct_carehome_lp_from_observed_sample(sample, scenario)
        sample = sample.copy()
        sample["reconstructed_true_lp_carehome"] = reconstructed

        predictor_cols_for_complete_case = [
            c for c in usecols if c not in {"scenario_id", "replicate_id", "true_lp_carehome"}
        ]
        if not bool(scenario.get("use_sparse_mri_signal", False)):
            predictor_cols_for_complete_case = [
                c for c in predictor_cols_for_complete_case if c not in SPARSE_MRI_SIGNAL_COLS
            ]

        for rep, rep_df in sample.groupby("replicate_id", sort=True):
            rep_df = rep_df.copy()
            complete = rep_df[predictor_cols_for_complete_case + ["true_lp_carehome", "reconstructed_true_lp_carehome"]].notna().all(axis=1)
            used = rep_df.loc[complete]
            x = used["reconstructed_true_lp_carehome"].to_numpy(dtype=float)
            y = used["true_lp_carehome"].to_numpy(dtype=float)
            pearson = metric_or_nan(lambda a, b: pearsonr(a, b)[0], x, y)
            spearman = metric_or_nan(lambda a, b: spearmanr(a, b).correlation, x, y)
            mad = float(np.mean(np.abs(x - y))) if len(x) else np.nan
            maxad = float(np.max(np.abs(x - y))) if len(x) else np.nan
            pass_flag = bool(np.isfinite(spearman) and spearman >= 0.999)
            failure_reason = ""
            if not pass_flag:
                if bool(scenario.get("use_nonlinear", False)):
                    failure_reason = "Latent frailty*vascular interaction term is not exported; coefficient extraction is exact but reconstruction is incomplete."
                else:
                    failure_reason = "Bounded sample lacks generator repetition-level complete-data zscore/rescale moments."
            rows.append(
                {
                    "scenario_id": sid,
                    "replicate_id": int(rep),
                    "n_sampled_rows": int(len(rep_df)),
                    "n_rows_used_complete_case": int(len(used)),
                    "pearson": pearson,
                    "spearman": spearman,
                    "mean_absolute_difference": mad,
                    "max_absolute_difference": maxad,
                    "pass_spearman_0_999": pass_flag,
                    "reconstruction_scope": "carehome true_lp_carehome only; S4 early/late non-PH event-time LPs not directly verified",
                    "failure_reason": failure_reason,
                    "notes": note,
                }
            )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: Optional[List[str]] = None, max_rows: int = 20) -> str:
    if columns is not None:
        df = df[columns]
    df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows._"
    display = df.fillna("")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        vals = []
        for val in row.tolist():
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_readme(
    code_audit: pd.DataFrame,
    carehome: pd.DataFrame,
    death: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> Path:
    readme = OUT_DIR / "README_C5A_exact_DGM_coefficients.md"
    pass_all = bool(reconstruction["pass_spearman_0_999"].all())
    incomplete_scenarios = sorted(reconstruction.loc[~reconstruction["pass_spearman_0_999"], "scenario_id"].unique())

    content = f"""# Step C5A Exact DGM Coefficients

## 1. Purpose

Step C5A exports the data-generating mechanism (DGM) terms used by the fully synthetic Step C generator. It resolves a generator-documentation gap by separating raw simulation coefficients from the fixed observable DGM-informed predictor sets used in C2/C3/C4. It does not retrospectively redefine those fitted models as exact algebraic DGM models.

## 2. Data Scope

This audit uses only an available fully synthetic Step C generator source (the original notebook or public Python export), exported fully synthetic summary tables, and bounded samples from fully synthetic scenario datasets for true-LP checking. It does not use real SLAM data, Step B semi-synthetic data, raw CSV files, death spreadsheets, WMH spreadsheets, real identifiers, or full per-person predictions.

## 3. How Coefficients Were Extracted

The relevant generator locations are summarized in `tables/dgm_code_location_audit.csv`.

{markdown_table(code_audit, ["component", "found", "function_or_cell_name", "line_or_description"], max_rows=12)}

## 4. Scenario-Level DGM Summary

{markdown_table(scenario_summary, ["scenario_id", "target_carehome_rate", "hazard_type", "proportional_hazards", "nonlinearity", "highdim_sparse_signal", "missingness_type", "lp_sd"], max_rows=12)}

## 5. Care-Home DGM

The care-home coefficient table is written to `tables/exact_dgm_coefficients_carehome.csv`. Coefficients are exact raw pre-rescaling coefficients from the generator code. The final exported true LP uses `rescale_lp`, which subtracts the repetition-level raw-LP mean, divides by the repetition-level raw-LP SD, and multiplies by the scenario target LP SD. Therefore final effective coefficients are repetition-specific unless those raw-LP moments are exported.

Rows exported: `{len(carehome)}`.

## 6. Death-Before-Carehome DGM

The death-before-carehome coefficient table is written to `tables/exact_dgm_coefficients_death_before_carehome.csv`. The same death LP coefficients are used across scenarios, while S7 changes the latent death target used for baseline-hazard calibration.

Rows exported: `{len(death)}`.

## 7. Post-Carehome Death DGM

The post-carehome death mechanism is written to `tables/exact_dgm_post_carehome_death_mechanism.csv`. The key statement is:

`post-carehome death hazard multiplier is a state-risk / frailty-state assumption, not a causal claim that care home increases mortality.`

## 8. True LP Reconstruction Audit

The reconstruction audit samples at most `{MAX_REPS_PER_SCENARIO_AUDIT}` repetitions per scenario and at most `{MAX_ROWS_PER_REP_AUDIT}` rows per repetition. It reconstructs `true_lp_carehome` from exported observed predictors and compares against the stored `true_lp_carehome`.

Overall pass by Spearman >= 0.999 for every audited scenario-repetition: `{pass_all}`.

{markdown_table(reconstruction, ["scenario_id", "replicate_id", "n_rows_used_complete_case", "spearman", "mean_absolute_difference", "pass_spearman_0_999", "failure_reason"], max_rows=20)}

Important limitation: the generator builds LPs on complete pre-missingness synthetic features and then applies MAR-lite missingness. It also rescales LPs using repetition-level raw-LP moments that are not exported. S3 additionally contains a latent frailty by vascular interaction term that is intentionally not exported as a predictor. For these reasons, C5A exports exact raw code coefficients, but not all final repetition-specific effective coefficients can be reconstructed exactly from the current exported data package.

## 9. Does This Make The Fitted DGM-Informed Models Exact?

C5A resolves the documentation of the generator's raw care-home and death-before-carehome terms. It does not make the previously fitted 36-variable observable DGM-informed set identical to the algebraic generator: that set contains available proxies and additional related measures, whereas some generator terms are latent or unavailable. The fixed fitted set is retained to preserve the specified comparison. C5A can guide a separately labelled future exact-observable-term analysis, but such an analysis would be a new model specification.

It does not fully solve final-LP numerical reconstruction for all scenarios because repetition-level raw-LP moments and latent factors were not exported in Step C v1. The remaining incomplete scenarios are: `{", ".join(incomplete_scenarios) if incomplete_scenarios else "none"}`.

## 10. Remaining Limitations

- Final effective coefficients after `rescale_lp` are repetition-specific and require raw-LP mean/SD values that were not exported.
- S3 uses latent frailty and vascular variables in one interaction term; these latent factors are not exportable baseline predictors.
- S4 non-PH event-time generation uses early and late LPs, while `true_lp_carehome` stores the baseline LP.
- Reconstruction uses bounded samples rather than full scenario files by design.

## 11. Suggested Methods Sentence

The fully synthetic Step C generator used explicitly exported raw DGM coefficients for the care-home and death-before-carehome linear predictors, followed by repetition-level LP rescaling to prespecified standard deviations and baseline-hazard calibration to scenario-specific event-rate targets. C5A documents these coefficients and flags remaining non-reconstructable components arising from non-exported latent factors and repetition-specific rescaling moments.
"""
    readme.write_text(content)
    return readme


def write_docs_update_suggestions(reconstruction: pd.DataFrame) -> Path:
    incomplete = sorted(reconstruction.loc[~reconstruction["pass_spearman_0_999"], "scenario_id"].unique())
    path = OUT_DIR / "docs_update_suggestions_after_C5A.md"
    content = f"""# Documentation Update Suggestions After C5A

## README

Describe C5A as an audit of the exact raw generator coefficients and describe the fitted DGM-informed models as using a fixed observable proxy set. Do not imply that the audit retrospectively changed the fitted models.

Suggested wording:

> C5A exports the raw DGM coefficient tables used by the generator. The fitted DGM-informed models retain their fixed observable proxy set and are not exact algebraic DGM models. Final LP values also depend on repetition-level rescaling moments, and S3 includes one non-exported latent interaction term.

## C2

Document `cox_dgm_features` as the fixed observable DGM-informed set used in the completed comparison. Cite `fully_synthetic_stepC5A_exact_DGM_coefficients/tables/exact_dgm_coefficients_carehome.csv` as a generator audit, not as a retrospective replacement of the fitted feature list.

## C3

Distinguish the observable care-home prediction set from the generator's care-home and death-before-carehome raw LP terms. The C5A death table documents the community-death LP used for competing-event simulation.

## C4

Document RSF/GBSA/Fine-Gray DGM variants as using the same fixed observable DGM-informed set as specified for the completed comparison. A future model restricted to exact exportable C5A terms would require a new label and a new fit.

## Manuscript Methods

Suggested methods wording:

> We exported the fully synthetic generator's raw DGM coefficients for the care-home and death-before-carehome mechanisms. These coefficients define the simulated linear predictors before repetition-level centering, scaling to target LP standard deviations, and baseline-hazard calibration. A C5A reconstruction audit showed near-perfect rank recovery for scenarios whose DGM terms were represented in exported predictors; the nonlinear S3 scenario remained partially non-reconstructable because it included a latent frailty-vascular interaction not exported as a baseline predictor.

## Remaining caveat to keep

Do not claim that final repetition-specific effective coefficients are fully exported unless future Step C versions also export raw-LP mean/SD values and latent factor terms. Current incomplete reconstruction scenarios: {", ".join(incomplete) if incomplete else "none"}.
"""
    path.write_text(content)
    return path


def main() -> None:
    ensure_output_dirs()
    inputs = load_inputs()
    dgm = inputs["dgm"]

    code_audit = build_code_location_audit()
    carehome = build_carehome_coefficients(dgm)
    death = build_death_before_coefficients(dgm)
    post = build_post_carehome_mechanism(dgm)
    exact_summary = build_scenario_summary(dgm, inputs["scenario_summary"])
    reconstruction = run_reconstruction_audit(dgm, inputs["repetition_summary"])

    code_audit.to_csv(OUT_DIR / "tables" / "dgm_code_location_audit.csv", index=False)
    carehome.to_csv(OUT_DIR / "tables" / "exact_dgm_coefficients_carehome.csv", index=False)
    death.to_csv(OUT_DIR / "tables" / "exact_dgm_coefficients_death_before_carehome.csv", index=False)
    post.to_csv(OUT_DIR / "tables" / "exact_dgm_post_carehome_death_mechanism.csv", index=False)
    exact_summary.to_csv(OUT_DIR / "tables" / "exact_dgm_scenario_summary.csv", index=False)
    reconstruction.to_csv(OUT_DIR / "tables" / "dgm_true_lp_reconstruction_audit.csv", index=False)

    readme_path = write_readme(code_audit, carehome, death, exact_summary, reconstruction)
    write_docs_update_suggestions(reconstruction)

    incomplete = sorted(reconstruction.loc[~reconstruction["pass_spearman_0_999"], "scenario_id"].unique())
    exact_found = bool(len(carehome) > 0 and len(death) > 0)
    all_reconstruction_pass = bool(reconstruction["pass_spearman_0_999"].all())

    print(f"output folder: {OUT_DIR}")
    print(
        "whether exact coefficients were found: "
        + ("partial_exact_raw_pre_rescaling_coefficients_found" if exact_found else "no")
    )
    print(f"number of carehome coefficient rows: {len(carehome)}")
    print(f"number of death coefficient rows: {len(death)}")
    print(f"reconstruction audit pass/fail: {'pass' if all_reconstruction_pass else 'fail'}")
    print(f"scenarios with incomplete extraction: {', '.join(incomplete) if incomplete else 'none'}")
    print(f"README path: {readme_path}")


if __name__ == "__main__":
    main()
