#!/usr/bin/env python
# coding: utf-8

# # Step C v1 — Fully synthetic exportable data generator
#
# This notebook generates fully synthetic SLAM-like multi-state prognosis datasets. It reads the internal SLAM semi-synthetic file only to estimate broad summary information, then exports synthetic data only.
#
# Run all cells from top to bottom. The default output folder is `fully_synthetic_stepC_v1/`.

# In[1]:


# -*- coding: utf-8 -*-
"""
Step C v1: Fully synthetic exportable multi-state simulation data generator
============================================================================

Purpose
-------
Generate fully synthetic, exportable SLAM-like dementia prognosis datasets.
This script DOES NOT export real SLAM patient-level rows. It reads an internal
SLAM semi-synthetic dataset only to estimate broad summary quantities such as
missingness rates, feature ranges, and death-time distribution parameters.

Inputs
------
The original generator is intended to run inside the authorised internal
environment where the semi-synthetic summary source exists. That source file is
not included in this GitHub repository.

Outputs
-------
The default generated output is ``fully_synthetic_stepC_v1/``, containing
synthetic scenario datasets, audit tables, and metadata tables. Large generated
scenario datasets are not tracked in GitHub.

Run
---
This script is included for methods transparency. It is not required to run C2
or C3 if an audited ``fully_synthetic_stepC_v1/`` package is already available
locally.

Confirmed design
----------------
- Generate inside SLAM, export only synthetic data.
- Clinically interpretable synthetic feature names.
- N_PER_REP = 5000.
- N_REPS = 50.
- Scenarios S0-S7.
- Multi-state DGM:
    State 0: baseline/community
    State 1: care home entry / institutionalisation
    State 2: death before care home
    State 3: death after care home
- Q3A: institutionalisation target means OBSERVED care-home entry rate after
  accounting for death before care home.
- Q5A: post-carehome death hazard multiplier = 1.5 x community death hazard.
- Missingness: MAR-lite using SLAM-informed block missingness with caps.
- Step C only generates datasets and summary tables; it does not run model comparison.

Important limitation
--------------------
No real care-home entry date is available. Therefore, the distribution of death
before vs after care-home entry is generated under explicit multi-state assumptions.
The death-time distribution is informed by observed SLAM death dates when these
are available in the internal input dataset.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass


# In[2]:


# =============================================================================
# CONFIG


# In[3]:


# =============================================================================

@dataclass
class StepCConfig:
    # Internal SLAM-derived source used only for summary estimation.
    # This file should stay inside SLAM. It is NOT exported.
    source_csv: str = os.environ.get("STEPC_INTERNAL_SOURCE_CSV", "")

    # Source-derived summaries are internal audit material. They must never be
    # written inside the exportable fully synthetic package.
    internal_summary_dir: str = os.environ.get(
        "STEPC_INTERNAL_AUDIT_DIR",
        "stepC_internal_source_summary_DO_NOT_EXPORT",
    )

    # Output folder contains only fully synthetic data + summary tables.
    out_dir: str = "fully_synthetic_stepC_v1"

    n_per_rep: int = 5000
    n_reps: int = 50
    horizon_years: float = 5.0
    random_seed: int = 42

    # Multi-state assumption confirmed by user.
    post_carehome_death_hr_multiplier: float = 1.5

    # If observed SLAM 5-year death rate cannot be estimated, use this fallback.
    fallback_death_5y_rate: float = 0.23

    # Strong death scenario S7 target for latent community death within 5 years.
    s7_latent_death_5y_rate: float = 0.40

    # Missingness caps. Real SLAM block missingness is estimated internally and capped.
    cap_missing_age: float = 0.00
    cap_missing_sex: float = 0.02
    cap_missing_mmse: float = 0.10
    cap_missing_mri_composite: float = 0.05
    cap_missing_mri_regional: float = 0.08
    cap_missing_comorbidity: float = 0.02
    cap_missing_deprivation: float = 0.10
    cap_missing_nlp_raw: float = 0.45
    cap_missing_wmh: float = 0.15

    # Output format.
    write_csv_gz: bool = True
    save_per_rep_files: bool = False  # usually False; each scenario file contains all reps.


CFG = StepCConfig()


# In[4]:


# =============================================================================
# SCENARIOS


# In[5]:


# =============================================================================

SCENARIOS = [
    {
        "scenario_id": "S0_linear_PH_inst30",
        "description": "Linear proportional hazards; observed 5-year care-home entry target 30%.",
        "dgm_type": "linear_PH",
        "target_observed_carehome_rate": 0.30,
        "lp_sd": 0.8,
        "missingness_multiplier": 1.0,
        "use_nonlinear": False,
        "use_non_ph": False,
        "use_sparse_mri_signal": False,
        "death_multiplier": 1.0,
    },
    {
        "scenario_id": "S1_linear_PH_inst15",
        "description": "Linear proportional hazards; low observed 5-year care-home entry target 15%.",
        "dgm_type": "linear_PH_low_event",
        "target_observed_carehome_rate": 0.15,
        "lp_sd": 0.8,
        "missingness_multiplier": 1.0,
        "use_nonlinear": False,
        "use_non_ph": False,
        "use_sparse_mri_signal": False,
        "death_multiplier": 1.0,
    },
    {
        "scenario_id": "S2_linear_PH_inst45",
        "description": "Linear proportional hazards; high observed 5-year care-home entry target 45%.",
        "dgm_type": "linear_PH_high_event",
        "target_observed_carehome_rate": 0.45,
        "lp_sd": 0.8,
        "missingness_multiplier": 1.0,
        "use_nonlinear": False,
        "use_non_ph": False,
        "use_sparse_mri_signal": False,
        "death_multiplier": 1.0,
    },
    {
        "scenario_id": "S3_nonlinear_interaction_inst30",
        "description": "Nonlinear + interaction DGM; observed 5-year care-home entry target 30%.",
        "dgm_type": "nonlinear_interaction",
        "target_observed_carehome_rate": 0.30,
        "lp_sd": 1.0,
        "missingness_multiplier": 1.0,
        "use_nonlinear": True,
        "use_non_ph": False,
        "use_sparse_mri_signal": False,
        "death_multiplier": 1.0,
    },
    {
        "scenario_id": "S4_nonPH_inst30",
        "description": "Non-proportional hazards: early cognition effect and later age/vascular effect; target 30% observed care-home entry.",
        "dgm_type": "non_proportional_hazards",
        "target_observed_carehome_rate": 0.30,
        "lp_sd": 0.9,
        "missingness_multiplier": 1.0,
        "use_nonlinear": False,
        "use_non_ph": True,
        "use_sparse_mri_signal": False,
        "death_multiplier": 1.0,
    },
    {
        "scenario_id": "S5_MAR_missingness_inst30",
        "description": "Linear PH with higher MAR-lite missingness; target 30% observed care-home entry.",
        "dgm_type": "high_missingness_MAR",
        "target_observed_carehome_rate": 0.30,
        "lp_sd": 0.8,
        "missingness_multiplier": 1.75,
        "use_nonlinear": False,
        "use_non_ph": False,
        "use_sparse_mri_signal": False,
        "death_multiplier": 1.0,
    },
    {
        "scenario_id": "S6_highdim_sparseMRI_inst30",
        "description": "High-dimensional MRI features with sparse true MRI signal; target 30% observed care-home entry.",
        "dgm_type": "high_dimensional_sparse_MRI_signal",
        "target_observed_carehome_rate": 0.30,
        "lp_sd": 0.9,
        "missingness_multiplier": 1.0,
        "use_nonlinear": False,
        "use_non_ph": False,
        "use_sparse_mri_signal": True,
        "death_multiplier": 1.0,
    },
    {
        "scenario_id": "S7_strong_death_competing_inst30",
        "description": "Linear PH with stronger death-before-carehome competing risk; target 30% observed care-home entry.",
        "dgm_type": "strong_death_competing_risk",
        "target_observed_carehome_rate": 0.30,
        "lp_sd": 0.8,
        "missingness_multiplier": 1.0,
        "use_nonlinear": False,
        "use_non_ph": False,
        "use_sparse_mri_signal": False,
        "death_multiplier": 1.0,  # overwritten by S7 death target calibration
    },
]


# In[6]:


# =============================================================================
# UTILS


# In[7]:


# =============================================================================

def ensure_dirs(out_dir: Path) -> None:
    for sub in ["scenario_datasets", "tables", "audit"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def prepare_internal_audit_dir(
    out_dir: Path,
    internal_audit_dir: Path,
) -> Path:
    """Create an internal-only audit directory outside the export package."""

    export_root = out_dir.resolve()
    audit_root = internal_audit_dir.resolve()
    if audit_root == export_root or export_root in audit_root.parents:
        raise RuntimeError(
            "The internal source-summary directory must be outside the "
            "exportable Step C package."
        )
    audit_root.mkdir(parents=True, exist_ok=True)
    return audit_root


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def robust_mean_sd(s: pd.Series, default_mean: float, default_sd: float) -> Tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 20:
        return default_mean, default_sd
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if not np.isfinite(mean):
        mean = default_mean
    if not np.isfinite(sd) or sd <= 1e-8:
        sd = default_sd
    return mean, sd


def clip_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-8:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def rescale_lp(lp: np.ndarray, target_sd: float) -> np.ndarray:
    lp = np.asarray(lp, dtype=float)
    lp = lp - np.nanmean(lp)
    sd = np.nanstd(lp)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(lp)
    return lp / sd * target_sd


def logistic(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_logistic_intercept(target_rate: float, risk_score: np.ndarray) -> float:
    """Find intercept a such that mean(sigmoid(a + risk_score)) ~= target_rate."""
    target_rate = float(np.clip(target_rate, 0.0001, 0.9999))
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        rate = logistic(mid + risk_score).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mar_missing_mask(rng: np.random.Generator, target_rate: float, risk_score: np.ndarray) -> np.ndarray:
    """MAR-lite missingness with exact-ish target missingness by intercept calibration."""
    target_rate = float(np.clip(target_rate, 0.0, 0.95))
    n = len(risk_score)
    if target_rate <= 1e-8:
        return np.zeros(n, dtype=bool)
    if target_rate >= 0.95:
        return rng.random(n) < target_rate
    rs = zscore(risk_score)
    intercept = calibrate_logistic_intercept(target_rate, rs)
    p = logistic(intercept + rs)
    return rng.random(n) < p


def draw_exponential_time(base_rate: float, lp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rate = np.maximum(float(base_rate) * np.exp(lp), 1e-12)
    e = rng.exponential(scale=1.0, size=len(lp))
    return e / rate


def time_from_exponential_unit(e: np.ndarray, base_rate: float, lp: np.ndarray) -> np.ndarray:
    rate = np.maximum(float(base_rate) * np.exp(lp), 1e-12)
    return e / rate


def draw_piecewise_time_from_unit_exp(
    e: np.ndarray,
    base_rate: float,
    lp_early: np.ndarray,
    lp_late: np.ndarray,
    cut_year: float,
) -> np.ndarray:
    """Piecewise exponential time with one cut point using unit exponential draws."""
    r1 = np.maximum(float(base_rate) * np.exp(lp_early), 1e-12)
    r2 = np.maximum(float(base_rate) * np.exp(lp_late), 1e-12)
    h1 = r1 * cut_year
    t = np.empty(len(e), dtype=float)
    early = e <= h1
    t[early] = e[early] / r1[early]
    t[~early] = cut_year + (e[~early] - h1[~early]) / r2[~early]
    return t


def calibrate_base_rate_for_event(
    target_rate: float,
    lp: np.ndarray,
    death_before_time: np.ndarray,
    horizon: float,
    e_inst: np.ndarray,
    non_ph: bool = False,
    lp_late: Optional[np.ndarray] = None,
    cut_year: float = 2.0,
) -> Tuple[float, np.ndarray, float]:
    """Calibrate institutionalisation base rate so observed status=1 matches target."""
    target_rate = float(np.clip(target_rate, 0.001, 0.95))
    lo, hi = 1e-6, 5.0

    def make_time(lam: float) -> np.ndarray:
        if non_ph:
            assert lp_late is not None
            return draw_piecewise_time_from_unit_exp(e_inst, lam, lp, lp_late, cut_year)
        return time_from_exponential_unit(e_inst, lam, lp)

    # Expand hi if needed.
    for _ in range(30):
        t_hi = make_time(hi)
        obs_hi = ((t_hi <= horizon) & (t_hi < death_before_time)).mean()
        if obs_hi >= target_rate:
            break
        hi *= 2.0

    for _ in range(80):
        mid = math.sqrt(lo * hi)
        t_mid = make_time(mid)
        obs_mid = ((t_mid <= horizon) & (t_mid < death_before_time)).mean()
        if obs_mid < target_rate:
            lo = mid
        else:
            hi = mid
    lam = math.sqrt(lo * hi)
    t_final = make_time(lam)
    obs_final = ((t_final <= horizon) & (t_final < death_before_time)).mean()
    return lam, t_final, float(obs_final)


def calibrate_base_rate_for_latent_event(
    target_rate: float,
    lp: np.ndarray,
    horizon: float,
    e_time: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Calibrate base rate so P(T <= horizon) matches target before competing events."""
    target_rate = float(np.clip(target_rate, 0.001, 0.95))
    lo, hi = 1e-6, 5.0
    for _ in range(30):
        t_hi = time_from_exponential_unit(e_time, hi, lp)
        if (t_hi <= horizon).mean() >= target_rate:
            break
        hi *= 2.0
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        t_mid = time_from_exponential_unit(e_time, mid, lp)
        if (t_mid <= horizon).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    lam = math.sqrt(lo * hi)
    t_final = time_from_exponential_unit(e_time, lam, lp)
    rate = float((t_final <= horizon).mean())
    return lam, t_final, rate


def compress_float_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float32")
        elif pd.api.types.is_integer_dtype(out[c]):
            # Keep status/replicate ints compact.
            if out[c].min(skipna=True) >= 0 and out[c].max(skipna=True) < 2**31:
                out[c] = out[c].astype("int32")
    return out


# In[8]:


# =============================================================================
# INTERNAL SOURCE SUMMARY


# In[9]:


# =============================================================================

def load_internal_source(cfg: StepCConfig) -> pd.DataFrame:
    if not str(cfg.source_csv).strip():
        raise FileNotFoundError(
            "No internal summary source was configured. Set "
            "STEPC_INTERNAL_SOURCE_CSV inside the authorised environment."
        )
    path = Path(cfg.source_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Internal source CSV not found: {path}. Run Step B/v0 first or update CFG.source_csv."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"[source] Loaded internal source: {path} | rows={len(df)} | cols={df.shape[1]}")
    return df


def infer_source_stats(
    df: pd.DataFrame,
    cfg: StepCConfig,
    internal_audit_dir: Path,
) -> Dict:
    """Infer broad summaries from internal SLAM source without exporting patient rows."""
    stats: Dict = {}

    # Age.
    age_col = "age" if "age" in df.columns else ("age_at_scan_date" if "age_at_scan_date" in df.columns else None)
    if age_col:
        age_s = to_numeric_series(df, age_col)
    else:
        age_s = pd.Series(np.nan, index=df.index)
    age_mean, age_sd = robust_mean_sd(age_s, 76.0, 8.0)
    stats["age_mean"] = age_mean
    stats["age_sd"] = max(age_sd, 4.0)

    # Sex.
    sex_col = "sex_Female" if "sex_Female" in df.columns else ("Gender_ID" if "Gender_ID" in df.columns else None)
    if sex_col:
        if sex_col == "sex_Female":
            sex_vals = pd.to_numeric(df[sex_col], errors="coerce")
            p_female = float(sex_vals.mean()) if sex_vals.notna().sum() > 20 else 0.55
        else:
            sx = df[sex_col].astype(str).str.lower()
            female = sx.isin(["female", "f", "1", "1.0"])
            male = sx.isin(["male", "m", "0", "0.0"])
            denom = int((female | male).sum())
            p_female = float(female.sum() / denom) if denom > 20 else 0.55
    else:
        p_female = 0.55
    stats["p_female"] = float(np.clip(p_female, 0.25, 0.75))

    # MMSE.
    mmse_col = "MMSE" if "MMSE" in df.columns else ("Pre_Mini_Mental_Total" if "Pre_Mini_Mental_Total" in df.columns else None)
    mmse_s = to_numeric_series(df, mmse_col) if mmse_col else pd.Series(np.nan, index=df.index)
    mmse_mean, mmse_sd = robust_mean_sd(mmse_s, 22.0, 5.0)
    stats["mmse_mean"] = float(np.clip(mmse_mean, 10, 28))
    stats["mmse_sd"] = float(np.clip(mmse_sd, 2.5, 7.0))

    # Deprivation.
    imd_col = None
    for c in ["IMD_Score_2019_closest_to_scan", "IMD_Score_2019_most_recent", "deprivation_score"]:
        if c in df.columns:
            imd_col = c
            break
    imd_s = to_numeric_series(df, imd_col) if imd_col else pd.Series(np.nan, index=df.index)
    imd_mean, imd_sd = robust_mean_sd(imd_s, 25.0, 12.0)
    stats["imd_mean"] = float(np.clip(imd_mean, 5, 60))
    stats["imd_sd"] = float(np.clip(imd_sd, 5, 25))

    # MRI composite means/sds.
    mri_comp_defaults = {
        "MTL_total_pct": (1.7, 0.35),
        "Temporal_lateral_total_pct": (7.0, 1.2),
        "Medial_parietal_total_pct": (2.4, 0.5),
        "Posterior_total_pct": (5.5, 1.0),
        "Ventricles_total_pct": (3.0, 1.2),
        "LI_PCgG": (0.0, 0.10),
        "LI_PCu": (0.0, 0.10),
        "LI_MTG": (0.0, 0.10),
        "LI_ITG": (0.0, 0.10),
        "LI_STG": (0.0, 0.10),
    }
    stats["mri_composite_stats"] = {}
    for c, (dm, ds) in mri_comp_defaults.items():
        mean, sd = robust_mean_sd(to_numeric_series(df, c), dm, ds)
        stats["mri_composite_stats"][c] = {
            "mean": float(mean if np.isfinite(mean) else dm),
            "sd": float(max(sd if np.isfinite(sd) else ds, 0.05)),
        }

    # Comorbidities.
    comorb_cols = [
        "falls", "hypertension", "Cerebrovascular_accident", "Epilepsy",
        "Diabetes_mellitus", "Chronic_kidney_disease", "Parkinsons_disease",
        "Transient_ischemic_attack", "Chronic_obstructive_lung_disease", "Arthritis",
        "Heart_failure", "Ischemic_heart_disease", "Atrial_fibrillation",
        "Chronic_liver_disease", "Coronary_arteriosclerosis",
    ]
    stats["comorbidity_prevalence"] = {}
    for c in comorb_cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1)
            p = float(x.mean())
        else:
            # defaults approximate older adult clinical population.
            default_prev = {
                "falls": 0.18, "hypertension": 0.35, "Cerebrovascular_accident": 0.10,
                "Epilepsy": 0.04, "Diabetes_mellitus": 0.18, "Chronic_kidney_disease": 0.10,
                "Parkinsons_disease": 0.06, "Transient_ischemic_attack": 0.08,
                "Chronic_obstructive_lung_disease": 0.10, "Arthritis": 0.20,
                "Heart_failure": 0.07, "Ischemic_heart_disease": 0.15,
                "Atrial_fibrillation": 0.12, "Chronic_liver_disease": 0.03,
                "Coronary_arteriosclerosis": 0.12,
            }.get(c, 0.10)
            p = default_prev
        stats["comorbidity_prevalence"][c] = float(np.clip(p, 0.005, 0.80))

    # NLP raw count columns.
    nlp_raw_cols = [
        c for c in ["pathology_ischaemic", "pathology_neurodegenerative_dementia"] if c in df.columns
    ]
    if not nlp_raw_cols:
        # fallback to engineered names, but raw synthetic columns will still be generated.
        nlp_raw_cols = ["pathology_ischaemic", "pathology_neurodegenerative_dementia"]
    stats["nlp_raw_cols"] = nlp_raw_cols
    stats["nlp_count_means_nonmissing"] = {}
    for c in nlp_raw_cols:
        x = to_numeric_series(df, c) if c in df.columns else pd.Series(np.nan, index=df.index)
        xm = x.dropna()
        stats["nlp_count_means_nonmissing"][c] = float(np.clip(xm.mean() if len(xm) else 1.5, 0.2, 8.0))

    # Death distribution from observed death dates if available.
    death_rate = np.nan
    death_time_median = np.nan
    if {"scan_date_sim", "death_date_sim"}.issubset(df.columns):
        scan = pd.to_datetime(df["scan_date_sim"], errors="coerce")
        death = pd.to_datetime(df["death_date_sim"], errors="coerce")
        dt = (death - scan).dt.days / 365.25
        valid_dt = dt[(dt > 0) & np.isfinite(dt)]
        if len(valid_dt) >= 20:
            death_rate = float(((dt > 0) & (dt <= cfg.horizon_years)).mean())
            death_time_median = float(valid_dt.median())
    elif {"Scan_Date", "Date_Of_Death"}.issubset(df.columns):
        scan = pd.to_datetime(df["Scan_Date"], errors="coerce")
        death = pd.to_datetime(df["Date_Of_Death"], errors="coerce")
        dt = (death - scan).dt.days / 365.25
        valid_dt = dt[(dt > 0) & np.isfinite(dt)]
        if len(valid_dt) >= 20:
            death_rate = float(((dt > 0) & (dt <= cfg.horizon_years)).mean())
            death_time_median = float(valid_dt.median())
    if not np.isfinite(death_rate):
        death_rate = cfg.fallback_death_5y_rate
    stats["source_latent_death_5y_rate"] = float(np.clip(death_rate, 0.05, 0.45))
    stats["source_positive_death_time_median_years"] = float(death_time_median) if np.isfinite(death_time_median) else np.nan

    # Missingness by block, SLAM-informed with caps.
    def block_missing_rate(cols: List[str], fallback: float, cap: float) -> float:
        present = [c for c in cols if c in df.columns]
        if not present:
            return float(min(fallback, cap))
        rates = df[present].isna().mean(axis=0).astype(float)
        rate = float(rates.mean())
        if not np.isfinite(rate):
            rate = fallback
        return float(np.clip(min(rate, cap), 0.0, cap))

    mri_cols = [c for c in df.columns if (str(c).endswith("_pct") or "__pct" in str(c) or str(c).startswith("MRI_"))]
    nlp_raw_source_cols = [c for c in ["pathology_ischaemic", "pathology_neurodegenerative_dementia"] if c in df.columns]
    deprivation_cols = [c for c in ["IMD_Score_2019_closest_to_scan", "IMD_Score_2019_most_recent", "deprivation_score"] if c in df.columns]

    stats["missing_rates"] = {
        "age": block_missing_rate([age_col] if age_col else [], 0.0, cfg.cap_missing_age),
        "sex": block_missing_rate([sex_col] if sex_col else [], 0.0, cfg.cap_missing_sex),
        "MMSE": block_missing_rate([mmse_col] if mmse_col else [], 0.05, cfg.cap_missing_mmse),
        "MRI_composite": block_missing_rate(list(stats["mri_composite_stats"].keys()), 0.03, cfg.cap_missing_mri_composite),
        "MRI_regional": block_missing_rate(mri_cols, 0.05, cfg.cap_missing_mri_regional),
        "comorbidity": block_missing_rate(comorb_cols, 0.0, cfg.cap_missing_comorbidity),
        "deprivation": block_missing_rate(deprivation_cols, 0.05, cfg.cap_missing_deprivation),
        "NLP_raw": block_missing_rate(nlp_raw_source_cols, 0.30, cfg.cap_missing_nlp_raw),
        "WMH": block_missing_rate(["WMH_77", "wmh_icv_pct", "wmh_log_icv"], 0.10, cfg.cap_missing_wmh),
    }

    # Save summary used. This is aggregate summary only, no patient rows.
    summary_flat = []
    for k, v in stats.items():
        if isinstance(v, (int, float, str)) or v is None:
            summary_flat.append({"item": k, "value": v})
    pd.DataFrame(summary_flat).to_csv(
        internal_audit_dir / "source_scalar_summary_used.csv",
        index=False,
    )
    pd.DataFrame([{"block": k, "missing_rate_used": v} for k, v in stats["missing_rates"].items()]).to_csv(
        internal_audit_dir / "source_block_missingness_used_capped.csv",
        index=False,
    )
    with open(
        internal_audit_dir / "source_summary_used.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stats, f, indent=2)

    print("[source summary] Key rates:")
    print("  source latent 5y death rate:", round(stats["source_latent_death_5y_rate"], 3))
    print("  missing rates used:", {k: round(v, 3) for k, v in stats["missing_rates"].items()})
    return stats


# In[10]:


# =============================================================================
# SYNTHETIC FEATURE GENERATION


# In[11]:


# =============================================================================

REGIONAL_MRI_NAMES = [
    # Interpretable synthetic ROI-like names. Values are synthetic, not real patient values.
    "Right_Hippocampus_pct", "Left_Hippocampus_pct",
    "Right_Entorhinal_pct", "Left_Entorhinal_pct",
    "Right_Parahippocampal_pct", "Left_Parahippocampal_pct",
    "Right_MTG_pct", "Left_MTG_pct",
    "Right_ITG_pct", "Left_ITG_pct",
    "Right_STG_pct", "Left_STG_pct",
    "Right_Fusiform_pct", "Left_Fusiform_pct",
    "Right_TemporalPole_pct", "Left_TemporalPole_pct",
    "Right_PosteriorCingulate_pct", "Left_PosteriorCingulate_pct",
    "Right_Precuneus_pct", "Left_Precuneus_pct",
    "Right_SPL_pct", "Left_SPL_pct",
    "Right_SOG_pct", "Left_SOG_pct",
    "Right_MOG_pct", "Left_MOG_pct",
    "Right_IOG_pct", "Left_IOG_pct",
    "Right_LateralVentricle_pct", "Left_LateralVentricle_pct",
    "Right_InfLatVentricle_pct", "Left_InfLatVentricle_pct",
]

# Add extra interpretable-but-generic cortical region names for high-dimensional setting.
for hemi in ["Right", "Left"]:
    for lobe in ["Frontal", "Parietal", "Temporal", "Occipital", "Insula", "Cingulate", "DeepNuclei"]:
        for idx in range(1, 7):
            REGIONAL_MRI_NAMES.append(f"{hemi}_{lobe}_Region{idx:02d}_pct")

SPARSE_MRI_SIGNAL_COLS = [
    "Right_Hippocampus_pct", "Left_Hippocampus_pct",
    "Right_Entorhinal_pct", "Left_Entorhinal_pct",
    "Right_LateralVentricle_pct", "Left_LateralVentricle_pct",
    "Right_Precuneus_pct", "Left_Precuneus_pct",
    "Right_Temporal_Region01_pct", "Left_Temporal_Region01_pct",
]


def generate_base_features(n: int, rep: int, scenario_id: str, rng: np.random.Generator, stats: Dict) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Generate complete synthetic features before applying missingness."""
    # Latent factors to induce realistic correlation without copying real rows.
    neuro = rng.normal(0, 1, n)
    vascular = rng.normal(0, 1, n)
    frailty = rng.normal(0, 1, n)
    reserve = rng.normal(0, 1, n)

    # Demographics.
    age = rng.normal(stats["age_mean"], stats["age_sd"], n)
    age = clip_array(age, 50, 98)
    sex_female = rng.binomial(1, stats["p_female"], n).astype(int)
    sex = np.where(sex_female == 1, "Female", "Male")

    # Deprivation.
    imd = rng.normal(stats["imd_mean"], stats["imd_sd"], n)
    imd = clip_array(imd, 1, 80)
    imd_decile = clip_array(np.ceil((imd.rank() if isinstance(imd, pd.Series) else pd.Series(imd)).rank(pct=True).to_numpy() * 10), 1, 10).astype(int)

    # Cognitive score.
    mmse = (
        stats["mmse_mean"]
        - 2.1 * neuro
        - 1.0 * frailty
        - 0.045 * (age - stats["age_mean"])
        + 0.8 * reserve
        + rng.normal(0, 2.0, n)
    )
    mmse = clip_array(mmse, 0, 30)

    # Synthetic diagnosis reference group, not intended as model predictor.
    severity_score = 0.9 * neuro + 0.5 * vascular + 0.5 * zscore(30 - mmse) + rng.normal(0, 0.6, n)
    p_mixed = logistic(-0.8 + 0.8 * vascular + 0.35 * neuro)
    p_ad = logistic(0.05 + 0.8 * neuro - 0.25 * vascular)
    u = rng.random(n)
    diagnosis = np.where(u < p_ad * 0.55, "AD", np.where(u < p_ad * 0.55 + p_mixed * 0.35, "Mixed", "Non-AD"))

    # MRI composites. Lower volume with higher neuro/frailty; ventricles increase.
    mri = {}
    for c, st in stats["mri_composite_stats"].items():
        mean, sd = st["mean"], st["sd"]
        if c == "Ventricles_total_pct":
            val = mean + 0.45 * sd * neuro + 0.35 * sd * frailty + 0.20 * sd * zscore(age) + rng.normal(0, 0.65 * sd, n)
            val = clip_array(val, 0.1, mean + 5 * sd)
        elif c.startswith("LI_"):
            val = mean + rng.normal(0, sd, n) + 0.03 * rng.normal(size=n)
            val = clip_array(val, -0.8, 0.8)
        else:
            val = mean - 0.45 * sd * neuro - 0.20 * sd * frailty - 0.15 * sd * zscore(age) + rng.normal(0, 0.65 * sd, n)
            val = clip_array(val, 0.05, mean + 4 * sd)
        mri[c] = val

    # Regional MRI features; correlated with composites and latent factors.
    regional = {}
    base_temporal = zscore(mri["Temporal_lateral_total_pct"])
    base_mtl = zscore(mri["MTL_total_pct"])
    base_post = zscore(mri["Posterior_total_pct"])
    base_vent = zscore(mri["Ventricles_total_pct"])
    for name in REGIONAL_MRI_NAMES:
        if "Hippocampus" in name or "Entorhinal" in name or "Parahippocampal" in name:
            center = 0.25 + 0.05 * base_mtl
            val = center - 0.04 * neuro + rng.normal(0, 0.025, n)
        elif "Ventricle" in name:
            center = 0.55 + 0.20 * base_vent
            val = center + 0.08 * neuro + 0.05 * frailty + rng.normal(0, 0.08, n)
        elif "Temporal" in name or "MTG" in name or "ITG" in name or "STG" in name or "Fusiform" in name or "TemporalPole" in name:
            center = 0.65 + 0.08 * base_temporal
            val = center - 0.04 * neuro + rng.normal(0, 0.05, n)
        elif "Precuneus" in name or "PosteriorCingulate" in name or "SPL" in name or "Occipital" in name or "SOG" in name or "MOG" in name or "IOG" in name:
            center = 0.55 + 0.08 * base_post
            val = center - 0.03 * neuro + rng.normal(0, 0.05, n)
        else:
            center = 0.60 - 0.02 * neuro + 0.01 * vascular
            val = center + rng.normal(0, 0.06, n)
        regional[name] = clip_array(val, 0.01, 2.5)

    # WMH synthetic features.
    wmh_latent = 0.9 * vascular + 0.45 * zscore(age) + 0.25 * frailty + rng.normal(0, 0.6, n)
    wmh_icv_pct = np.exp(-4.0 + 0.55 * wmh_latent + rng.normal(0, 0.45, n)) * 100
    wmh_icv_pct = clip_array(wmh_icv_pct, 0, 5.0)
    wmh_log_icv = np.log1p(wmh_icv_pct / 100.0)

    # Comorbidities.
    comorb_data = {}
    for c, base_p in stats["comorbidity_prevalence"].items():
        comorb_risk = 0.45 * zscore(age) + 0.45 * vascular + 0.25 * frailty + 0.10 * zscore(imd)
        # Different comorbidities emphasize different latent risks.
        if any(k in c.lower() for k in ["cerebro", "transient", "atrial", "heart", "ischemic", "coronary", "hypertension"]):
            comorb_risk += 0.35 * vascular
        if c.lower() in ["falls", "parkinsons_disease", "epilepsy"]:
            comorb_risk += 0.25 * frailty + 0.15 * neuro
        intercept = calibrate_logistic_intercept(base_p, 0.65 * zscore(comorb_risk))
        p = logistic(intercept + 0.65 * zscore(comorb_risk))
        comorb_data[c] = rng.binomial(1, p, n).astype(int)
    comorb_matrix = np.column_stack([comorb_data[c] for c in stats["comorbidity_prevalence"].keys()])
    comorb_burden = comorb_matrix.sum(axis=1)

    # NLP counts, present/missing/log engineered later after missingness.
    nlp_ischaemic_mean = stats["nlp_count_means_nonmissing"].get("pathology_ischaemic", 1.4)
    nlp_neuro_mean = stats["nlp_count_means_nonmissing"].get("pathology_neurodegenerative_dementia", 1.6)
    lam_isch = np.clip(nlp_ischaemic_mean * np.exp(0.45 * vascular + 0.15 * frailty - 0.2), 0.05, 20)
    lam_neuro = np.clip(nlp_neuro_mean * np.exp(0.45 * neuro + 0.15 * zscore(30 - mmse) - 0.2), 0.05, 20)
    pathology_ischaemic = rng.poisson(lam_isch).astype(float)
    pathology_neurodegenerative = rng.poisson(lam_neuro).astype(float)
    # Force some zeros even when notes are present.
    pathology_ischaemic[rng.random(n) < 0.25] = 0
    pathology_neurodegenerative[rng.random(n) < 0.20] = 0

    df = pd.DataFrame({
        "synthetic_id": [f"SYN_{scenario_id}_R{rep:03d}_{i:05d}" for i in range(n)],
        "scenario_id": scenario_id,
        "replicate_id": rep,
        "age": age,
        "sex": sex,
        "sex_Female": sex_female,
        "MMSE": mmse,
        "IMD_Score_2019_synthetic": imd,
        "IMD_Decile_2019_synthetic": imd_decile,
        "diagnosis_reference_NOT_PREDICTOR": diagnosis,
        "wmh_icv_pct": wmh_icv_pct,
        "wmh_log_icv": wmh_log_icv,
    })

    for c, v in mri.items():
        df[c] = v
    for c, v in regional.items():
        df[c] = v
    for c, v in comorb_data.items():
        df[c] = v

    df["pathology_ischaemic"] = pathology_ischaemic
    df["pathology_neurodegenerative_dementia"] = pathology_neurodegenerative

    latent = {
        "neuro": neuro,
        "vascular": vascular,
        "frailty": frailty,
        "reserve": reserve,
        "comorbidity_burden": comorb_burden,
        "severity_score": severity_score,
    }
    return df, latent


def apply_missingness(df: pd.DataFrame, latent: Dict[str, np.ndarray], rng: np.random.Generator, stats: Dict, scenario: Dict) -> pd.DataFrame:
    """Apply MAR-lite missingness after outcome generation."""
    out = df.copy()
    mr = stats["missing_rates"].copy()
    mult = float(scenario.get("missingness_multiplier", 1.0))

    # Apply multiplier and caps again.
    caps = {
        "age": CFG.cap_missing_age,
        "sex": CFG.cap_missing_sex,
        "MMSE": CFG.cap_missing_mmse,
        "MRI_composite": CFG.cap_missing_mri_composite,
        "MRI_regional": CFG.cap_missing_mri_regional,
        "comorbidity": CFG.cap_missing_comorbidity,
        "deprivation": CFG.cap_missing_deprivation,
        "NLP_raw": CFG.cap_missing_nlp_raw,
        "WMH": CFG.cap_missing_wmh,
    }
    mr = {k: min(float(v) * mult, caps.get(k, float(v))) for k, v in mr.items()}

    # MAR score: older, lower MMSE, more frailty/comorbidity/deprivation -> more missingness.
    mar_score = (
        0.35 * zscore(out["age"].to_numpy())
        + 0.35 * zscore(30 - out["MMSE"].to_numpy())
        + 0.25 * zscore(latent["frailty"])
        + 0.20 * zscore(latent["comorbidity_burden"])
        + 0.15 * zscore(out["IMD_Score_2019_synthetic"].to_numpy())
    )

    # Age/sex usually complete.
    if mr["age"] > 0:
        out.loc[mar_missing_mask(rng, mr["age"], mar_score), "age"] = np.nan
    if mr["sex"] > 0:
        mask = mar_missing_mask(rng, mr["sex"], mar_score)
        out.loc[mask, "sex"] = pd.NA
        out.loc[mask, "sex_Female"] = np.nan

    # MMSE.
    out.loc[mar_missing_mask(rng, mr["MMSE"], mar_score), "MMSE"] = np.nan

    # Deprivation.
    dep_mask = mar_missing_mask(rng, mr["deprivation"], mar_score)
    out.loc[dep_mask, ["IMD_Score_2019_synthetic", "IMD_Decile_2019_synthetic"]] = np.nan

    # MRI composites and regional.
    composite_cols = list(stats["mri_composite_stats"].keys())
    region_cols = [c for c in REGIONAL_MRI_NAMES if c in out.columns]
    for c in composite_cols:
        out.loc[mar_missing_mask(rng, mr["MRI_composite"], mar_score), c] = np.nan
    for c in region_cols:
        out.loc[mar_missing_mask(rng, mr["MRI_regional"], mar_score), c] = np.nan

    # WMH.
    wmh_mask = mar_missing_mask(rng, mr["WMH"], mar_score + 0.2 * zscore(latent["vascular"]))
    out.loc[wmh_mask, ["wmh_icv_pct", "wmh_log_icv"]] = np.nan
    out["wmh_missing"] = wmh_mask.astype(int)

    # Comorbidities low missingness.
    for c in stats["comorbidity_prevalence"].keys():
        if mr["comorbidity"] > 0 and c in out.columns:
            out.loc[mar_missing_mask(rng, mr["comorbidity"], mar_score), c] = np.nan

    # NLP raw missingness: missing indicators/present/log generated after masking.
    nlp_score = mar_score + 0.15 * zscore(latent["vascular"]) + 0.15 * zscore(latent["neuro"])
    for c in ["pathology_ischaemic", "pathology_neurodegenerative_dementia"]:
        if c in out.columns:
            miss = mar_missing_mask(rng, mr["NLP_raw"], nlp_score)
            out.loc[miss, c] = np.nan
            out[f"{c}__missing"] = out[c].isna().astype(int)
            out[f"{c}__present"] = (pd.to_numeric(out[c], errors="coerce").fillna(0) > 0).astype(int)
            out[f"{c}__log1p"] = np.log1p(pd.to_numeric(out[c], errors="coerce").fillna(0))

    return out


# In[12]:


# =============================================================================
# DGM / OUTCOME GENERATION


# In[13]:


# =============================================================================

def build_lp_components(df_complete: pd.DataFrame, latent: Dict[str, np.ndarray], scenario: Dict) -> Dict[str, np.ndarray]:
    """Build care-home and death LPs from complete synthetic features."""
    age_z = zscore(df_complete["age"].to_numpy())
    mmse_z = zscore(df_complete["MMSE"].to_numpy())
    low_mmse_z = zscore(30 - df_complete["MMSE"].to_numpy())
    mtl_z = zscore(df_complete["MTL_total_pct"].to_numpy())
    temp_z = zscore(df_complete["Temporal_lateral_total_pct"].to_numpy())
    post_z = zscore(df_complete["Posterior_total_pct"].to_numpy())
    vent_z = zscore(df_complete["Ventricles_total_pct"].to_numpy())
    imd_z = zscore(df_complete["IMD_Score_2019_synthetic"].to_numpy())
    nlp_vasc_z = zscore(np.log1p(df_complete["pathology_ischaemic"].to_numpy()))
    nlp_neuro_z = zscore(np.log1p(df_complete["pathology_neurodegenerative_dementia"].to_numpy()))
    comorb_z = zscore(latent["comorbidity_burden"])

    # Baseline linear care-home LP.
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

    if scenario.get("use_nonlinear", False):
        mmse = df_complete["MMSE"].to_numpy()
        ventricles = df_complete["Ventricles_total_pct"].to_numpy()
        lp += 0.45 * (mmse < 18).astype(float)
        lp += 0.25 * (mmse < 23).astype(float)
        lp += 0.35 * (zscore(ventricles) > 1.0).astype(float)
        lp += 0.25 * age_z * low_mmse_z
        lp += 0.20 * zscore(latent["frailty"]) * zscore(latent["vascular"])

    if scenario.get("use_sparse_mri_signal", False):
        sparse_terms = np.zeros(len(df_complete), dtype=float)
        used_cols = []
        for c in SPARSE_MRI_SIGNAL_COLS:
            if c in df_complete.columns:
                sign = 1.0 if "Ventricle" in c else -1.0
                sparse_terms += sign * 0.18 * zscore(df_complete[c].to_numpy())
                used_cols.append(c)
        lp = 0.55 * lp + sparse_terms

    lp_inst = rescale_lp(lp, float(scenario.get("lp_sd", 0.8)))

    # Non-PH: early dominated by cognition/neuro; late dominated by age/vascular/frailty.
    lp_early = rescale_lp(
        0.25 * age_z + 0.75 * low_mmse_z - 0.35 * mtl_z + 0.25 * vent_z + 0.20 * nlp_neuro_z,
        float(scenario.get("lp_sd", 0.9)),
    )
    lp_late = rescale_lp(
        0.55 * age_z + 0.25 * low_mmse_z + 0.50 * nlp_vasc_z + 0.40 * comorb_z + 0.30 * zscore(latent["frailty"]),
        float(scenario.get("lp_sd", 0.9)),
    )

    # Community death LP.
    lp_death = rescale_lp(
        0.75 * age_z
        + 0.30 * low_mmse_z
        + 0.60 * comorb_z
        + 0.30 * nlp_vasc_z
        + 0.25 * zscore(latent["frailty"])
        + 0.12 * (1 - df_complete["sex_Female"].to_numpy()),
        0.9,
    )

    return {
        "lp_inst": lp_inst,
        "lp_inst_early": lp_early,
        "lp_inst_late": lp_late,
        "lp_death_before": lp_death,
        "age_z": age_z,
        "low_mmse_z": low_mmse_z,
        "comorb_z": comorb_z,
        "nlp_vascular_z": nlp_vasc_z,
    }


def generate_outcomes(
    df_complete: pd.DataFrame,
    latent: Dict[str, np.ndarray],
    scenario: Dict,
    rng: np.random.Generator,
    stats: Dict,
    cfg: StepCConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate multi-state outcome variables."""
    n = len(df_complete)
    horizon = float(cfg.horizon_years)
    comps = build_lp_components(df_complete, latent, scenario)

    # Latent death target based on source death distribution, with S7 strong death override.
    if scenario["scenario_id"].startswith("S7"):
        latent_death_target = float(cfg.s7_latent_death_5y_rate)
    else:
        latent_death_target = float(stats["source_latent_death_5y_rate"])

    # Fixed random variates for monotonic calibration.
    e_death = rng.exponential(scale=1.0, size=n)
    e_inst = rng.exponential(scale=1.0, size=n)
    e_postdeath = rng.exponential(scale=1.0, size=n)

    lambda_death, t_death_before, latent_death_rate = calibrate_base_rate_for_latent_event(
        latent_death_target,
        comps["lp_death_before"],
        horizon,
        e_death,
    )

    # Calibrate care-home base rate to observed care-home entry rate after death before care-home.
    if scenario.get("use_non_ph", False):
        lambda_inst, t_carehome, observed_inst_rate = calibrate_base_rate_for_event(
            scenario["target_observed_carehome_rate"],
            comps["lp_inst_early"],
            t_death_before,
            horizon,
            e_inst,
            non_ph=True,
            lp_late=comps["lp_inst_late"],
            cut_year=2.0,
        )
    else:
        lambda_inst, t_carehome, observed_inst_rate = calibrate_base_rate_for_event(
            scenario["target_observed_carehome_rate"],
            comps["lp_inst"],
            t_death_before,
            horizon,
            e_inst,
            non_ph=False,
        )

    # Primary status.
    carehome_first = (t_carehome <= horizon) & (t_carehome < t_death_before)
    death_before_first = (t_death_before <= horizon) & (t_death_before < t_carehome)
    status = np.zeros(n, dtype=int)
    status[carehome_first] = 1
    status[death_before_first] = 2
    duration = np.full(n, horizon, dtype=float)
    duration[carehome_first] = t_carehome[carehome_first]
    duration[death_before_first] = t_death_before[death_before_first]

    # Post-carehome death among those who entered care home before death.
    # Q5A: post-carehome hazard is 1.5x community death hazard.
    death_rate_subject = np.maximum(lambda_death * np.exp(comps["lp_death_before"]), 1e-12)
    post_rate = cfg.post_carehome_death_hr_multiplier * death_rate_subject
    t_wait_post = e_postdeath / post_rate
    t_death_after = t_carehome + t_wait_post
    death_after_carehome = carehome_first & (t_death_after <= horizon)
    time_carehome_to_death = np.where(death_after_carehome, t_wait_post, np.nan)

    # Overall death by 5 years in synthetic data: death before + death after carehome.
    any_death_5y = death_before_first | death_after_carehome

    # True 5-year care-home risk before competing death (latent) and observable with death.
    if scenario.get("use_non_ph", False):
        # Approximate individual latent 5y care-home risk under piecewise hazard.
        h1 = lambda_inst * np.exp(comps["lp_inst_early"]) * min(2.0, horizon)
        h2 = lambda_inst * np.exp(comps["lp_inst_late"]) * max(0.0, horizon - 2.0)
        true_risk_latent = 1 - np.exp(-(h1 + h2))
    else:
        true_risk_latent = 1 - np.exp(-lambda_inst * np.exp(comps["lp_inst"]) * horizon)

    # Observable care-home probability approximation under independent exponential hazards.
    # For non-PH this is approximate; observed event in generated data is exact.
    death_hazard = lambda_death * np.exp(comps["lp_death_before"])
    inst_hazard = lambda_inst * np.exp(comps["lp_inst"])
    total_h = inst_hazard + death_hazard
    true_risk_observable_approx = np.where(
        total_h > 0,
        inst_hazard / total_h * (1 - np.exp(-total_h * horizon)),
        0.0,
    )

    out = df_complete.copy()
    out["t_carehome_latent_years"] = t_carehome
    out["t_death_before_carehome_latent_years"] = t_death_before
    out["t_death_after_carehome_years"] = np.where(death_after_carehome, t_death_after, np.nan)
    out["time_carehome_to_death_years"] = time_carehome_to_death
    out["duration_years"] = duration
    out["status"] = status
    out["status_label"] = np.where(status == 1, "carehome", np.where(status == 2, "death_before_carehome", "event_free_censored"))
    out["event_carehome"] = (status == 1).astype(int)
    out["event_death_before_carehome"] = (status == 2).astype(int)
    out["event_free_or_censored"] = (status == 0).astype(int)
    out["death_after_carehome"] = death_after_carehome.astype(int)
    out["any_death_5y"] = any_death_5y.astype(int)
    out["multi_state_path"] = np.where(
        status == 2,
        "0_to_2_death_before_carehome",
        np.where(
            status == 1,
            np.where(death_after_carehome, "0_to_1_carehome_then_3_death", "0_to_1_carehome_alive_or_censored"),
            "0_event_free_censored",
        ),
    )

    out["true_lp_carehome"] = comps["lp_inst"]
    out["true_lp_death_before_carehome"] = comps["lp_death_before"]
    out["true_risk_carehome_5y_latent"] = true_risk_latent
    out["true_risk_carehome_5y_observable_approx"] = true_risk_observable_approx
    out["lambda0_carehome"] = lambda_inst
    out["lambda0_death_before_carehome"] = lambda_death
    out["target_observed_carehome_rate"] = scenario["target_observed_carehome_rate"]
    out["target_latent_death_5y_rate"] = latent_death_target
    out["post_carehome_death_hr_multiplier"] = cfg.post_carehome_death_hr_multiplier

    summary = {
        "scenario_id": scenario["scenario_id"],
        "n": n,
        "target_observed_carehome_rate": scenario["target_observed_carehome_rate"],
        "observed_carehome_rate": float((status == 1).mean()),
        "observed_death_before_carehome_rate": float((status == 2).mean()),
        "observed_event_free_or_censored_rate": float((status == 0).mean()),
        "death_after_carehome_rate": float(death_after_carehome.mean()),
        "any_death_5y_rate": float(any_death_5y.mean()),
        "latent_death_5y_target": latent_death_target,
        "latent_death_5y_rate_before_carehome_process": latent_death_rate,
        "median_duration_years": float(np.median(duration)),
        "median_t_carehome_latent_years": float(np.median(t_carehome)),
        "median_t_death_before_carehome_latent_years": float(np.median(t_death_before)),
        "lambda0_carehome": float(lambda_inst),
        "lambda0_death_before_carehome": float(lambda_death),
        "post_carehome_death_hr_multiplier": cfg.post_carehome_death_hr_multiplier,
    }
    return out, summary


# In[14]:


# =============================================================================
# SUMMARY / AUDIT


# In[15]:


# =============================================================================

def summarise_repetition(df: pd.DataFrame, scenario: Dict, rep: int) -> Dict:
    return {
        "scenario_id": scenario["scenario_id"],
        "replicate_id": rep,
        "n": len(df),
        "observed_carehome_rate": float(df["event_carehome"].mean()),
        "observed_death_before_carehome_rate": float(df["event_death_before_carehome"].mean()),
        "observed_event_free_or_censored_rate": float(df["event_free_or_censored"].mean()),
        "death_after_carehome_rate": float(df["death_after_carehome"].mean()),
        "any_death_5y_rate": float(df["any_death_5y"].mean()),
        "median_duration_years": float(df["duration_years"].median()),
        "mean_true_risk_carehome_5y_observable_approx": float(df["true_risk_carehome_5y_observable_approx"].mean()),
        "missing_rate_MMSE": float(df["MMSE"].isna().mean()) if "MMSE" in df.columns else np.nan,
        "missing_rate_MRI_region_mean": float(df[[c for c in REGIONAL_MRI_NAMES if c in df.columns]].isna().mean().mean()),
        "missing_rate_NLP_ischaemic_raw": float(df["pathology_ischaemic"].isna().mean()),
        "missing_rate_NLP_neuro_raw": float(df["pathology_neurodegenerative_dementia"].isna().mean()),
    }


def build_feature_dictionary(example_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in example_cols:
        block = "other"
        role = "predictor_or_output"
        if c in ["synthetic_id", "scenario_id", "replicate_id"]:
            block = "identifier_synthetic"
            role = "synthetic_identifier_or_design"
        elif c in ["age", "sex", "sex_Female"]:
            block = "demographic"
            role = "synthetic_predictor"
        elif c == "MMSE":
            block = "cognition"
            role = "synthetic_predictor"
        elif c.startswith("IMD_"):
            block = "deprivation"
            role = "synthetic_predictor"
        elif c in ["diagnosis_reference_NOT_PREDICTOR"]:
            block = "diagnosis_reference"
            role = "reference_only_not_for_prediction"
        elif c in REGIONAL_MRI_NAMES or c.endswith("_pct") or c.startswith("LI_") or c in ["MTL_total_pct", "Temporal_lateral_total_pct", "Medial_parietal_total_pct", "Posterior_total_pct", "Ventricles_total_pct"]:
            block = "MRI"
            role = "synthetic_predictor"
        elif c.startswith("pathology_"):
            block = "NLP"
            role = "synthetic_predictor_engineered" if "__" in c else "synthetic_raw_count"
        elif c.startswith("wmh"):
            block = "WMH"
            role = "synthetic_predictor"
        elif c in ["duration_years", "status", "status_label", "event_carehome", "event_death_before_carehome", "event_free_or_censored", "death_after_carehome", "any_death_5y", "multi_state_path"]:
            block = "outcome"
            role = "synthetic_outcome"
        elif c.startswith("t_") or c.startswith("true_") or c.startswith("lambda0") or c.startswith("target_") or c.startswith("post_carehome"):
            block = "DGM_truth"
            role = "simulation_truth_or_audit"
        rows.append({"column": c, "block": block, "role": role})
    return pd.DataFrame(rows)


def export_safety_audit(out_dir: Path) -> pd.DataFrame:
    forbidden_patterns = [
        r"\bBrcId\b", r"\bScanID\b", r"\bPK\b", r"Date_Of_Death", r"Scan_Date",
        r"cleaneddateofbirth", r"postcode", r"xml_file", r"orig_row_id",
    ]
    rows = []
    for p in sorted((out_dir / "scenario_datasets").glob("*.csv*")):
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        hits = []
        for c in cols:
            for pat in forbidden_patterns:
                if re.search(pat, str(c), flags=re.IGNORECASE):
                    hits.append(c)
                    break
        rows.append({
            "file": p.name,
            "n_columns": len(cols),
            "n_forbidden_column_name_hits": len(hits),
            "forbidden_column_name_hits": "; ".join(hits),
            "safe_to_export_column_names": len(hits) == 0,
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "audit" / "export_safety_audit.csv", index=False)
    return audit


def write_readme(out_dir: Path, cfg: StepCConfig) -> None:
    readme = f"""# Step C v1 fully synthetic exportable datasets

Generated by `stepC_fully_synthetic_exportable_generator_v1`.

## Design

- Fully synthetic predictors and outcomes.
- No real BrcId, ScanID, Scan_Date, Date_Of_Death, postcode, or real patient-level rows are exported.
- Generated inside SLAM using only broad internal summary information.
- n per repetition: {cfg.n_per_rep}
- repetitions per scenario: {cfg.n_reps}
- horizon: {cfg.horizon_years} years
- multi-state DGM: baseline -> care home; baseline -> death before care home; care home -> death after care home
- care-home target definition: observed status=1 rate after accounting for death before care home (Q3A)
- post-carehome death hazard multiplier: {cfg.post_carehome_death_hr_multiplier}

## Primary status coding

- `status = 0`: event-free / censored at horizon
- `status = 1`: care home entry before death and before horizon
- `status = 2`: death before care home before horizon

## File layout

- `scenario_datasets/*.csv.gz`: one compressed CSV per scenario, containing all replicate_id values.
- `tables/scenario_summary.csv`: scenario-level summaries.
- `tables/repetition_summary.csv`: per scenario and repetition summaries.
- `tables/dgm_definition_table.csv`: scenario definitions.
- `tables/feature_dictionary.csv`: column roles.
- `tables/missingness_summary.csv`: output missingness by scenario.
- `audit/export_safety_audit.csv`: check for forbidden real-data column names.

## Important limitation

No real care-home entry date was available. Death-time distribution is informed by observed SLAM death dates when available, but death before/after care-home ordering is generated under explicit synthetic multi-state assumptions.
"""
    (out_dir / "README_stepC_fully_synthetic.md").write_text(readme, encoding="utf-8")


# In[16]:


# =============================================================================
# MAIN


# In[17]:


# =============================================================================

def run_stepC(cfg: StepCConfig = CFG) -> None:
    out_dir = Path(cfg.out_dir)
    ensure_dirs(out_dir)
    internal_audit_dir = prepare_internal_audit_dir(
        out_dir,
        Path(cfg.internal_summary_dir),
    )

    print("=" * 88)
    print("Step C v1 fully synthetic exportable data generator")
    print("=" * 88)
    print(json.dumps(asdict(cfg), indent=2))

    source = load_internal_source(cfg)
    stats = infer_source_stats(source, cfg, internal_audit_dir)

    scenario_summary_rows = []
    rep_summary_rows = []
    missingness_rows = []
    example_cols = None

    for sc_idx, scenario in enumerate(SCENARIOS):
        sid = scenario["scenario_id"]
        print("\n" + "=" * 88)
        print(f"[scenario] {sid}")
        print("=" * 88)
        scenario_parts = []
        scenario_rep_summaries = []

        for rep in range(1, cfg.n_reps + 1):
            seed = cfg.random_seed + 100000 * sc_idx + rep
            rng = np.random.default_rng(seed)

            df_complete, latent = generate_base_features(cfg.n_per_rep, rep, sid, rng, stats)
            df_outcome, outcome_summary = generate_outcomes(df_complete, latent, scenario, rng, stats, cfg)
            df_final = apply_missingness(df_outcome, latent, rng, stats, scenario)

            # Compress for memory.
            df_final = compress_float_cols(df_final)
            scenario_parts.append(df_final)
            rs = summarise_repetition(df_final, scenario, rep)
            scenario_rep_summaries.append(rs)
            rep_summary_rows.append(rs)

            if rep == 1 or rep % 10 == 0 or rep == cfg.n_reps:
                print(
                    f"  rep {rep:03d}/{cfg.n_reps}: "
                    f"carehome={rs['observed_carehome_rate']:.3f}, "
                    f"death_pre={rs['observed_death_before_carehome_rate']:.3f}, "
                    f"death_after={rs['death_after_carehome_rate']:.3f}, "
                    f"censored={rs['observed_event_free_or_censored_rate']:.3f}"
                )

        scenario_df = pd.concat(scenario_parts, axis=0, ignore_index=True)
        if example_cols is None:
            example_cols = scenario_df.columns.tolist()

        ext = ".csv.gz" if cfg.write_csv_gz else ".csv"
        out_file = out_dir / "scenario_datasets" / f"{sid}{ext}"
        scenario_df.to_csv(out_file, index=False, compression="gzip" if cfg.write_csv_gz else None)
        print(f"[saved] {out_file} | rows={len(scenario_df):,} | cols={scenario_df.shape[1]}")

        # Scenario aggregate summary.
        agg = pd.DataFrame(scenario_rep_summaries)
        row = {
            "scenario_id": sid,
            "description": scenario["description"],
            "dgm_type": scenario["dgm_type"],
            "n_total_rows": len(scenario_df),
            "n_reps": cfg.n_reps,
            "n_per_rep": cfg.n_per_rep,
            "target_observed_carehome_rate": scenario["target_observed_carehome_rate"],
        }
        for c in [
            "observed_carehome_rate", "observed_death_before_carehome_rate",
            "observed_event_free_or_censored_rate", "death_after_carehome_rate",
            "any_death_5y_rate", "median_duration_years",
            "mean_true_risk_carehome_5y_observable_approx",
            "missing_rate_MMSE", "missing_rate_MRI_region_mean",
            "missing_rate_NLP_ischaemic_raw", "missing_rate_NLP_neuro_raw",
        ]:
            row[f"mean_{c}"] = float(agg[c].mean())
            row[f"sd_{c}"] = float(agg[c].std(ddof=1)) if len(agg) > 1 else np.nan
        scenario_summary_rows.append(row)

        # Missingness summary for this scenario.
        for col in scenario_df.columns:
            missingness_rows.append({
                "scenario_id": sid,
                "column": col,
                "missing_rate": float(scenario_df[col].isna().mean()),
            })

        # Free memory.
        del scenario_df, scenario_parts

    # Save tables.
    scenario_summary = pd.DataFrame(scenario_summary_rows)
    rep_summary = pd.DataFrame(rep_summary_rows)
    dgm_def = pd.DataFrame(SCENARIOS)
    feature_dict = build_feature_dictionary(example_cols or [])
    missingness_summary = pd.DataFrame(missingness_rows)

    scenario_summary.to_csv(out_dir / "tables" / "scenario_summary.csv", index=False)
    rep_summary.to_csv(out_dir / "tables" / "repetition_summary.csv", index=False)
    dgm_def.to_csv(out_dir / "tables" / "dgm_definition_table.csv", index=False)
    feature_dict.to_csv(out_dir / "tables" / "feature_dictionary.csv", index=False)
    missingness_summary.to_csv(out_dir / "tables" / "missingness_summary.csv", index=False)

    # Compact missingness by block.
    if not missingness_summary.empty and not feature_dict.empty:
        miss_block = missingness_summary.merge(feature_dict, on="column", how="left")
        miss_block_summary = (
            miss_block.groupby(["scenario_id", "block"], as_index=False)
            .agg(mean_missing_rate=("missing_rate", "mean"), max_missing_rate=("missing_rate", "max"), n_columns=("column", "nunique"))
        )
        miss_block_summary.to_csv(out_dir / "tables" / "missingness_by_block_summary.csv", index=False)

    audit = export_safety_audit(out_dir)
    write_readme(out_dir, cfg)

    print("\n" + "=" * 88)
    print("STEP C COMPLETE")
    print("=" * 88)
    print(f"Output folder: {out_dir.resolve()}")
    print("Core outputs:")
    print(" -", out_dir / "scenario_datasets")
    print(" -", out_dir / "tables" / "scenario_summary.csv")
    print(" -", out_dir / "tables" / "repetition_summary.csv")
    print(" -", out_dir / "tables" / "feature_dictionary.csv")
    print(" -", out_dir / "audit" / "export_safety_audit.csv")
    print("\nExport safety audit:")
    print(audit.to_string(index=False))
    print("\nScenario summary:")
    print(scenario_summary[[
        "scenario_id", "target_observed_carehome_rate", "mean_observed_carehome_rate",
        "mean_observed_death_before_carehome_rate", "mean_death_after_carehome_rate",
        "mean_observed_event_free_or_censored_rate",
    ]].to_string(index=False))


if __name__ == "__main__":
    run_stepC(CFG)
