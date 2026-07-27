# Step C4 Extended Model Comparison

Step C4 extends the completed Step C2 and Step C3 analyses. It reads the fully
synthetic Step C package but does not regenerate the data or change the DGM.

## Models

- Fine-Gray through R `cmprsk`, using DGM-informed or training-reduced all-safe
  predictors.
- Two-cause Random Survival Forest models through `scikit-survival`.
- Two-cause Gradient Boosting Survival models through `scikit-survival`.
- Canonical DeepSurv through `pycox.models.CoxPH`, fitted separately for the
  two cause-specific hazards.
- Canonical competing-risk DeepHit through `pycox.models.DeepHit`, retaining
  status 0, 1, and 2 after time discretisation.

For cause-specific models, five-year cumulative incidence is reconstructed from
the estimated care-home and death cumulative hazards.

## Neural Training

DeepSurv and DeepHit use:

- preprocessing fitted inside the outer training sample;
- a 20% internal validation split;
- ReLU networks with adaptive hidden width and 0.10 dropout;
- Adam optimisation with early stopping;
- network weights fitted on the 80% development portion of the outer training
  set and not refitted after model selection;
- for DeepSurv, baseline hazards estimated on the complete outer training set
  after network fitting;
- no use of outer test data for training or tuning.

DeepHit uses 32 time bins in full mode. Its likelihood weight `alpha` is chosen
from `0.2, 0.5, 0.8, 1.0` by internal-validation five-year Brier score.

The reported five-year Brier score is the observed-status binary Brier score.
It is valid here because the audit confirms no loss to follow-up before five
years and death before care-home entry is an observed competing outcome.
Generic single-event survival IPCW Brier routines are not applied to CIF
predictions; those output fields remain unavailable rather than mixing
cause-specific survival and cumulative-incidence estimands.

## Leakage Controls

The all-safe set permits baseline demographic, cognition, MRI, NLP, WMH,
comorbidity, and deprivation blocks. It excludes:

- identifiers and scenario/repetition labels;
- outcomes and post-baseline variables;
- hazards, risks, censoring, and simulation truth;
- diagnosis-reference blocks and any role marked not for prediction.

The full-run gate reads `predictor_leakage_audit_C4.csv`; a missing or non-empty
audit fails the gate. Every available core and neural model must complete.

## Execution

A conventional full run is:

```bash
bash scripts/run_stepC4_full.sh
```

The checkpointed scenario-sharded runner is:

```bash
python run_stepC4_postQC_sharded.py --max-workers 8
```

The sharded runner reuses only previously verified classical rows whose actual
selected-predictor strings contain no forbidden field. Strict all-safe RSF and
GBSA rows and all canonical neural rows are fitted anew. It then validates and
merges exactly 4,000 unique scenario-repetition-model rows.

Generated tables and figures are ignored by Git and require explicit author
approval before public release.

## Oracle-Screen Qualification

Oracle-exceedance screens remain publication-blocking in the seven PH
scenarios. In S4, the exported risk and LP are approximate non-PH proxies, so
an apparent model exceedance is retained as a diagnostic but is not interpreted
as leakage by itself. S4 truth-based errors are exploratory; observed-outcome
metrics remain primary.
