#!/usr/bin/env bash
set -euo pipefail

python src/stepC6_publication_analysis/stepC6_publication_analysis.py \
  --c2-dir fully_synthetic_stepC2_model_comparison \
  --c3-dir fully_synthetic_stepC3_competing_risk_evaluation \
  --c4-dir fully_synthetic_stepC4_extended_models
