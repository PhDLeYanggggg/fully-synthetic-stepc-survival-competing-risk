# Research Rationale

Care-home entry and institutionalisation are clinically meaningful downstream outcomes in dementia research. They reflect a combination of cognitive decline, functional impairment, health burden, social context, and competing mortality risk. However, when the real care-home outcome is not yet available or fully curated, a direct clinical prediction model cannot be responsibly trained and evaluated.

This project therefore uses simulation as a methodological bridge. The aim is to prepare the modelling pipeline, define outcome coding, compare modelling families, and identify evaluation pitfalls before real outcomes become available.

## Why Simulation Was Needed

The internal SLAM diagnostic modelling dataset already provided a realistic structure of baseline predictors. These predictors included demographics, cognitive scores, MRI-derived regional volume percentages, MRI composite features, NLP pathology features, comorbidity variables, deprivation measures, and WMH-related features.

The missing component was a validated real time-to-care-home endpoint. A simulation framework made it possible to:

- implement a time-to-event pipeline;
- test how death before care home should be handled;
- compare Cox, penalised Cox, and XGBoost survival models;
- evaluate whether complex models genuinely improve on simpler models;
- expose leakage risks from outcome, time, death, or truth variables;
- prepare a reproducible workflow for later real-outcome validation.

## Scientific Positioning

The results in this repository are synthetic benchmark results. They support methodological decisions but do not estimate clinical performance on real patients. Any future real-world claims require validation using real care-home outcome data in an authorised environment.
