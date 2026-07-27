# Project Overview

This project is a synthetic-methods study for dementia institutionalisation prediction. It uses the structure of an internal SLAM dementia imaging and clinical dataset to design a simulation framework for time-to-care-home entry and 5-year competing-risk prediction.

The repository is not a clinical prediction model release. It is an auditable
code and documentation package for testing methodology before real care-home
entry outcomes are available for direct modelling. Exact numerical
reproduction additionally requires the local fully synthetic data package,
which is not distributed here.

## Core Question

The project asks how different survival and competing-risk approaches behave when the endpoint is care-home entry or institutionalisation and death before care home can prevent the endpoint from being observed.

The analysis compares simple, interpretable Cox models with higher-dimensional penalised Cox models and strict leakage-guarded XGBoost survival models. It also evaluates whether individualised competing-risk cumulative incidence predictions improve on a nonparametric Aalen-Johansen baseline.

## Main Stages

1. Start from the internal SLAM diagnostic modelling data structure.
2. Build internal semi-synthetic Step B data using real baseline predictors and simulated care-home times.
3. Identify and fix leakage risks in survival-model evaluation.
4. Generate exportable fully synthetic Step C data.
5. Run Step C2 cause-specific survival model comparison.
6. Run Step C3 5-year competing-risk absolute-risk evaluation.
7. Release code and documentation only unless a specific aggregate artifact is
   separately reviewed and approved.

## Repository Boundary

This repository does not contain real SLAM records, internal Step B data,
synthetic scenario datasets, numerical result tables, or figures. It contains
code, English documentation, environment files, and automated tests.

Large scenario datasets and per-person predictions are intentionally excluded.
