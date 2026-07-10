# Dementia Risk Cohort Analysis — NHANES 2017–2018

**[View Live Dashboard →](https://amy-way05.github.io/dementia-risk-cohort/dementia_cohort_dashboard.html)**

---

## Overview

This project identifies a cognitive impairment risk cohort from NHANES 2017–2018 public health data, applying the same analytical workflow used in EHR-based population health research: cohort definition, multi-source data integration, quality control, statistical modeling, and interactive reporting.

**Analytic cohort:** 1,629 adults age 60+ | **Impairment rate:** 29.1% | **Mean age:** 70.3 years

---

## Methodology

**Cohort Definition**
Participants aged 60+ were flagged as cognitively impaired using clinically-grounded thresholds: animal fluency score <11 or digit symbol score <35, consistent with published NHANES cognitive aging literature.

**Data Integration**
Five NHANES modules merged on participant ID: demographics, cognitive function, hypertension, diabetes, and smoking history - mirroring multi-table EHR linkage across clinical domains.

**Quality Control**
Systematic missingness report across 8 key variables prior to analysis. Final analytic cohort retained after dropping participants with missing cognitive scores (5% missingness rate).

**Statistical Modeling**
Logistic regression (GLM) modeling cognitive impairment as a function of age, sex, hypertension, diabetes, smoking, and income-to-poverty ratio.

| Predictor | Odds Ratio | 95% CI | p-value |
|-----------|-----------|--------|---------|
| Age | 1.033 | 1.018–1.049 | <0.001 |
| Income-to-Poverty Ratio | 0.881 | 0.815–0.954 | 0.002 |

Age and income were the only statistically significant predictors, consistent with established dementia risk literature.

---

## Dashboard

Interactive 6-panel Plotly dashboard including:
- Impairment rate trend by age group
- Normalized cognitive score comparison
- Comorbidity prevalence by group
- Income vs. fluency scatter with 95% CI trend lines
- Box and violin distribution plots

---

## Stack

`Python` · `Pandas` · `NumPy` · `SciPy` · `Statsmodels` · `Plotly` · `NHANES Public Data`

---

*Amrutha Ravikumar · MPS Analytics, Northeastern University*
