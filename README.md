# Dementia Risk Cohort Analysis · NHANES

**Live app:** https://dementia-risk-cohort.streamlit.app  
**Data:** CDC NHANES Cognitive Functioning Questionnaire (CFQ), cycles 2011-12 and 2013-14  
**Cohort:** 3,126 adults aged 60+, real public-use data

An interactive Streamlit dashboard analyzing cognitive impairment risk in older adults using two waves of real NHANES data. Built to demonstrate end-to-end data science: data wrangling, statistical modeling, experiment design, and communicating findings to both technical and non-technical audiences.

## Four tabs

| Tab | What it shows |
|-----|--------------|
| Cohort Profile | Age-stratified impairment rates, recall score distributions, income gradients |
| Prevalence Trends | Multi-wave analysis with bootstrap 95% CIs — 8.8 pp decline 2012→2014 |
| Predictive Model | Logistic regression, AUC=0.716, forest plot, coefficient table |
| Experiment Design | Interactive A/B test simulator with power curves and feasibility check |

## Stack
Python · Streamlit · Plotly · statsmodels · scikit-learn · SciPy · CDC NHANES (public use)

## Key findings
- Cognitive impairment prevalence declined 8.8 percentage points between 2011-12 (32.2%) and 2013-14 (23.4%), consistent with cohort-level improvements driven by rising educational attainment and better cardiovascular risk management
- Age is the strongest predictor (OR=1.099 per year, p<0.001); higher income-to-poverty ratio is protective (OR=0.810)
- Model AUC = 0.716 with McFadden R² = 0.103
- To detect a 5pp reduction in impairment with 80% power at α=0.05 requires 1,170 participants per arm

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
