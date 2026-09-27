"""
Cognitive Impairment Prevalence Forecasting
Uses four NHANES waves (2011-18) to build a time-series forecast
of age-60+ cognitive impairment prevalence to 2030.

Runs BOTH:
  - Prophet (primary, trend + uncertainty intervals)
  - ARIMA via statsmodels (robustness check)

Outputs: outputs/forecast_results.csv, outputs/forecast_panel.html
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Load pooled data ─────────────────────────────────────────────────────────

df = pd.read_csv("data/nhanes_pooled.csv")

# Compute weighted prevalence per wave (unweighted for simplicity;
# add WTMEC2YR weighting if you want publication-grade estimates)
prevalence = (
    df.groupby(["wave_year", "mid_year"])["impaired"]
    .agg(n="count", impaired_n="sum")
    .reset_index()
)
prevalence["prevalence"] = prevalence["impaired_n"] / prevalence["n"]

# Bootstrap 95% CI per wave
def bootstrap_ci(x, n_boot=2000, ci=0.95):
    boot_means = [np.mean(np.random.choice(x, size=len(x), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi

print("Computing bootstrap CIs per wave ...")
cis = []
for wy, grp in df.groupby("wave_year"):
    lo, hi = bootstrap_ci(grp["impaired"].values)
    cis.append({"wave_year": wy, "ci_lo": lo, "ci_hi": hi})

prevalence = prevalence.merge(pd.DataFrame(cis), on="wave_year")
print(prevalence[["wave_year", "n", "prevalence", "ci_lo", "ci_hi"]].to_string(index=False))

# ── Prophet forecast ─────────────────────────────────────────────────────────

try:
    from prophet import Prophet

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(prevalence["mid_year"].astype(str) + "-07-01"),
        "y":  prevalence["prevalence"],
    })

    m = Prophet(
        interval_width=0.95,
        changepoint_prior_scale=0.3,  # flexible trend
        seasonality_mode="additive",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=6, freq="2YE")  # 2029-30
    forecast = m.predict(future)

    prophet_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    prophet_out["year"] = prophet_out["ds"].dt.year
    prophet_out["method"] = "Prophet"
    prophet_out.rename(columns={
        "yhat": "prevalence_est",
        "yhat_lower": "lower_95",
        "yhat_upper": "upper_95",
    }, inplace=True)

    print("\nProphet forecast (2019-2030):")
    print(prophet_out[prophet_out["year"] >= 2019][
        ["year", "prevalence_est", "lower_95", "upper_95"]
    ].to_string(index=False))

except ImportError:
    print("Prophet not installed. Using ARIMA only. "
          "Install with: pip install prophet")
    prophet_out = None


# ── ARIMA fallback / robustness check ────────────────────────────────────────

try:
    from statsmodels.tsa.arima.model import ARIMA

    ts = prevalence.set_index("mid_year")["prevalence"].sort_index()

    # ARIMA(1,0,0) with only 4 points -- keep it simple
    model = ARIMA(ts, order=(1, 0, 0))
    result = model.fit()

    forecast_years = [2019.5, 2021.5, 2023.5, 2025.5, 2027.5, 2029.5]
    steps = len(forecast_years)
    pred = result.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    pred_ci   = pred.conf_int(alpha=0.05)

    arima_out = pd.DataFrame({
        "year":            [int(y) for y in forecast_years],
        "prevalence_est":  pred_mean.values,
        "lower_95":        pred_ci.iloc[:, 0].values,
        "upper_95":        pred_ci.iloc[:, 1].values,
        "method":          "ARIMA(1,0,0)",
    })

    print("\nARIMA(1,0,0) forecast (2019-2030):")
    print(arima_out.to_string(index=False))

except ImportError:
    print("statsmodels not installed. Using linear trend only. "
          "Install with: pip install statsmodels")
    arima_out = None


# ── Linear trend (always runs as baseline) ───────────────────────────────────

x = prevalence["mid_year"].values
y = prevalence["prevalence"].values

slope, intercept, r, p, se = stats.linregress(x, y)
future_years = np.arange(2019.5, 2031, 2)
trend_est    = slope * future_years + intercept

# Prediction interval via t-distribution
n   = len(x)
x_bar = x.mean()
t_crit = stats.t.ppf(0.975, df=n - 2)
s_err  = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))

pred_se = s_err * np.sqrt(
    1 + 1/n + (future_years - x_bar)**2 / np.sum((x - x_bar)**2)
)

linear_out = pd.DataFrame({
    "year":           [int(y) for y in future_years],
    "prevalence_est": trend_est,
    "lower_95":       trend_est - t_crit * pred_se,
    "upper_95":       trend_est + t_crit * pred_se,
    "method":         "Linear Trend",
})

print(f"\nLinear trend: slope={slope:.4f}/yr, R²={r**2:.3f}, p={p:.3f}")
print(linear_out.to_string(index=False))


# ── Save results ─────────────────────────────────────────────────────────────

import os
os.makedirs("outputs", exist_ok=True)

observed_out = pd.DataFrame({
    "year":           prevalence["mid_year"].apply(lambda x: int(x) + 1),
    "prevalence_est": prevalence["prevalence"],
    "lower_95":       prevalence["ci_lo"],
    "upper_95":       prevalence["ci_hi"],
    "method":         "Observed (NHANES)",
})

all_results = [observed_out, linear_out]
if prophet_out is not None:
    all_results.append(prophet_out[prophet_out["year"] >= 2019][[
        "year", "prevalence_est", "lower_95", "upper_95", "method"
    ]])
if arima_out is not None:
    all_results.append(arima_out)

results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv("outputs/forecast_results.csv", index=False)
print("\nResults saved to outputs/forecast_results.csv")
