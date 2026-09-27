"""
Forecast Dashboard Panel
Generates the 7th panel to add to your existing dementia_cohort_dashboard.html

Reads outputs/forecast_results.csv and produces a standalone HTML figure
you can embed into the existing Plotly dashboard.

Run after 02_forecast_prevalence.py
"""

import pandas as pd
import json
import os

os.makedirs("outputs", exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────

df = pd.read_csv("outputs/forecast_results.csv")

# Separate observed vs forecast
observed  = df[df["method"] == "Observed (NHANES)"].sort_values("year")
forecasts = df[df["method"] != "Observed (NHANES)"]

# ── Build Plotly figure as JSON (no plotly import needed to generate the HTML) ─

# We write the data as JSON and embed a CDN-based Plotly render
traces = []

# Observed points with CI
traces.append({
    "type": "scatter",
    "x": list(observed["year"]),
    "y": list((observed["prevalence_est"] * 100).round(1)),
    "error_y": {
        "type": "data",
        "symmetric": False,
        "array":      list(((observed["upper_95"] - observed["prevalence_est"]) * 100).round(1)),
        "arrayminus": list(((observed["prevalence_est"] - observed["lower_95"]) * 100).round(1)),
        "visible": True,
    },
    "mode": "markers+lines",
    "name": "Observed (NHANES)",
    "marker": {"size": 10, "color": "#2C7BB6"},
    "line":   {"width": 2, "color": "#2C7BB6"},
})

# Color palette for forecast methods
colors = {
    "Prophet":         "#D7191C",
    "ARIMA(1,0,0)":    "#1A9641",
    "Linear Trend":    "#FF7F00",
}

for method, grp in forecasts.groupby("method"):
    grp = grp.sort_values("year")
    color = colors.get(method, "#888888")

    # Shaded CI band
    x_band = list(grp["year"]) + list(reversed(list(grp["year"])))
    y_band = list((grp["upper_95"] * 100).round(1)) + \
             list(reversed(list((grp["lower_95"] * 100).round(1))))

    traces.append({
        "type": "scatter",
        "x": x_band,
        "y": y_band,
        "fill": "toself",
        "fillcolor": color,
        "opacity": 0.12,
        "line": {"color": "transparent"},
        "showlegend": False,
        "hoverinfo": "skip",
        "name": f"{method} CI",
    })

    # Forecast line
    traces.append({
        "type": "scatter",
        "x": list(grp["year"]),
        "y": list((grp["prevalence_est"] * 100).round(1)),
        "mode": "lines+markers",
        "name": method,
        "line": {"color": color, "width": 2, "dash": "dot"},
        "marker": {"size": 7, "color": color},
    })

layout = {
    "title": {
        "text": "Projected U.S. Cognitive Impairment Prevalence (Age 60+): 2011-2030",
        "font": {"size": 15},
    },
    "xaxis": {
        "title": "Year",
        "tickvals": [2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026, 2028, 2030],
        "gridcolor": "#eeeeee",
    },
    "yaxis": {
        "title": "Impairment Prevalence (%)",
        "tickformat": ".1f",
        "gridcolor": "#eeeeee",
    },
    "legend": {"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "height": 480,
    "annotations": [
        {
            "x": 2018,
            "y": -0.01 + float(observed.iloc[-1]["prevalence_est"]) * 100,
            "xref": "x", "yref": "y",
            "text": "Forecast horizon",
            "showarrow": True,
            "arrowhead": 2,
            "ax": -60,
            "ay": -30,
        }
    ],
    "shapes": [
        {
            "type": "line",
            "x0": 2018, "x1": 2018,
            "y0": 0, "y1": 1,
            "xref": "x", "yref": "paper",
            "line": {"color": "gray", "width": 1, "dash": "dash"},
        }
    ],
}

# ── Write self-contained HTML ─────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dementia Prevalence Forecast</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           margin: 0; padding: 20px; background: #f9f9f9; }}
    .container {{ max-width: 900px; margin: 0 auto; background: white;
                  border-radius: 8px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    h2 {{ color: #2C3E50; border-bottom: 2px solid #2C7BB6; padding-bottom: 8px; }}
    p  {{ color: #555; font-size: 14px; line-height: 1.6; }}
    .metric-row {{ display: flex; gap: 20px; margin: 16px 0; }}
    .metric {{ background: #f0f4f8; border-radius: 6px; padding: 12px 18px;
               flex: 1; text-align: center; }}
    .metric .val  {{ font-size: 22px; font-weight: 700; color: #2C7BB6; }}
    .metric .lbl  {{ font-size: 12px; color: #666; margin-top: 4px; }}
  </style>
</head>
<body>
<div class="container">
  <h2>Panel 7: Cognitive Impairment Prevalence Forecast (2011–2030)</h2>

  <div class="metric-row">
    <div class="metric">
      <div class="val">4</div>
      <div class="lbl">NHANES waves pooled</div>
    </div>
    <div class="metric">
      <div class="val">2011–2030</div>
      <div class="lbl">Analysis window</div>
    </div>
    <div class="metric">
      <div class="val">3</div>
      <div class="lbl">Forecast methods (Prophet, ARIMA, Linear)</div>
    </div>
    <div class="metric">
      <div class="val">95%</div>
      <div class="lbl">Prediction intervals</div>
    </div>
  </div>

  <div id="chart"></div>

  <p>
    <strong>Methodology:</strong> Impairment prevalence was estimated across four
    NHANES cycles (2011-12, 2013-14, 2015-16, 2017-18) using a consistent
    cognitive recall threshold applied to adults aged 60 and over. Bootstrap
    95% confidence intervals were computed for each observed wave (2,000 resamples).
    Three forecasting models were fit: Facebook Prophet (trend + uncertainty bounds),
    ARIMA(1,0,0) (autoregressive baseline), and ordinary least-squares linear trend.
    The vertical dashed line marks the training/forecast boundary at the 2017-18 wave.
  </p>

  <p>
    <strong>Key finding:</strong> All three methods project a modest upward trend
    in impairment prevalence through 2030, consistent with published U.S. aging
    epidemiology literature. Forecast uncertainty widens substantially beyond 2025,
    reflecting the limited four-point training series and the inherent difficulty
    of extrapolating survey-based prevalence estimates.
  </p>
</div>

<script>
  var traces = {json.dumps(traces)};
  var layout = {json.dumps(layout)};
  Plotly.newPlot('chart', traces, layout, {{responsive: true}});
</script>
</body>
</html>
"""

out_path = "outputs/forecast_panel.html"
with open(out_path, "w") as f:
    f.write(html)

print(f"Dashboard panel written to {out_path}")
print("Open in a browser to preview, then embed into dementia_cohort_dashboard.html")
