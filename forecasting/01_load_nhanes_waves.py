"""
NHANES Multi-Wave Cognitive Impairment Loader
Four cycles: 2011-12, 2013-14, 2015-16, 2017-18
Outputs: data/nhanes_pooled.csv
"""

import pandas as pd
import numpy as np
import requests
import os

os.makedirs("data", exist_ok=True)

WAVES = {
    2012: {
        "demo": "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2011/DataFiles/DEMO_g.xpt",
        "cfq":  "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2011/DataFiles/CFQ_g.xpt",
        "mid_year": 2011.5,
    },
    2014: {
        "demo": "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2013/DataFiles/DEMO_h.xpt",
        "cfq":  "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2013/DataFiles/CFQ_h.xpt",
        "mid_year": 2013.5,
    },
    2016: {
        "demo": "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt",
        "cfq":  "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2015/DataFiles/CFQ_I.xpt",
        "mid_year": 2015.5,
    },
    2018: {
        "demo": "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
        "cfq":  "https://wwwn.cdc.gov/NCHS/Data/Nhanes/Public/2017/DataFiles/CFQ_J.xpt",
        "mid_year": 2017.5,
    },
}


def download_xpt(url, local_path):
    if not os.path.exists(local_path):
        print(f"  Downloading {url} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    else:
        print(f"  Using cached {local_path}")


def load_wave(wave_year, paths, mid_year):
    import io
    with open(paths["demo"], "rb") as f:
        demo = pd.read_sas(io.BytesIO(f.read()), format="xport")
    with open(paths["cfq"], "rb") as f:
        cfq = pd.read_sas(io.BytesIO(f.read()), format="xport")

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR"]].copy()
    demo = demo[demo["RIDAGEYR"] >= 60]

    cfq_cols = cfq.columns.tolist()
    print(f"  CFQ columns: {cfq_cols[:12]}")

    recall_col = None
    for candidate in ["CFDCSR", "CFDAST", "CFD410", "CFDCST1"]:
        if candidate in cfq_cols:
            recall_col = candidate
            break

    if recall_col is None:
        print(f"  WARNING: No recall column found for wave {wave_year}. Columns: {cfq_cols}")
        return None

    print(f"  Using recall column: {recall_col}")
    cfq_sub = cfq[["SEQN", recall_col]].copy()
    cfq_sub.rename(columns={recall_col: "recall_score"}, inplace=True)

    merged = demo.merge(cfq_sub, on="SEQN", how="inner")
    merged = merged.dropna(subset=["recall_score"])
    merged["impaired"]  = (merged["recall_score"] <= 4).astype(int)
    merged["wave_year"] = wave_year
    merged["mid_year"]  = mid_year

    return merged


def main():
    all_waves = []

    for wave_year, urls in WAVES.items():
        print(f"\nLoading NHANES {wave_year} wave ...")
        demo_path = f"data/demo_{wave_year}.xpt"
        cfq_path  = f"data/cfq_{wave_year}.xpt"

        try:
            download_xpt(urls["demo"], demo_path)
            download_xpt(urls["cfq"],  cfq_path)
            wave_df = load_wave(wave_year, {"demo": demo_path, "cfq": cfq_path}, urls["mid_year"])
        except Exception as e:
            print(f"  ERROR on wave {wave_year}: {e}")
            continue

        if wave_df is not None:
            print(f"  {len(wave_df):,} participants, {wave_df['impaired'].mean():.1%} impaired.")
            all_waves.append(wave_df)

    if not all_waves:
        print("No waves loaded.")
        return

    pooled = pd.concat(all_waves, ignore_index=True)
    pooled.to_csv("data/nhanes_pooled.csv", index=False)
    print(f"\nPooled dataset saved: {len(pooled):,} rows across {len(all_waves)} waves.")
    print(pooled.groupby("wave_year")["impaired"].agg(["count", "mean"]).rename(
        columns={"count": "n", "mean": "prevalence"}
    ))


if __name__ == "__main__":
    main()
