"""Build the cleaned dataset in Data_clean/ from the raw Sensor Logger recordings in data/.

Raw data is never modified. Each recording is read either from its extracted folder
(data/<activity>/<Name>/) or directly from its zip (data/<activity>/<Name>-<timestamp>.zip).

Outputs
    Data_clean/train/<activity>/, Data_clean/test/<activity>/   one CSV per recording: timestamp, ax, ay, az, gx, gy, gz
    Data_clean/manifest.csv               per-file provenance and quality control
    Data_clean/excluded.csv               recordings left out, with the reason and evidence
    Data_clean/trim_rule_impact.csv       what a fixed 1.5 s / 1.5 s trim with a 5 s minimum would keep

Run from the repository root:  python prepare_data.py
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"
OUT = ROOT / "Data_clean"

ACTIVITIES = ["standing", "walking", "jumping", "still"]
STATIC = {"standing", "still"}

TRIM_START_S = 1.0          # pick-up / button-press transient
TRIM_END_S = 0.5            # stop-button press
MAX_EXTRA_START_S = 2.5     # adaptive extra trim for static activities
STATIC_BLOCK_STD = 0.15     # m/s^2; a 0.5 s block above this is still a transient
MAX_LEN_S = 10.0
MIN_LEN_S = 4.0             # two full 2 s windows at 50 % overlap need >= 3 s; 4 s gives >= 3 windows
MERGE_TOL_NS = 5_000_000    # 5 ms
N_TEST_PER_ACTIVITY = 3     # last 3 recordings by start time are held out


def discover():
    """Return {(activity, recording_name): source} for every recording, preferring folders over zips."""
    recs = {}
    for act in ACTIVITIES:
        for p in sorted((RAW / act).iterdir()):
            m = re.match(r"([A-Za-z]+)_?(\d+)", p.name)
            if not m:
                continue
            name = f"{m.group(1).capitalize()}{int(m.group(2)):02d}"
            if p.is_dir() and (p / "Accelerometer.csv").exists():
                recs[(act, name)] = p
            elif p.suffix.lower() == ".zip":
                recs.setdefault((act, name), p)
    return recs


def read_csv(src, fname):
    if src.is_dir():
        return pd.read_csv(src / fname)
    with zipfile.ZipFile(src) as z:
        return pd.read_csv(io.BytesIO(z.read(fname)))


def load_recording(src):
    """Merge calibrated Accelerometer.csv and Gyroscope.csv by nearest timestamp (columns matched by name)."""
    acc = read_csv(src, "Accelerometer.csv").rename(columns={"x": "ax", "y": "ay", "z": "az"})
    gyr = read_csv(src, "Gyroscope.csv").rename(columns={"x": "gx", "y": "gy", "z": "gz"})
    meta = read_csv(src, "Metadata.csv").iloc[0]
    acc = acc[["time", "ax", "ay", "az"]].sort_values("time")
    gyr = gyr[["time", "gx", "gy", "gz"]].sort_values("time")
    merged = pd.merge_asof(acc, gyr, on="time", direction="nearest", tolerance=MERGE_TOL_NS)
    n_unmatched = int(merged[["gx", "gy", "gz"]].isna().any(axis=1).sum())
    merged = merged.dropna().rename(columns={"time": "timestamp"}).reset_index(drop=True)
    info = {
        "n_acc": len(acc), "n_gyr": len(gyr), "n_unmatched": n_unmatched,
        "device": meta["device name"], "device_id": meta["device id"], "platform": meta["platform"],
        "recording_time": meta["recording time"], "sample_rate_ms": meta["sampleRateMs"],
        "app_version": meta["appVersion"],
    }
    return merged, info


def raw_qc(df):
    t = df["timestamp"].to_numpy(np.int64)
    dt = np.diff(t) / 1e9
    return {
        "n_nan": int(df.isna().sum().sum()),
        "n_dup_ts": int(pd.Series(t).duplicated().sum()),
        "n_non_monotonic": int((dt <= 0).sum()),
        "dt_min_ms": dt.min() * 1000,
        "dt_max_ms": dt.max() * 1000,
        "dt_std_ms": dt.std() * 1000,
        "measured_hz": 1 / np.median(dt),
        "original_duration_s": (t[-1] - t[0]) / 1e9,
        "original_rows": len(df),
    }


def trim(df, activity, hz):
    """Fixed trim at both ends; for static activities extend the start trim past any remaining pick-up movement."""
    start = int(round(TRIM_START_S * hz))
    end = len(df) - int(round(TRIM_END_S * hz))
    extra = 0.0
    if activity in STATIC:
        mag = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2).to_numpy()
        block = int(round(0.5 * hz))
        while extra < MAX_EXTRA_START_S and mag[start:start + block].std() > STATIC_BLOCK_STD:
            start += block
            extra += 0.5
    end = min(end, start + int(round(MAX_LEN_S * hz)))
    return df.iloc[start:end].reset_index(drop=True), extra


def main():
    recs = discover()
    rows, excluded = [], []
    for (act, name), src in sorted(recs.items()):
        df, info = load_recording(src)
        qc = raw_qc(df)
        clean, extra = trim(df, act, qc["measured_hz"])
        dur = (clean["timestamp"].iloc[-1] - clean["timestamp"].iloc[0]) / 1e9
        mag = np.sqrt(clean["ax"] ** 2 + clean["ay"] ** 2 + clean["az"] ** 2)
        notes = []
        if info["n_acc"] != info["n_gyr"]:
            notes.append(f"acc/gyro rows {info['n_acc']}/{info['n_gyr']}")
        if info["n_unmatched"]:
            notes.append(f"{info['n_unmatched']} unmatched rows dropped")
        if extra:
            notes.append(f"extra {extra:.1f} s start trim (pick-up transient)")
        row = dict(original_source=src.relative_to(RAW).as_posix(), recording=name, activity=act, **info, **qc,
                   duration_s=dur, n_samples=len(clean), acc_mag_std=mag.std(),
                   gyr_mag_std=np.sqrt(clean["gx"] ** 2 + clean["gy"] ** 2 + clean["gz"] ** 2).std(),
                   extra_trim_s=extra)
        if dur < MIN_LEN_S:
            excluded.append(dict(original_source=row["original_source"], activity=act, recording_time=info["recording_time"],
                                 reason=f"only {dur:.2f} s left after trimming (minimum {MIN_LEN_S:.0f} s)",
                                 evidence=f"pick-up movement in the first {TRIM_START_S + extra:.1f} s "
                                          f"(0.5 s block std > {STATIC_BLOCK_STD} m/s^2); "
                                          f"original {qc['original_duration_s']:.2f} s"))
            continue
        row["notes"] = "; ".join(notes)
        row["_df"] = clean
        rows.append(row)

    man = pd.DataFrame(rows).sort_values(["activity", "recording_time"]).reset_index(drop=True)
    man["split"] = "train"
    for act, g in man.groupby("activity"):
        man.loc[g.index[-N_TEST_PER_ACTIVITY:], "split"] = "test"
    man["new_name"] = man.groupby("activity").cumcount().add(1).map("{:02d}".format)
    man["new_name"] = man["activity"] + "_" + man["new_name"] + ".csv"

    for split in ("train", "test"):
        d = OUT / split
        for f in d.rglob("*.csv"):
            f.unlink()
        for act in ACTIVITIES:
            (d / act).mkdir(parents=True, exist_ok=True)
    for _, r in man.iterrows():
        r["_df"].to_csv(OUT / r["split"] / r["activity"] / r["new_name"], index=False)

    cols = ["new_name", "original_source", "recording", "activity", "split", "recording_time",
            "original_duration_s", "duration_s", "original_rows", "n_samples", "measured_hz",
            "dt_min_ms", "dt_max_ms", "dt_std_ms", "n_nan", "n_dup_ts", "n_non_monotonic", "n_unmatched",
            "extra_trim_s", "acc_mag_std", "gyr_mag_std", "notes"]
    man[cols].round(4).to_csv(OUT / "manifest.csv", index=False)
    pd.DataFrame(excluded, columns=["original_source", "activity", "recording_time", "reason", "evidence"]) \
        .to_csv(OUT / "excluded.csv", index=False)

    # What the stricter rule (fixed 1.5 s trim at both ends, keep only >= 5 s) would do to this dataset
    all_dur = pd.DataFrame([(a, n, raw_qc(load_recording(s)[0])["original_duration_s"]) for (a, n), s in recs.items()],
                           columns=["activity", "recording", "original_duration_s"])
    all_dur["fixed_1.5s_trim_s"] = all_dur["original_duration_s"] - 3.0
    impact = all_dur.groupby("activity").agg(
        recordings=("recording", "count"), original_s=("original_duration_s", "sum"),
        fixed_rule_files_kept=("fixed_1.5s_trim_s", lambda s: int((s >= 5).sum())),
        fixed_rule_seconds_kept=("fixed_1.5s_trim_s", lambda s: s[s >= 5].sum()))
    adopted = man.groupby("activity").agg(adopted_rule_files_kept=("new_name", "count"),
                                          adopted_rule_seconds_kept=("duration_s", "sum"))
    impact = impact.join(adopted).round(1)
    impact.to_csv(OUT / "trim_rule_impact.csv")

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    print("Device:", man["device"].unique(), "| platform:", man["platform"].unique(),
          "| device ids:", man["device_id"].nunique(), "| app version:", man["app_version"].unique(),
          "| sampleRateMs:", man["sample_rate_ms"].unique())
    print("Raw QC totals: NaN", man["n_nan"].sum(), "| duplicate ts", man["n_dup_ts"].sum(),
          "| non-monotonic", man["n_non_monotonic"].sum(), "| unmatched", man["n_unmatched"].sum(),
          f"| dt range {man.dt_min_ms.min():.3f}-{man.dt_max_ms.max():.3f} ms")
    print(f"Measured Hz: min {man.measured_hz.min():.3f} max {man.measured_hz.max():.3f} std {man.measured_hz.std():.4f}\n")
    print(man[["new_name", "original_source", "split", "original_duration_s", "duration_s", "measured_hz",
               "dt_max_ms", "acc_mag_std", "gyr_mag_std", "notes"]].round(3).to_string(index=False))
    print("\nExcluded:\n", pd.read_csv(OUT / "excluded.csv").to_string(index=False))
    print("\nTrim-rule impact:\n", impact.to_string())
    summ = man.groupby(["activity", "split"]).agg(files=("new_name", "count"), seconds=("duration_s", "sum")).unstack()
    print("\n", summ.round(1).to_string())


if __name__ == "__main__":
    main()
