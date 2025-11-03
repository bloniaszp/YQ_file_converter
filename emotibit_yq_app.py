import streamlit as st
import zipfile
import tempfile
import os
import io
import json
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import pandas as pd

# --------------------------------------------------
# PAGE CONFIG + LIGHT STYLING
# --------------------------------------------------
st.set_page_config(page_title="EmotiBit → YQ Converter", page_icon="📦", layout="centered")

# global CSS
st.markdown(
    """
    <style>
    /* take away default top padding */
    .block-container {padding-top: 1.5rem; max-width: 900px;}
    /* header */
    .hero {
        background: radial-gradient(circle at top, #2563eb 0%, #0f172a 45%, #0f172a 100%);
        border-radius: 18px;
        padding: 1.6rem 1.75rem 1.3rem 1.75rem;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    .hero-title {font-size: 1.6rem; font-weight: 700; margin-bottom: 0.35rem;}
    .hero-sub {opacity: 0.8; font-size: 0.9rem;}
    /* upload card */
    .card {
        background: white;
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 14px;
        padding: 1.25rem 1.35rem 1.15rem 1.35rem;
        box-shadow: 0 8px 20px rgba(15,23,42,0.03);
        margin-bottom: 1rem;
    }
    /* footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 2.5rem;
    }
    /* hide main menu & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 1. JSON parsing
# --------------------------------------------------
def load_emotibit_json(json_path: str) -> Tuple[dict, Dict[str, dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"{json_path} is empty or invalid")

    raw_meta = data[0].get("info", {})

    device_name = (
        raw_meta.get("name")
        or raw_meta.get("source_id")
        or raw_meta.get("type")
        or "UnknownDevice"
    )
    device_id = (
        raw_meta.get("device_id")
        or raw_meta.get("source_id")
        or device_name.replace(" ", "_")
    )

    device_meta = dict(raw_meta)
    device_meta["__device_name__"] = device_name
    device_meta["__device_id__"] = device_id

    channels: Dict[str, dict] = {}
    for entry in data[1:]:
        info = entry.get("info", {})
        name = info.get("name", info.get("type", "Channel"))
        tags = info.get("typeTags", [])
        sr = info.get("nominal_srate", None)
        units = info.get("units", None)
        for tag in tags:
            label = f"{name}_{tag}".replace(" ", "")
            channels[tag] = {
                "label": label,
                "nominal_srate": sr,
                "units": units,
                "raw_info": info,
            }

    return device_meta, channels

# --------------------------------------------------
# 2. CSV parsing
# --------------------------------------------------
def parse_emotibit_csv(csv_path: str, device_created_at: str, channels_meta: Dict[str, dict]) -> pd.DataFrame:
    base_dt = None
    if device_created_at:
        base_dt = datetime.strptime(device_created_at, "%Y-%m-%d_%H-%M-%S-%f")

    records = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue

            try:
                system_ms = int(parts[0])
            except ValueError:
                continue

            tag = parts[3]
            try:
                n_samples = int(parts[2])
            except ValueError:
                n_samples = 1

            values = parts[6:]
            if tag not in channels_meta:
                continue

            ch_meta = channels_meta.get(tag, {})
            sr = ch_meta.get("nominal_srate") or 25.0

            if base_dt is not None:
                base_ts = base_dt + timedelta(milliseconds=system_ms)
                base_ts_ms = int(base_ts.timestamp() * 1000)
            else:
                base_ts_ms = system_ms

            for idx, val in enumerate(values[:n_samples]):
                try:
                    v = float(val)
                except ValueError:
                    continue
                ts_ms = base_ts_ms + int((idx * 1.0 / sr) * 1000.0)
                records.append({"timestamp": ts_ms, "tag": tag, "value": v})

    if not records:
        return pd.DataFrame(columns=["timestamp"])

    df_long = pd.DataFrame.from_records(records)
    df_wide = (
        df_long.pivot_table(index="timestamp", columns="tag", values="value", aggfunc="mean")
        .sort_index()
        .reset_index()
    )

    first_ts = df_wide["timestamp"].iloc[0]
    df_wide.insert(1, "Time", (df_wide["timestamp"] - first_ts) / 1000.0)

    rename_map = {}
    for tag, meta in channels_meta.items():
        if tag in df_wide.columns:
            rename_map[tag] = meta.get("label", tag)
    df_wide = df_wide.rename(columns=rename_map)

    return df_wide

# --------------------------------------------------
# 3. YQ writer
# --------------------------------------------------
def write_yq_folder(out_dir: str, device_df: pd.DataFrame, device_meta: dict):
    os.makedirs(out_dir, exist_ok=True)

    device_id = device_meta.get("__device_id__", device_meta.get("device_id", "emotibit_device"))
    device_name = device_meta.get("__device_name__", device_meta.get("name", "EmotiBit"))
    fn_stub = device_id.lower().replace(" ", "_")
    device_csv_name = f"{fn_stub}_device.csv"

    readme_text = f"""Hi! I'm a small file meant to describe the contents of your folder. 

You: Quantified saves the data recorded from each of your devices as separate "csv" files. There is an additional file with the metadata for each device.

The data gathered from the web browser can have unreliable time synchrony or sampling rates, so be aware of its usage for research purposes. 

If you have questions or suggestions, please visit the repository at https://github.com/esromerog/You-Quantified.

This folder contains the following files: {device_csv_name}, metadata.csv
""".strip()

    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme_text)

    device_df.to_csv(os.path.join(out_dir, device_csv_name), index=False)

    sampling_rate_guess = 0
    if "Time" in device_df.columns and len(device_df) > 1:
        diffs = device_df["Time"].diff().dropna()
        if not diffs.empty and diffs.median() > 0:
            sampling_rate_guess = round(1.0 / diffs.median())

    meta_df = pd.DataFrame(
        [
            {
                "recording id": f"{device_id} : device",
                "file name": device_csv_name,
                "device id": device_id,
                "device name": device_name,
                "type": "physio",
                "sampling_rate": sampling_rate_guess,
            }
        ]
    )
    meta_df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)

# --------------------------------------------------
# 4. Robust matcher
# --------------------------------------------------
def find_emotibit_sessions(root_dir: str) -> List[dict]:
    all_jsons, all_csvs = [], []
    for root, _, files in os.walk(root_dir):
        if "__MACOSX" in root:
            continue
        for f in files:
            if f.startswith("._"):
                continue
            fp = os.path.join(root, f)
            if f.lower().endswith(".json") and "_info" in f.lower():
                all_jsons.append(fp)
            elif f.lower().endswith(".csv"):
                all_csvs.append(fp)

    sessions = []
    for json_path in all_jsons:
        stem = os.path.splitext(os.path.basename(json_path))[0]
        idx = stem.lower().find("_info")
        if idx == -1:
            continue
        base_key = stem[:idx]

        # match csv that starts with base_key
        candidates = []
        for csv_path in all_csvs:
            csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
            if csv_stem.startswith(base_key):
                candidates.append(csv_path)
        if not candidates:
            continue
        candidates.sort(key=lambda p: len(os.path.basename(p)))
        csv_path = candidates[0]
        sessions.append({"json": json_path, "csv": csv_path, "name": base_key})
    return sessions

# --------------------------------------------------
# 5. ZIP helper
# --------------------------------------------------
def zip_directory(src_dir: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, src_dir)
                z.write(full_path, rel_path)
    buf.seek(0)
    return buf.read()

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">📦 EmotiBit → You:Quantified converter</div>
      <div class="hero-sub">Drop an EmotiBit export (.zip) → get a YQ-style folder (.zip) you can upload.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# UPLOAD CARD
# --------------------------------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload EmotiBit ZIP", type=["zip"])
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded is not None:
    st.info(f"📁 Received: **{uploaded.name}** ({uploaded.size/1024:.1f} kB)")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_zip_path = os.path.join(tmpdir, "input.zip")
            with open(in_zip_path, "wb") as f:
                f.write(uploaded.read())

            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(in_zip_path, "r") as z:
                z.extractall(extract_dir)

            found_files = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), extract_dir)
                    found_files.append(rel)

            with st.expander("📄 Files inside uploaded ZIP", expanded=False):
                st.code("\n".join(found_files) or "(empty)")

            sessions = find_emotibit_sessions(extract_dir)

            if not sessions:
                st.error("😕 I couldn't find any `<something>.csv` + `<something>_info*.json` pair. Check the ZIP and try again.")
            else:
                yq_out_dir = os.path.join(tmpdir, "YQ_out")
                os.makedirs(yq_out_dir, exist_ok=True)

                for sess in sessions:
                    device_meta, channels = load_emotibit_json(sess["json"])
                    created_at = device_meta.get("created_at", None)
                    df_device = parse_emotibit_csv(sess["csv"], created_at, channels)

                    sess_out_dir = os.path.join(yq_out_dir, sess["name"])
                    os.makedirs(sess_out_dir, exist_ok=True)
                    write_yq_folder(sess_out_dir, df_device, device_meta)

                final_zip = zip_directory(yq_out_dir)
                st.success(f"✅ Converted **{len(sessions)}** session(s).")
                st.download_button(
                    "⬇️ Download YQ_out.zip",
                    data=final_zip,
                    file_name="YQ_out.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    except Exception as e:
        st.error("🚨 Error during conversion.")
        st.exception(e)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown('<div class="footer">Built for EmotiBit → YQ data pipelines.</div>', unsafe_allow_html=True)
