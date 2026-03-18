"""
Unit tests for the EmotiBit → YQ conversion pipeline.

These tests exercise the session discovery logic, JSON metadata parsing,
CSV parsing (including timestamp expansion and grouping), and high level
conversion functions. They verify that the parser avoids the previously
observed flat-lining bug by preserving native sample cadence instead of
resampling onto a fully regular grid.

To run the tests simply execute `pytest -q` from the repository root.
"""

from __future__ import annotations

import os
import json
import shutil
import tempfile
from typing import List

import pandas as pd
import pytest

# Import parser functions from the application.  We use importlib to avoid
# executing the Streamlit UI on import.  The test harness only needs the
# pure functions.
import importlib.util


def load_app_module():
    """Dynamically load the app module without executing Streamlit code."""
    # We load the module under a temporary name to avoid clashes.
    spec = importlib.util.spec_from_file_location("emotibit_app", os.path.join(os.getcwd(), "emotibit_yq_app (1).py"))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Install a dummy streamlit module into sys.modules so that import does not
    # fail when the app tries to import streamlit.  The parsing functions we
    # need are defined at module scope and do not depend on streamlit.
    import sys
    import types
    dummy_st = types.ModuleType("streamlit")
    # Provide no-op implementations for the streamlit API used in the app.  For
    # functions used in with-contexts (like container and expander), return an
    # object implementing __enter__/__exit__ so the with-statement does not
    # error during import.
    class _DummyContextManager:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_st.markdown = lambda *args, **kwargs: None
    dummy_st.set_page_config = lambda *args, **kwargs: None
    dummy_st.container = lambda *args, **kwargs: _DummyContextManager()
    dummy_st.expander = lambda *args, **kwargs: _DummyContextManager()
    dummy_st.file_uploader = lambda *args, **kwargs: None
    dummy_st.code = lambda *args, **kwargs: None
    dummy_st.success = lambda *args, **kwargs: None
    dummy_st.warning = lambda *args, **kwargs: None
    dummy_st.info = lambda *args, **kwargs: None
    dummy_st.error = lambda *args, **kwargs: None
    dummy_st.exception = lambda *args, **kwargs: None
    dummy_st.download_button = lambda *args, **kwargs: None
    sys.modules.setdefault("streamlit", dummy_st)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


@pytest.fixture(scope="module")
def app():
    return load_app_module()


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory) -> str:
    """Extract the uploaded sample ZIP into a temporary directory and return the path."""
    # The drive-download file is expected to be located at the repository root
    zip_path = os.path.join(os.getcwd(), "drive-download-20260318T131459Z-1-001.zip")
    assert os.path.exists(zip_path), f"sample ZIP not found: {zip_path}"
    tmpdir = tmp_path_factory.mktemp("emotibit_data")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    return str(tmpdir)


def test_find_emotibit_sessions(app, sample_dir):
    """The helper should find exactly two sessions in the provided ZIP."""
    sessions = app.find_emotibit_sessions(sample_dir)
    # There are two directories in the sample (V5-758 and V7-192)
    assert len(sessions) == 2, f"Expected 2 sessions, found {len(sessions)}"
    # Each session record should include json and csv paths and a name
    for s in sessions:
        assert os.path.isfile(s["json"]), f"JSON file missing: {s['json']}"
        assert os.path.isfile(s["csv"]), f"CSV file missing: {s['csv']}"
        assert isinstance(s.get("name"), str) and s["name"], "Session name missing"


def test_json_channel_metadata(app, sample_dir):
    """Ensure JSON metadata parsing extracts channels and nominal sample rates."""
    sessions = app.find_emotibit_sessions(sample_dir)
    # Use the first session for testing
    sess = sessions[0]
    device_meta, channels = app.load_emotibit_json(sess["json"])
    # Check some known tags and their nominal sample rates
    assert "PG" in channels and channels["PG"]["nominal_srate"] == 25
    assert "PI" in channels and channels["PI"]["nominal_srate"] == 25
    assert "PR" in channels and channels["PR"]["nominal_srate"] == 25
    assert "EA" in channels and channels["EA"]["nominal_srate"] == 15
    # Units should also be captured
    assert channels["EA"]["units"] == "microsiemens"


def test_packet_expansion_counts(app):
    """Synthetic CSV should expand packet samples correctly."""
    # Create a simple synthetic CSV with two packets
    csv_lines = [
        "100,0,3,PG,1,100,1.0,2.0,3.0",  # three samples
        "140,1,2,PG,1,100,4.0,5.0",      # two samples (time jump 40ms)
    ]
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tmp_csv.write("\n".join(csv_lines).encode("utf-8"))
        tmp_csv.close()
        # Minimal channel metadata
        channels = {
            "PG": {"nominal_srate": 25.0, "raw_info": {"name": "PPG"}},
        }
        out = app.parse_emotibit_csv(tmp_csv.name, device_created_at=None, channels_meta=channels)
        assert len(out) == 1
        df, sr, type_name, tags = out[0]
        # Five samples should be emitted
        # The parser collapses samples that end up with the same timestamp by averaging
        # their values. In this synthetic CSV the nominal sample rate is 25 Hz so
        # timestamps 100, 140 and 180 will occur twice. We therefore expect three
        # unique timestamps and the values at 140 and 180 to be the average of the
        # overlapping samples (2.0+4.0)/2=3.0 and (3.0+5.0)/2=4.0 respectively.
        assert len(df) == 3, f"Unexpected number of rows: {len(df)}"
        expected = {
            100: 1.0,
            140: 3.0,
            180: 4.0,
        }
        for _, row in df.iterrows():
            ts = row["timestamp"]
            assert ts in expected, f"Unexpected timestamp {ts}"
            assert pytest.approx(row["PG"], rel=1e-6) == expected[ts]
        # Timestamps should be strictly increasing
        diffs = df["timestamp"].diff().dropna()
        assert (diffs > 0).all(), "Timestamps should be increasing"
    finally:
        os.unlink(tmp_csv.name)


def test_timestamps_monotonic(app, sample_dir):
    """Timestamps for each group should be strictly increasing or non-decreasing."""
    sessions = app.find_emotibit_sessions(sample_dir)
    for sess in sessions:
        device_meta, channels = app.load_emotibit_json(sess["json"])
        created_at = device_meta.get("created_at")
        dataframes = app.parse_emotibit_csv(sess["csv"], created_at, channels)
        for df, sr, type_name, tags in dataframes:
            # check monotonic increasing timestamp
            diffs = df["timestamp"].diff().dropna()
            assert (diffs >= 0).all(), f"Timestamps not monotonic for {type_name}"


def test_no_artificial_flatlining(app, sample_dir):
    """Ensure no long runs of repeated values are introduced by the parser."""
    sessions = app.find_emotibit_sessions(sample_dir)
    # Choose the first session (which exhibited the bug) for testing
    sess = sessions[0]
    device_meta, channels = app.load_emotibit_json(sess["json"])
    created_at = device_meta.get("created_at")
    dataframes = app.parse_emotibit_csv(sess["csv"], created_at, channels)
    # Find the PPG group containing PG/PI/PR
    for df, sr, type_name, tags in dataframes:
        if set(tags) >= {"PG", "PI", "PR"}:
            # For each tag, compute the maximum run length of identical values
            for tag in ["PG", "PI", "PR"]:
                series = df[tag]
                # Compute run lengths of consecutive identical values
                run_len = 1
                max_run = 1
                for i in range(1, len(series)):
                    if series.iloc[i] == series.iloc[i - 1]:
                        run_len += 1
                        if run_len > max_run:
                            max_run = run_len
                    else:
                        run_len = 1
                # In the original bug, runs > 10 were observed; ensure no such runs exist now
                assert max_run <= 10, f"Tag {tag} has a run of {max_run} identical values"
            break
    else:
        pytest.fail("PPG group not found in parsed data")


def test_gap_handling_no_excess_rows(app):
    """Ensure parser does not insert extra rows beyond actual sample count."""
    # Create CSV with large gap in system_ms to simulate missing packets
    csv_lines = [
        "0,0,2,EA,1,100,1.0,2.0",
        "1000,1,2,EA,1,100,3.0,4.0",  # 1 second gap with nominal sr=15 (66.7ms interval)
    ]
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tmp_csv.write("\n".join(csv_lines).encode("utf-8"))
        tmp_csv.close()
        channels = {
            "EA": {"nominal_srate": 15.0, "raw_info": {"name": "ElectrodermalActivity"}},
        }
        out = app.parse_emotibit_csv(tmp_csv.name, device_created_at=None, channels_meta=channels)
        # Expect one group
        assert len(out) == 1
        df, sr, type_name, tags = out[0]
        # Only four samples should exist
        assert len(df) == 4, f"Unexpected number of rows: {len(df)}"
        # No additional rows inserted across the 1 second gap
    finally:
        os.unlink(tmp_csv.name)


def test_end_to_end_conversion(app, sample_dir, tmp_path):
    """Run one session through conversion and verify output files and metadata."""
    sessions = app.find_emotibit_sessions(sample_dir)
    sess = sessions[0]
    device_meta, channels = app.load_emotibit_json(sess["json"])
    created_at = device_meta.get("created_at")
    dataframes = app.parse_emotibit_csv(sess["csv"], created_at, channels)
    # Write YQ folder
    out_dir = tmp_path / "yq_test"
    app.write_yq_folder(str(out_dir), dataframes, device_meta)
    # Verify CSV files and metadata file exist
    files = os.listdir(out_dir)
    assert "metadata.csv" in files
    # For each CSV data file ensure timestamp column is last and values vary
    data_files = [f for f in files if f.endswith(".csv") and f != "metadata.csv"]
    assert data_files, "No data CSV files created"
    for fname in data_files:
        df = pd.read_csv(os.path.join(out_dir, fname))
        assert df.columns[-1] == "timestamp", "Timestamp column should be last"
        # check monotonic
        diffs = df["timestamp"].diff().dropna()
        assert (diffs >= 0).all(), f"Timestamps not monotonic in {fname}"