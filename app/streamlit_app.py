import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from acousticwise.inference import InferenceError, InputError, Predictor

FEW_VESSELS = {"rescue", "dredger", "pilotvessel"}
PLOT_POINTS = 2000  # about a chart's width in pixels


@st.cache_resource
def load_predictor(run_dir: str) -> Predictor:
    """Load the run once per run folder, not on every rerun."""
    return Predictor.load(Path(run_dir))


run_dir = os.environ.get("ACOUSTICWISE_RUN_DIR")
if run_dir is None:
    st.error("No model run is set. Start the app with `uv run acousticwise-app`.")
    st.stop()
try:
    predictor = load_predictor(run_dir)
except FileNotFoundError:
    st.error(f"No trained model found in {run_dir}. Train one, or set ACOUSTICWISE_RUN_DIR.")
    st.stop()

st.title("AcousticWise")
uploaded = st.file_uploader("Upload a WAV recording", type=["wav"])
if st.button("Classify") and uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)
    try:
        prediction = predictor.predict(tmp_path)
    except InputError as err:
        st.error(str(err))
    except InferenceError:
        st.error("The model could not classify this file.")
    else:
        st.subheader(prediction.label)
        for label, score in prediction.scores.items():
            st.write(f"{label}: {score:.0%} model confidence")
        st.caption(
            "The model was trained on recordings from one hydrophone network; "
            "recordings from other places may score differently."
        )
        if prediction.label in FEW_VESSELS:
            st.warning(
                "This class had few vessels in training, so the result may reflect "
                "one ship rather than the vessel type."
            )

        step = max(1, len(prediction.waveform) // PLOT_POINTS)
        st.line_chart(prediction.waveform[::step], x_label="Time", y_label="Amplitude")

        mel = prediction.mel
        scaled = (mel - mel.min()) / (mel.max() - mel.min())
        st.image(np.flipud(scaled), caption="Log-mel spectrogram", width="stretch")
    finally:
        tmp_path.unlink()
