import streamlit as st

st.title("AcousticWise")
uploaded = st.file_uploader("Upload a WAV recording", type=["wav"])
st.button("Classify")
