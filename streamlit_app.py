# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import easyocr
import re
from PIL import Image
import plotly.graph_objects as go

# Load model and OCR
model = pickle.load(open('parkinson_model.pkl', 'rb'))
reader = easyocr.Reader(['en'], gpu=False)

# Set Streamlit config
st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Detection from Voice Parameters")
st.markdown("Choose an input method and analyze the risk of Parkinson's disease.")

# Initialize
values = {}
text = ""

# Selection Mode
upload_mode = st.radio("Select Input Method", ["Image Upload (OCR)", "CSV Upload", "Text Box Input"])

# OCR Mode
if upload_mode == "Image Upload (OCR)":
    uploaded = st.file_uploader("📷 Upload Report Image", type=['png', 'jpg', 'jpeg'])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        results = reader.readtext(np.array(img), detail=0)
        text = ' '.join(results)

# Text Box Input
elif upload_mode == "Text Box Input":
    text = st.text_area("📝 Paste text from the report:")
    
# CSV Upload
elif upload_mode == "CSV Upload":
    csv_file = st.file_uploader("📄 Upload CSV with feature columns", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.dataframe(df.head())
        if df.shape[1] >= 6:
            latest = df.iloc[-1]
            values = {
                'fo': latest.get('fo', 0.0),
                'fhi': latest.get('fhi', 0.0),
                'flo': latest.get('flo', 0.0),
                'jitter_percent': latest.get('jitter_percent', 0.0),
                'rap': latest.get('rap', 0.0),
                'ppe': latest.get('ppe', 0.0)
            }

# Pattern Matching
def find(pattern):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

if upload_mode in ["Image Upload (OCR)", "Text Box Input"] and text:
    patterns = {
        'fo': r'Fo\(Hz\)[^\d]*([\d.]+)',
        'fhi': r'Fhi\(Hz\)[^\d]*([\d.]+)',
        'flo': r'Flo\(Hz\)[^\d]*([\d.]+)',
        'jitter_percent': r'Jitter\(.*?%\)[^\d]*([\d.]+)',
        'rap': r'RAP[^\d]*([\d.]+)',
        'ppe': r'PPE[^\d]*([\d.]+)'
    }
    values = {k: find(p) for k, p in patterns.items()}
    st.info("📄 Extracted OCR/Text Features:")
    for k, v in values.items():
        st.write(f"{k.upper()}: {v}")

# Input Form (editable)
fo = st.number_input("Fo", value=values.get('fo', 0.0), step=0.1)
fhi = st.number_input("Fhi", value=values.get('fhi', 0.0), step=0.1)
flo = st.number_input("Flo", value=values.get('flo', 0.0), step=0.1)
jitter = st.number_input("Jitter(%)", value=values.get('jitter_percent', 0.0), step=0.001)
rap = st.number_input("RAP", value=values.get('rap', 0.0), step=0.001)
ppe = st.number_input("PPE", value=values.get('ppe', 0.0), step=0.001)

features = np.array([[fo, fhi, flo, jitter, rap, ppe]])

if st.button("🔍 Predict"):
    prediction = model.predict(features)[0]
    st.success("✅ Parkinson's Unlikely" if prediction == 0 else "⚠️ Parkinson's Likely!")

    # Visualization
    labels = ['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']
    raw_values = [fo, fhi, flo, jitter, rap, ppe]
    scaled = [v / m for v, m in zip(raw_values, [300, 400, 300, 1.0, 0.2, 1.0])]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=scaled, theta=labels, fill='toself', name="Features"))
    fig_radar.update_layout(title="🧭 Feature Radar Chart", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig_radar)

    fig_bar = go.Figure([go.Bar(x=labels, y=raw_values)])
    fig_bar.update_layout(title="📊 Raw Feature Values")
    st.plotly_chart(fig_bar)
