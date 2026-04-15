import streamlit as st
import joblib
import pandas as pd
import re

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f4f6f7;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------

model = joblib.load("phishing_model.pkl")

# -------------------------------
# FEATURE EXTRACTION FUNCTION
# -------------------------------

def extract_features(url):
    return {
        "Have_IP": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        "Have_At": 1 if "@" in url else 0,
        "URL_Length": len(url),
        "URL_Depth": url.count('/'),
        "Redirection": 1 if '//' in url[8:] else 0,
        "https_Domain": 1 if "https" in url else 0,
        "TinyURL": 1 if any(short in url for short in ["bit.ly", "tinyurl"]) else 0,
        "Prefix/Suffix": 1 if "-" in url else 0
    }

# -------------------------------
# UI HEADER
# -------------------------------

st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🛡️ Phishing URL Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter a website URL to check if it is Legitimate or Phishing</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# USER INPUT
# -------------------------------

url_input = st.text_input("🔗 Enter Website URL")

if st.button("Check URL"):

    if url_input:

        features = extract_features(url_input)
        input_df = pd.DataFrame([features])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error("⚠️ This URL is PHISHING!")
        else:
            st.success("✅ This URL is LEGITIMATE!")

        st.info(f"Phishing Probability: {round(probability * 100, 2)}%")

    else:
        st.warning("Please enter a URL first.")

st.markdown("---")
st.caption("Developed using Machine Learning | Random Forest Classifier")
