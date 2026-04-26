import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np

st.set_page_config(
    page_title="Phishing URL Detection System",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        font-size: 15px;
        border-radius: 6px;
        height: 2.8em;
        width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("phishing_model.pkl")

model = load_model()

def calculate_entropy(url):
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def extract_features(url):
    return {
        "Have_IP": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        "Have_At": 1 if "@" in url else 0,
        "URL_Length": len(url),
        "URL_Depth": len([i for i in url.split('/') if i]),
        "Redirection": 1 if '//' in url[8:] else 0,
        "https_Domain": 1 if url.startswith("https") else 0,
        "TinyURL": 1 if any(short in url for short in ["bit.ly", "tinyurl"]) else 0,
        "Prefix_Suffix": 1 if "-" in url else 0,
        "Digit_Count": sum(c.isdigit() for c in url),
        "Letter_Count": sum(c.isalpha() for c in url),
        "Special_Char_Count": len(re.findall(r'[^a-zA-Z0-9]', url)),
        "Entropy": calculate_entropy(url)
    }

st.title("Phishing URL Detection System")

url_input = st.text_input("Enter Website URL")

if st.button("Analyze URL"):

    if url_input:

        if not re.match(r'https?://', url_input):
            st.warning("Enter a valid URL starting with http:// or https://")
            st.stop()

        features = extract_features(url_input)
        input_df = pd.DataFrame([features])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("PHISHING")
            else:
                st.success("LEGITIMATE")

        with col2:
            st.metric("Probability", f"{round(probability * 100, 2)}%")

        if probability > 0.8:
            st.error("Risk Level: High")
        elif probability > 0.5:
            st.warning("Risk Level: Moderate")
        else:
            st.success("Risk Level: Low")

        with st.expander("Feature Details"):
            st.dataframe(input_df)

    else:
        st.warning("Please enter a URL.")
