import streamlit as st
import joblib

import re

def custom_preprocessing(text):
    """
    Custom text preprocessing for IMDB reviews
    Keeps alphabets and numbers
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load the trained model
model = joblib.load("sentiment_model.pkl")

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Title and description
st.title("🎬 IMDB Review Sentiment Analysis")
st.write("Enter a movie review to predict whether it is **Positive** or **Negative**.")

# Text input
review = st.text_area(
    "Movie Review",
    placeholder="Type or paste an IMDB movie review here...",
    height=150
)

# Prediction button
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = model.predict([review])[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
