import streamlit as st
import joblib
import re
import random

# =========================================================
# PAGE CONFIGURATION (FIRST STREAMLIT CALL)
# =========================================================
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# =========================================================
# CUSTOM TEXT PREPROCESSING
# (MUST EXIST BEFORE MODEL LOADING)
# =========================================================
def custom_preprocessing(text):
    """
    Custom text preprocessing for IMDB reviews
    Keeps alphabets and numbers
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)      # remove HTML breaks
    text = re.sub(r"[^a-z0-9\s]", "", text)     # keep letters + numbers
    text = re.sub(r"\s+", " ", text).strip()    # normalize spaces
    return text

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

# =========================================================
# SESSION STATE INIT
# =========================================================
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# =========================================================
# SAMPLE REVIEWS
# =========================================================
positive_reviews = [
    "An absolutely brilliant movie with stunning performances and a powerful story.",
    "The film was engaging from start to finish with excellent direction and music.",
    "Outstanding acting and a well-written script made this movie a joy to watch.",
    "A visually beautiful movie with strong emotions and memorable characters.",
    "One of the best movies I’ve seen this year, truly entertaining and meaningful."
]

negative_reviews = [
    "The movie was boring, predictable, and far too long.",
    "Poor acting and a weak storyline completely ruined the experience.",
    "The plot made no sense and the characters were badly written.",
    "A disappointing film with terrible pacing and forgettable scenes.",
    "I struggled to finish this movie due to its dull script and bad performances."
]

# =========================================================
# UI LAYOUT
# =========================================================
st.title("🎬 IMDB Review Sentiment Analyzer")
st.caption("Analyze movie reviews using Machine Learning & NLP")
st.divider()

# Text input bound to session state
review = st.text_area(
    "Enter a movie review",
    value=st.session_state.review_text,
    placeholder="Example: The movie had great acting but a weak storyline...",
    height=180
)

st.caption("✔ Supports long reviews • ✔ Numbers & punctuation handled")

# Example buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🎉 Try Positive Example"):
        st.session_state.review_text = random.choice(positive_reviews)

with col2:
    if st.button("💔 Try Negative Example"):
        st.session_state.review_text = random.choice(negative_reviews)

# =========================================================
# PREDICTION
# =========================================================
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        with st.spinner("Analyzing review..."):
            cleaned_review = custom_preprocessing(review)
            prediction = model.predict([cleaned_review])[0]

            if hasattr(model, "predict_proba"):
                confidence = max(model.predict_proba([cleaned_review])[0]) * 100
            else:
                confidence = None

        st.divider()

        if prediction == 1:
            st.success("✅ Positive Review")
            st.caption("The model predicts an overall positive sentiment.")
            st.progress(90)
        else:
            st.error("❌ Negative Review")
            st.caption("The model predicts an overall negative sentiment.")
            st.progress(40)

        if confidence is not None:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📌 About This App")
st.sidebar.write("""
- Trained on IMDB movie reviews  
- TF-IDF + Machine Learning  
- Custom NLP preprocessing  
- Deployed using Streamlit Cloud  
""")

st.sidebar.markdown(
    "🔗 **GitHub Repository:**  \n"
    "[View Source Code](https://github.com/Sugnik27/imdb-sentiment-analysis)"
)

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("Built by Sugnik Mondal | NLP & Machine Learning Project")
