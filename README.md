🎬 IMDB Sentiment Analysis Web App

🔗 Live App:
👉 https://imdb-sentiment-analysis-jy6mtanbwvnhgespqdkpjk.streamlit.app/

📂 Source Code:
👉 https://github.com/Sugnik27/imdb-sentiment-analysis

📌 Project Overview

This project is an end-to-end NLP and Machine Learning application that predicts whether an IMDB movie review expresses a Positive or Negative sentiment.

The model is deployed as a live web application using Streamlit Cloud, allowing users to input movie reviews and receive real-time sentiment predictions along with confidence scores.

The goal of this project is not just sentiment prediction, but to demonstrate production-ready ML practices, including preprocessing, model serialization, UI development, and cloud deployment.

🚀 Live Demo

The application is hosted on Streamlit Cloud and is accessible 24/7:

👉 https://imdb-sentiment-analysis.streamlit.app

Users can:

Enter custom movie reviews

Try multiple predefined positive and negative examples

View sentiment predictions with confidence scores

Interact with a clean, user-friendly UI

🧠 Problem Statement

Manually reading and analyzing thousands of user reviews is not scalable for platforms like IMDB, Netflix, or Amazon.

This application automates sentiment analysis at scale, enabling:

Faster insight into audience perception

Automated feedback analysis

Data-driven decision making

🛠️ Tech Stack

Programming Language: Python

NLP: TF-IDF Vectorization

Machine Learning: Scikit-learn (classification models)

Model Serialization: Joblib

Web Framework: Streamlit

Deployment: Streamlit Cloud

Version Control: Git & Git LFS

🔍 Key Features

Custom text preprocessing (lowercasing, HTML removal, alphanumeric filtering)

Trained ML pipeline using TF-IDF and classification models

Interactive Streamlit UI with:

Real-time predictions

Prediction confidence score

Random positive/negative example reviews

Cloud deployment independent of local machine

Git LFS used for managing large model files

📂 Project Structure
imdb-sentiment-analysis/
│
├── app.py                  # Streamlit application
├── sentiment_model.pkl     # Trained ML model (Git LFS)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

📈 Model Workflow

User inputs a movie review

Custom preprocessing cleans and normalizes text

TF-IDF vectorization converts text to numerical features

ML classifier predicts sentiment

Prediction and confidence score are displayed in the UI

💡 Why This Project Matters

While sentiment analysis is a known problem, this project demonstrates:

Correct handling of custom preprocessing in serialized models

Avoidance of data leakage

Understanding of model deployment constraints

Real-world ML engineering workflow

Ability to explain ML solutions in business terms
