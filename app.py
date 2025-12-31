import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from datetime import datetime

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ================================
# DOWNLOAD STOPWORDS (SAFE)
# ================================
@st.cache_data
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = load_stopwords()

# ================================
# LOAD MODEL & VECTORIZER
# ================================
@st.cache_resource
def load_models():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# ================================
# TEXT CLEANING FUNCTION
# ================================
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ================================
# UI HEADER
# ================================
st.markdown("""
# 📧 Spam Email / SMS Detection
### 🚀 AI-powered Message Classifier
Detect whether a message is **Spam** or **Not Spam** using Machine Learning.
""")

st.divider()

# ================================
# USER INPUT
# ================================
user_input = st.text_area(
    "✍️ Enter your Email or SMS message:",
    height=150,
    placeholder="Congratulations! You've won a free ticket..."
)

# ================================
# PREDICTION BUTTON
# ================================
if st.button("🔍 Analyze Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 **SPAM ALERT!**")
            st.progress(int(prob[1] * 100))
            st.write(f"**Spam Confidence:** {prob[1]*100:.2f}%")
        else:
            st.success("✅ **This message is SAFE (Not Spam)**")
            st.progress(int(prob[0] * 100))
            st.write(f"**Safety Confidence:** {prob[0]*100:.2f}%")

# ================================
# SIDEBAR
# ================================
st.sidebar.title("ℹ️ App Info")
st.sidebar.markdown("""
**Model:** Logistic Regression  
**Vectorizer:** TF-IDF  
**Dataset:** SMS Spam Collection  
**Accuracy:** ~97%
""")

st.sidebar.markdown("---")
st.sidebar.write("📅 Date:", datetime.now().strftime("%d %B %Y"))
st.sidebar.write("👨‍💻 Developed by: *Your Name*")

# ================================
# FOOTER
# ================================
st.markdown("""
---
🔒 *This app does not store any messages.*  
🤖 *Built with Streamlit & Machine Learning.*
""")

