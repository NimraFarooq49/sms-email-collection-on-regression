import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

st.title("📧 Spam Email Detection App")
st.write("Logistic Regression Model")

user_input = st.text_area("Enter Email / SMS Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT Spam")
