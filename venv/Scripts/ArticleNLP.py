import os
import pickle
import streamlit as st
from scipy.sparse import hstack

# Load the saved vectorizers and model
file_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer1.pkl')
with open(file_path, 'rb') as file:
    vectorizer1 = pickle.load(file)
file_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer2.pkl')
with open(file_path, 'rb') as file:
    vectorizer2 = pickle.load(file)
file_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer3.pkl')
with open(file_path, 'rb') as file:
    vectorizer3 = pickle.load(file)
file_path = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl')
with open(file_path, 'rb') as file:
    model = pickle.load(file)
#'================'
print("Is fitted:", hasattr(vectorizer1, 'idf_'))
#'================'
# Streamlit app
st.title("Reliable Article check")

# Get user input for each text column
text_input_1 = st.text_input("Enter title of the news article")
text_input_2 = st.text_input("Enter author of the news article")
text_input_3 = st.text_input("Enter text/content of the news article")

if st.button("Predict"):
    # Transform each text input using the corresponding vectorizer
    input_tfidf_1 = vectorizer1.transform([text_input_1])
    input_tfidf_2 = vectorizer2.transform([text_input_2])
    input_tfidf_3 = vectorizer3.transform([text_input_3])

    # Combine the transformed features
    input_tfidf_combined = hstack([input_tfidf_1, input_tfidf_2, input_tfidf_3])

    # Make prediction using the Logistic Regression model
    prediction = model.predict(input_tfidf_combined)
    if prediction[0] == 1:
        st.write("This Article is unreliable")
    else:
        st.write("This Article is reliable")
