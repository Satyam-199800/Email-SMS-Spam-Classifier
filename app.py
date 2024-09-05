import streamlit as st
import pickle as pkl
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

# Define a directory for NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK resources
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)

# Instantiate the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the vectorizer and model
tfidf = pkl.load(open('vectorizer.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input field for the SMS text
input_sms = st.text_area("Enter the message")

# Button to trigger prediction
if st.button('Predict'):
    # Transform the input text
    transformed_sms = transform_text(input_sms)
    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the loaded model
    result = model.predict(vector_input)[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")