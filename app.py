import streamlit as st
import pickle as pkl
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Instantiate the PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    # Convert text to lowercase
    text = text.lower()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords and i not in string.punctuation:
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