import streamlit as st
import pickle as pkl
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import nltk

# Get the full path to the nltk_data directory
nltk_data_path = os.path.abspath(os.path.join(os.getcwd(), 'nltk_data'))
nltk.data.path.append(nltk_data_path)

# Print out the current NLTK data paths
print("NLTK data paths:", nltk.data.path)

# Check if the punkt tokenizer is accessible
punkt_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt')
print("Punkt tokenizer exists:", os.path.exists(punkt_path))
print("Contents of nltk_data/tokenizers/punkt:", os.listdir(punkt_path))


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


tfidf = pkl.load(open('vectorizer.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

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