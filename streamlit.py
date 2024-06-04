import warnings
warnings.filterwarnings('ignore')

import re
import pandas as pd
import numpy as np
import streamlit as st

from tensorflow import keras
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

CLASSES = {
    0: {
        "name": "NEGATIVE",
        "color": "red"
    },
    1:{
        "name": "SOMEWHAT NEGATIVE",
        "color": "orange"
    },
    2: {
        "name": "NEUTRAL",
        "color": "gray"
    },
    3: {
        "name": "SOMEWHAT POSITIVE",
        "color": "blue"
    },
    4: {
        "name": "POSITIVE",
        "color": "green"
    }
}
MAX_TOKENS = 13743
OUTPUT_SEQUENCE_LENGTH = 48

model = keras.models.load_model("./model.keras")

vectorize_layer = keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    pad_to_max_tokens=True,
    output_mode='int',
    output_sequence_length=OUTPUT_SEQUENCE_LENGTH)

X_train_s = pd.read_csv("./data/streamlit.csv")
X_train_s = X_train_s["Text"].values
vectorize_layer.adapt(np.array(X_train_s, dtype=str))

st.title("Movie Review Sentiment Analysis")
text_input = st.text_area("Enter a review", key="text_input")

if text_input != "":
    text_input = re.sub("[^a-zA-Z]"," ", text_input)
    input_words = word_tokenize(text_input.lower())
    lemma_words = [lemmatizer.lemmatize(i) for i in input_words]

    test_input = vectorize_layer([' '.join(lemma_words)])
    test_input = np.array(test_input)

    prediction_dist = model.predict(test_input)
    prediction = np.argmax(prediction_dist)
    st.write(f"This review is :{CLASSES[prediction]['color']}[{CLASSES[prediction]['name']}]")

template = pd.DataFrame({"Text": ["This is a great movie", "This is a bad movie", "This is a movie"]})

st.header("Upload .csv file to analyse: ")
st.download_button("Download Template", 
                   data=template.to_csv(index=False).encode('utf-8'), 
                   file_name="template.csv", 
                   mime="application/octet-stream")

uploaded_file = st.file_uploader("Attach .csv file here", type=["csv"])

if uploaded_file is not None:
    st.write(f":green[File uploaded successfully!]")
    st.write("Analyzing the file...")
    df = pd.read_csv(uploaded_file)

    reviews = []
    for sentence in df['Text']:
        if type(sentence) is not str:
            sentence = str(sentence)
        
        review_text = re.sub("[^a-zA-Z]"," ", sentence)
        words = word_tokenize(review_text.lower())
        lemma_words = [lemmatizer.lemmatize(i) for i in words]
    
        reviews.append(' '.join(lemma_words))

    test_input = vectorize_layer(reviews)
    test_input = np.array(test_input)

    predictions = model.predict(test_input)
    predictions = np.argmax(predictions, axis=1)
    df["Sentiment"] = predictions
    df["Sentiment"] = df["Sentiment"].apply(lambda x: CLASSES[x]["name"])
    
    st.write("Analysis completed! This is the result:")
    st.write(df)

