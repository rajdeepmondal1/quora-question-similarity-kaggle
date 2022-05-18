import os
import pickle

import streamlit as st
import tensorflow as tf

from utils import predict_json

st.set_page_config(page_title='Are your questions similar?',
                   layout='wide',
                   initial_sidebar_state='auto')

# Setup environment credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "poised-beach-350613-8034dfebc3af.json"
PROJECT = "poised-beach-350613"
REGION = "us-central1"

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.header("Are your questions similar?")


def convert_text(txt, tokenizer, padder):
    x = tokenizer.texts_to_sequences(txt)
    x = padder(x, maxlen=40)
    return x


def make_prediction(qs1, qs2, model):
    """
    Takes 2 questions and outputs a similarity score.

    Returns:
     Similarity predictions of 2 input questions.
    """
    with open('tokenizer.pickle', 'rb') as handle:
        new_tokenizer = pickle.load(handle)

    a1 = convert_text([qs1], new_tokenizer, tf.keras.preprocessing.sequence.pad_sequences)
    a2 = convert_text([qs2], new_tokenizer, tf.keras.preprocessing.sequence.pad_sequences)

    data = [{
        'input_1': a1[0].tolist(),
        'input_2': a2[0].tolist(),
        'input_3': a1[0].tolist(),
        'input_4': a2[0].tolist(),
        'input_5': a1[0].tolist(),
        'input_6': a2[0].tolist()
    }]  # works
    predictions = predict_json(project=PROJECT,
                               region=REGION,
                               model=model,
                               instances=data)
    return predictions[0][0]


question_1 = st.text_input(label="Enter your first Question",
                           value="Can nuclear fusion power cause global warming?")
question_2 = st.text_input(label="Enter your second Question",
                           value="Is nuclear fusion power the reason for global warming?")

if question_1 and question_2:
    question_1 = question_1
    question_2 = question_2
    pred_button = st.button("Predict")
else:
    st.warning("Please Input both questions")
    st.stop()

# Did the user press the predict button?
if pred_button:
    pred_button = True

MODEL = 'quora_similarity_model'
if pred_button:
    prediction = make_prediction(question_1, question_2, MODEL)
    st.write(f"Your questions are {round(prediction * 100, 3)}% similar.")
