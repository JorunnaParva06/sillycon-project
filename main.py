from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import streamlit as st
import time

DATA_SET = 'datasets/flirtingdatawithpunctuation.csv'

def sentencePredictor(sentence, vectorizer, model):
    #Convert the sentence to a vector
    vectorizedSentence = vectorizer.transform([sentence])
    #Predict the probability 
    probability = model.predict_proba(vectorizedSentence)[0, 1]
    return probability

def extract_x(data):
    x = []
    for i in range(len(data)):
        x.append(data.at[i, 'final_messages'])
    return x

def extract_y(data):
    y = []
    for i in range(len(data)):
        y.append(data.at[i, 'polarity'].item())
    return y
  
def main():
    data = pd.read_csv(DATA_SET)
    x_train = extract_x(data)
    y_train = extract_y(data)

    #Vectorize data
    vectorizer = TfidfVectorizer(lowercase = True, ngram_range = (1, 3), stop_words = "english")
    X_train = vectorizer.fit_transform(x_train)

    #Create and train the model    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # UI
    st.markdown("<h1 style='text-align: center; font-size: 5rem; color: #FF0000;'>Rate My Rizz</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 2rem; color: #FF0000;'>Enter a pickup line and check its Rizz Level.</h2>", unsafe_allow_html=True)

    # Define styles for bar
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, white, pink, red);
    }


    .stProgress > div > div > div {
        background-color: grey;
    }
    </style>
    """, unsafe_allow_html=True)

    progress_text = "Rizz Meter"
    my_bar = st.progress(0, text=progress_text)
   
    pickup_line = st.text_input("Enter your text", "")
    button = st.button("Check", type = "primary")
    percent = 0

    if button:  # Basically your onClick listener
        # Get rizz percentage
        percent = sentencePredictor(pickup_line, vectorizer, model)*200
        if (percent > 100):
            percent = 100
        elif(percent < 0):
            percent = 0
        
        output = f"Your Rizz Level is: {int(percent)}%"
        st.markdown(f"<h3 style='text-align: center; font-size: 2.5rem; color: #FF0000;'>Your Rizz Level is: {int(percent)}%</h3>", unsafe_allow_html=True)
        #st.title(f"Your Rizz Level is: {int(percent)}%")  # Show the rizz percentage in numeric format

    for percent_complete in range(int(percent)):
        time.sleep(0.015)
        my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.01)

if __name__ == "__main__":
    main()