from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

TRAINING_SET = 'datasets/flirtingdatawithpunc_80_percent.csv'
TEST_SET = 'datasets/flirtingdatawithpunc_20_percent.csv'

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
    data = pd.read_csv(TRAINING_SET)
    x_train = extract_x(data)
    y_train = extract_y(data)

    #Vectorize data
    vectorizer = TfidfVectorizer(lowercase = True, ngram_range = (1, 3), stop_words = "english")
    X_train = vectorizer.fit_transform(x_train)

    #Create and train the model    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #Test the data
    test_data = pd.read_csv(TEST_SET)
    x_test = extract_x(test_data)
    y_test = extract_y(test_data)
    X_test = vectorizer.transform(x_test)

    #Make predictions for the test data
    y_pred = model.predict(X_test)

    #Determine how accurate our model is
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    value = sentencePredictor(sentence, vectorizer, model)*200
    if (value > 100):
        value = 100
    elif(value < 0):
        value = 0
    
        

if __name__ == "__main__":
    main()
