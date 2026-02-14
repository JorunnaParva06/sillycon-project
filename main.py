from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
TRAINING_SET = 'datasets/flirtingdatawithpunc_80_percent.csv'
TEST_SET = 'datasets/flirtingdatawithpunc_20_percent.csv'

def sentencePredictor(sentence, vectorizer, model):
    #Converts new sentence in a vector
    vectorizedSentence = vectorizer.transform([sentence])

    sparse_matrix = vectorizer.transform([sentence])
    # feature_names = vectorizer.get_feature_names_out() # list of all grams

    # Convert the sparse matrix to a dense matrix (i.e. a matrix of all grams extracted,associting each gram with a value. Non-zero values correspond to relevant grams)
    dense_matrix = sparse_matrix.todense()
    dense_matrix_list = dense_matrix.tolist()
    msg_sum = sum(dense_matrix_list[0])

     #Computes the probability that the sentence is flirtatious
    probability = model.predict_proba(msg_sum)[0, 1]
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
    vectorizer = TfidfVectorizer(lowercase = True, ngram_range = (1, 3), stop_words = "english")
    
    sparse_matrix = vectorizer.fit_transform(x_train)
    # feature_names = vectorizer.get_feature_names_out() # list of all grams

    # Convert the sparse matrix to a dense matrix (i.e. a matrix of all grams extracted,associting each gram with a value. Non-zero values correspond to relevant grams)
    dense_matrix = sparse_matrix.todense()
    dense_matrix_list = dense_matrix.tolist()
    # print(dense_matrix_list)  

    message_sums = []
    # Sum the ngram values for each message
    for message in dense_matrix_list:
        msg_sum = sum(message)
        message_sums.append([msg_sum])

    # Create and train the model    
    model = LogisticRegression()
    model.fit(message_sums, y_train)

    # repeat the above training steps but for the test data
    test_data = pd.read_csv(TEST_SET)
    x_test = extract_x(test_data)
    y_test = extract_y(test_data)

    test_sparse_mtrx = vectorizer.transform(x_test)
    test_dense_matrix = test_sparse_mtrx.todense()
    test_dense_mtrx_list = test_dense_matrix.tolist()
    test_message_sums = []

    for message in test_dense_mtrx_list:
        msg_sum = sum(message)
        test_message_sums.append([msg_sum])

    # Make predictions for the test data
    y_pred = model.predict(test_message_sums)
    # Determine how accurate our model is
    print("Accuracy:", accuracy_score(y_test, y_pred))

    sentence = input("Enter your sentence: ")
    print(f"There is a{sentencePredictor(sentence, vectorizer, model)*100}% chance that this message is flirtatious")

if __name__ == "__main__":
    main()
