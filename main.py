from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
FILENAME = 'datasets/flirtingdatawithpunc_80_percent.csv'

def extract_x(data):
    x = []
    for i in range(len(data)):
        x.append(data.iloc[i][1])
    return x
def extract_y(data):
    y = []
    for i in range(len(data)):
        y.append(data.iloc[i][0])
    return y
  
def main():
    data = pd.read_csv(FILENAME)
    data.to_string()
    x_train = extract_x(data)
    y_train = extract_y(data)

    vectorizer = TfidfVectorizer(lowercase = True, ngram_range = (1, 3), stop_words = "english")

    # test_statements = [
    #     "Hello Super Mario I think you are very beautiful",
    #     "Hey Princess Peach I like your dress",
    #     "Hi I am Bowser and I believe flowers are super beautiful just like you",
    #     "Alrighty I am Bowser Junior and did you know that you are ugly",
    #     "You are very stinky gross"
    # ]

    sparse_matrix = vectorizer.fit_transform(test_statements)
    feature_names = vectorizer.get_feature_names_out()

    # Convert the sparse matrix to a dense matrix
    dense_matrix = sparse_matrix.todense()
    dense_matrix_list = dense_matrix.tolist()
    print(dense_matrix_list)  # Non-zero values correspond to relevant words

    all_keywords = []

    # Extracting the ngrams from each sentence
    for text in dense_matrix_list:
        index = 0
        keywords = []
        for word in text:
            if word > 0:  # If it's an actual word that matters, keep track of it
                keywords.append(feature_names[index])
            index += 1
        all_keywords.append(keywords)
    print(all_keywords)  # Only showing relevant ngrams, so many ommited (they have a TFIDF value of 0.0)
    
    model = LogisticRegression()
    model.fit(x_train,y_train)
if __name__ == "__main__":
    main()
