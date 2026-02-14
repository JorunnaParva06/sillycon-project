from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase = True, ngram_range = (1, 3), stop_words = "english")

test_statements = [
    "Hello Super Mario I think you are very beautiful",
    "Hey Princess Peach I like your dress",
    "Hi I am Bowser and I believe flowers are super beautiful just like you",
    "Alrighty I am Bowser Junior and did you know that you are ugly",
    "You are very stinky gross"
]

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