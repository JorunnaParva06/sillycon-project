# from sklearn.model_selection import *
# from sklearn.tree import *
# from sklearn.metrics import *
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

    model = LogisticRegression()
    model.fit(x_train,y_train)
if __name__ == "__main__":
    main()