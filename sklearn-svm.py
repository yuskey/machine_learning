import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

class dataset(object):
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name, header=0, encoding="utf-8")
        self.file_name = file_name
        self.x = self.data['CONTENT']
        self.y = self.data['CLASS']


class model(object):
    def __init__(self):
        self.clf = SVC()
        self.count_vect = CountVectorizer()
        self.pipe = Pipeline([('vect', CountVectorizer()),
                              ('idf', TfidfTransformer()),
                              ('clf', SVC(kernel='linear')),])  #kernels = linear, rbf, sigmoid, poly

    def train_pipe(self, training_data):
        self.pipe.fit(training_data.x, training_data.y)
        #print(self.pipe.score(training_data.x[301:], training_data.y[301:]))

    def test_pipe(self, testing_data):
        print(self.pipe.score(testing_data.x, testing_data.y))

def main():
    training = dataset("./youtube_spam/Youtube04-Eminem.csv")
    testing = dataset("./youtube_spam/Youtube05-Shakira.csv")

    classifier = model()
    classifier.train_pipe(training)
    classifier.test_pipe(testing)

if __name__ == "__main__":
    main()
