import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
import unicodedata

"""Dataset is Youtube04-Eminem, which is spam dataset with 4 different features and 1 class
Features: COMMENT_ID, AUTHOR, DATE, CONTENT
Class: CLASS (0 or 1)

"""
class model(object):
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name, header=0, encoding='utf-8')
        self.headers = list(self.data.columns.values)
        self.y = self.data['CLASS']
        self.x = self.data['CONTENT']
        self.clf = MultinomialNB()

    def print_data(self):
        self.data['CONTENT']=self.data['CONTENT'].str.encode('utf-8', 'ignore')
        print(self.data)

    def vectorize(self):
        self.count_vect = CountVectorizer()
        self.bag_of_words = self.count_vect.fit_transform(self.x)

    def tf_idf_vectorize(self):
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_bag_of_words = self.tfidf_transformer.fit_transform(self.bag_of_words)
        #print(self.tfidf_bag_of_words.shape)

    def train_mnb(self):
        self.clf.fit(self.bag_of_words[:300],self.y[:300])
        print(self.clf.score(self.bag_of_words[301:], self.y[301:]))

    def predict(self, new_data):
        new_data_bow = self.count_vect.transform(new_data)
        new_data_tfidf = self.tfidf_transformer.transform(new_data_bow)
        predicted = self.clf.predict(new_data_bow)
        print(predicted)

    def cross_validation(self):
        score = cross_val_score(self.clf, self.bag_of_words, self.y, cv=5)
        print(score)


def main():
    file_name ="Youtube04-Eminem.csv"
    classifier = model(file_name)
    #classifier.print_data()
    classifier.vectorize()
    classifier.tf_idf_vectorize()
    classifier.train_mnb()
    classifier.predict(['Eminem is bad', 'Check out my website'])
    classifier.cross_validation()

if __name__ == "__main__":
    main()
