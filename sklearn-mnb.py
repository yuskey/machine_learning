import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
import unicodedata

"""Training Dataset is Youtube04-Eminem, which is spam dataset with 4 different features and 1 class
Features: COMMENT_ID, AUTHOR, DATE, CONTENT
Class: CLASS (0 or 1)
"""
class dataset(object):
    """This class will hold the text data and labels"""
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name, header = 0, encoding = 'utf-8')
        self.file_name = file_name
        self.y = self.data['CLASS']
        self.x = self.data['CONTENT']

    def print_data(self):
        self.data['CONTENT']=self.data['CONTENT'].str.encode('utf-8', 'ignore')
        print(self.data)


class model(object):
    """Class for the model, will utilize the dataset classes to perform classification tasks"""
    def __init__(self):
        self.clf = MultinomialNB()
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def train_mnb(self, train_dataset):
        self.x = self.vectorize(train_dataset.x)
        self.y = train_dataset.y
        self.clf.fit(self.x, self.y)

    def cross_validation(self):
        score = cross_val_score(self.clf, self.x, self.y, cv=5)
        print("Cross Validation scores: ")
        print(str(score)+'\n')

    def vectorize(self, x):
        """Creates a bag of words representation of the data in x
        Keeps track of occurrences of words in the documents"""
        return self.count_vect.fit_transform(x)

    def tf_idf_vectorize(self):
        """Creates a tf-idf representation of the bag of words model
        This takes into account word frequency across documents and document frequency"""
        self.tfidf_bag_of_words = self.tfidf_transformer.fit_transform(self.bag_of_words)

    def test_on_dataset(self, test_dataset):
        """Converts test dataset to bag of words representation using transform vocabulary, which was trained when vectorize() was run
        predicts based on text in test_dataset then determines accuracy based on test dataset labels"""
        test_x = self.count_vect.transform(test_dataset.x)
        predicted = self.clf.predict(test_x)
        accuracy = np.mean(predicted == test_dataset.y)
        print("Accuracy on test set " +str(test_dataset.file_name) + " is: "+ str(accuracy) + '\n')

    def predict(self, new_data):
        """Predicts labels for new text documents"""
        bow = self.count_vect.transform(new_data)
        predicted = self.clf.predict(bow)
        for text, pred in zip(new_data, predicted):
            print(str(text) + ", " + str(pred))
        print('\n')



def main():
    training = dataset("./youtube_spam/Youtube04-Eminem.csv")
    testing = dataset("./youtube_spam/Youtube05-Shakira.csv")

    classifier = model()
    classifier.train_mnb(training)
    classifier.cross_validation()
    classifier.test_on_dataset(testing)
    classifier.predict(['Eminem is bad', 'Check out my website'])


if __name__ == "__main__":
    main()
