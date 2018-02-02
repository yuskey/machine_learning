import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

"""Training Dataset is Youtube04-Eminem, which is spam dataset with 4 different features and 1 class
Features: COMMENT_ID, AUTHOR, DATE, CONTENT
Class: CLASS (0 or 1)
"""
class dataset(object):
    """This class will hold the data and labels"""
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name, header = 0, encoding = 'utf-8')
        self.file_name = file_name
        self.y = self.data['CLASS']
        self.x = self.data['CONTENT']
        self.target_names = ['spam', 'not spam']

    def print_data(self):
        self.data['CONTENT']=self.data['CONTENT'].str.encode('utf-8', 'ignore')
        print(self.data)


class model(object):
    """Class for the model, will utilize the dataset classes to perform classification tasks"""
    def __init__(self):
        self.clf = MultinomialNB()
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()
        self.pipe = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()), #can comment in and out to use idf
                              ('clf', MultinomialNB()),])
        #format for params is pipe name __ parameter, for example vect__ngram_range is for the
        #count_vectorizer() parameter called ngram_range
        self.params = {'vect__ngram_range': [(1,1), (1,2), (1,3)],
                       'tfidf__use_idf': (True, False),
                       'clf__alpha': (1e-2,1e-3)}

    def parameter_search(self,train_dataset):
        """
            This function will search for the optimum parameters for a model and set the pipe parameters to said optimal parameters
            Requires setting the self.params dictionary with valid parameters to check
        """
        #n_jobs allows for running parameter combinations in parallel if there are multicores: -1 tells it to search for how many cores
        gs_clf = GridSearchCV(self.pipe, self.params, scoring='roc_auc', n_jobs=-1)
        gs_clf = gs_clf.fit(train_dataset.x, train_dataset.y)
        print("Best score is: " + str(gs_clf.best_score_))
        for param_name in sorted(self.params.keys()):
            print("%s : %r"% (param_name, gs_clf.best_params_[param_name]))
        self.pipe.set_params(**gs_clf.best_params_) #sets pipe params to best grid search params, ** unpacks the best_params_ dict

    def train_pipe(self, train_dataset):
        """
            Trains all aspects of the pipeline on given training data
            This trains the Vectorizer, the TFIDF transformer, and the MNB classifier itself
            Requires a training dataset with data and matching labels
        """
        self.pipe.fit(train_dataset.x, train_dataset.y)

    def test_pipe(self, test_dataset):
        """
            This function will predict the labels on the test dataset and then report the metrics for prediction
            Requires a test_dataset with data and labels
        """
        predicted = self.pipe.predict(test_dataset.x)
        accuracy = np.mean(predicted == test_dataset.y)
        print("Scores for testing on %s"%test_dataset.file_name)
        print(metrics.classification_report(test_dataset.y, predicted, target_names=test_dataset.target_names))
        fpr,tpr,thresholds = metrics.roc_curve(test_dataset.y, predicted)
        print("AUC is: " + str(metrics.auc(fpr,tpr)) + "\n")


    def cross_validation(self, train_dataset):
        """
            Will perform 10 fold cross-validation on the training dataset and report metrics
        """
        predicted = cross_val_predict(self.pipe, train_dataset.x, train_dataset.y, cv=10)
        score = cross_val_score(self.pipe, train_dataset.x, train_dataset.y, cv=5)
        #print("Cross Validation scores: ")
        #print(str(score)+'\n')
        print("Scores for cross validation on %s"%train_dataset.file_name)
        print(metrics.classification_report(train_dataset.y, predicted, target_names=train_dataset.target_names))
        fpr,tpr,thresholds = metrics.roc_curve(train_dataset.y, predicted)
        print("AUC is: " + str(metrics.auc(fpr,tpr)) + "\n")


    """ Outdated, you can just use the pipeline to train and test and vectorize
    def train_mnb(self, train_dataset):
        Trains the model and the volcabulary for the Vectorizer
        self.x = self.vectorize(train_dataset.x)
        self.y = train_dataset.y
        self.clf.fit(self.x, self.y)

    def vectorize(self, x):
        Creates a bag of words representation of the data in x
        Keeps track of occurrences of words in the documents
        return self.count_vect.fit_transform(x)

    def tf_idf_vectorize(self):
        Creates a tf-idf representation of the bag of words model
        This takes into account word frequency across documents and document frequency
        self.tfidf_bag_of_words = self.tfidf_transformer.fit_transform(self.bag_of_words)

    def test_on_dataset(self, test_dataset):
        Converts test dataset to bag of words representation using transform vocabulary, which was trained when vectorize() was run
        predicts based on text in test_dataset then determines accuracy based on test dataset labels
        test_x = self.count_vect.transform(test_dataset.x)
        predicted = self.clf.predict(test_x)
        accuracy = np.mean(predicted == test_dataset.y)
        print("Accuracy on test set " +str(test_dataset.file_name) + " is: "+ str(accuracy) + '\n')

    def predict(self, new_data):
        Predicts labels for new text documents
        bow = self.count_vect.transform(new_data)
        predicted = self.clf.predict(bow)
        for text, pred in zip(new_data, predicted):
            print(str(text) + ", " + str(pred))
        print('\n')
    """


def main():
    training = dataset("Youtube04-Eminem.csv")
    testing = dataset("Youtube05-Shakira.csv")
    classifier = model()
    classifier.parameter_search(training)
    classifier.cross_validation(training)
    classifier.train_pipe(training)
    classifier.test_pipe(testing)



if __name__ == "__main__":
    main()
