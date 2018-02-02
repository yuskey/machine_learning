import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

class dataset(object):
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name, header=0, encoding="utf-8")
        self.file_name = file_name
        self.x = self.data['CONTENT']
        self.y = self.data['CLASS']
        self.target_names = ["Spam", "Not Spam"]


class model(object):
    def __init__(self):
        self.clf = SVC()
        self.count_vect = CountVectorizer()
        self.pipe = Pipeline([('vect', CountVectorizer()),
                              ('idf', TfidfTransformer()), #can comment in and out to use idf
                              ('clf', SVC()),])  #kernels = linear, rbf, sigmoid, poly

        self.params = {"vect__ngram_range": [(1,1), (1,2), (1,3)],
                       "idf__use_idf":(True, False),
                       "clf__kernel":['linear','poly','rbf','sigmoid']}

    def grid_search(self, training_data):
        """
            This function will search for the optimum parameters for a model and set the pipe parameters to said optimal parameters
            Requires setting the self.params dictionary with valid parameters to check
        """
        gs_clf = GridSearchCV(self.pipe, self.params, scoring='roc_auc', n_jobs=-1)
        gs_clf = gs_clf.fit(training_data.x, training_data.y)
        print("Best score is: " + str(gs_clf.best_score_))
        for param_name in sorted(self.params.keys()):
            print("%s: %r"%(param_name, gs_clf.best_params_[param_name]))
        self.pipe.set_params(**gs_clf.best_params_) #sets pipe params to best grid search params, ** unpacks the best_params_ dict


    def train_pipe(self, training_data):
        """
            Trains all aspects of the pipeline on given training data
            This trains the Vectorizer, the TFIDF transformer, and the SVC classifier itself
            Requires a training dataset with data and matching labels
        """
        self.pipe.fit(training_data.x, training_data.y)
        #print(self.pipe.score(training_data.x[301:], training_data.y[301:]))

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

def main():
    training = dataset("Youtube04-Eminem.csv")
    testing = dataset("Youtube05-Shakira.csv")

    classifier = model()
    classifier.grid_search(training)
    classifier.cross_validation(training)
    classifier.train_pipe(training)
    classifier.test_pipe(testing)

if __name__ == "__main__":
    main()
