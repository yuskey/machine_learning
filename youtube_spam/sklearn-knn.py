import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

class model():
    def __init__(self):
        self.clf = KNeighborsClassifier()
        self.count_vect = CountVectorizer()
        self.tfidf = TfidfTransformer()
        self.pipe = Pipeline([('vect', self.count_vect),
                              ('tfidf', self.tfidf),
                              ('clf', self.clf),])

        self.params = {'vect__ngram_range':[(1,1),(1,2),(1,3)],
                       'tfidf__use_idf': (True, False),
                       'clf__n_neighbors':(2,5,10,15,20),
                       'clf__weights': ['uniform','distance'],
                       'clf__algorithm':['brute']}#with sparse inputs has to use brute force

    def grid_search(self, train_dataset):
        gs_clf = GridSearchCV(self.pipe, self.params, scoring='roc_auc', n_jobs=-1)
        gs_clf = gs_clf.fit(train_dataset.x, train_dataset.y)
        print("Best Score is: "  + str(gs_clf.best_score_))
        for param_name in sorted(self.params.keys()):
            print("%s:%r"%(param_name, gs_clf.best_params_[param_name]))
        self.pipe.set_params(**gs_clf.best_params_)

    def train_pipe(self,train_dataset):
        self.pipe.fit(train_dataset.x, train_dataset.y)

    def test_pipe(self, test_dataset):
        predicted = self.pipe.predict(test_dataset.x)
        accuracy = np.mean(predicted == test_dataset.y)
        print("Accuracy on %s is %r"%(test_dataset.file_name, accuracy))
        print("Scores for testing on %s:"%(test_dataset.file_name))
        print(metrics.classification_report(test_dataset.y, predicted, target_names=test_dataset.target_names))
        fpr,tpr,thresholds = metrics.roc_curve(test_dataset.y, predicted)
        print("AUC is: " + str(metrics.auc(fpr,tpr)) + "\n")

    def cross_validation(self, train_dataset):
        predicted = cross_val_predict(self.pipe, train_dataset.x, train_dataset.y, cv=10)
        #score = cross_val_score(self.pipe, train_dataset.x, train_dataset.y, cv=10)
        #print("Cross Validation scores: ")
        #print(str(score)+'\n')
        print("Scores for 10-fold Cross-validation:")
        print(metrics.classification_report(train_dataset.y, predicted, target_names=train_dataset.target_names))
        fpr,tpr,thresholds=metrics.roc_curve(train_dataset.y, predicted)
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
