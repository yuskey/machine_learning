import numpy as np
import pandas as pd

#sklearn imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib

#keras imports
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

"""Training Dataset is pima-indians-diabetes.csv
Attribute Information:
    1. Number of times pregnant
    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Diastolic blood pressure (mm Hg)
    4. Triceps skin fold thickness (mm)
    5. 2-Hour serum insulin (mu U/ml)
    6. Body mass index (weight in kg/(height in m)^2)
    7. Diabetes pedigree function
    8. Age (years)
    9. Class variable (0 or 1)
"""
class dataset(object):
    """This class will hold the data and labels"""
    def __init__(self, file_name):
        self.data = np.loadtxt(file_name, delimiter=',')
        self.file_name = file_name
        self.x = self.data[:,0:8]       #[:,0:8] designates all of the instances, columns 0-7
        self.y = self.data[:,8]         #[:8] designates all of the instances, column 8 only
        self.target_names = ['diabetes', 'no diabetes']

class model():
    def __init__(self):
        self.model = KerasClassifier(build_fn=self.create_nn, verbose=0)
        self.params= {'optimizer':['rmsprop', 'adam'],
                      'init':['glorot_uniform', 'normal', 'uniform'],
                      'epochs':[50, 100, 150],
                      'batch_size':[5, 10, 20]}


    def create_nn(self,optimizer='rmsprop', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
        model.add(Dense(8, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        #compile model
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return model

    def grid_search(self, train_dataset):
        gs_clf = GridSearchCV(estimator=self.model, param_grid=self.params)
        gs_clf = gs_clf.fit(train_dataset.x, train_dataset.y)
        # summarize results
        print("Best: %f using %s" % (gs_clf.best_score_, gs_clf.best_params_))
        means = gs_clf.cv_results_['mean_test_score']
        stds = gs_clf.cv_results_['std_test_score']
        params = gs_clf.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        self.save_model(gs_clf, 'keras_model.pkl')
        self.model.set_params(**gs_clf.best_params_)

    def cross_validation(self, train_dataset):
        predicted = cross_val_predict(self.model, train_dataset.x, train_dataset.y, cv=10)
        print(metrics.classification_report(train_dataset.y, predicted, target_names=train_dataset.target_names))
        fpr,tpr,thresholds = metrics.roc_curve(train_dataset.y, predicted)
        print("AUC is: " + str(metrics.auc(fpr,tpr)) + "\n")

    def save_model(self, gs_clf, file_name):
        joblib.dump(gs_clf.best_estimator_, file_name)

    def load_model_params(self, file_name):
        self.model.set_params(**joblib.load(file_name).best_params_)

def main():
    training = dataset("pima-indians-diabetes.csv")
    classifier = model()
    classifier.cross_validation(training)
    classifier.grid_search(training)


if __name__ == "__main__":
    main()
