import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import warnings

"""
    To Do:
        1. Build the remaining models and try different parameters, could use gridCV but 
           not many parameters to try so will just stick with manual tuning

        2. Try a convolutional neural network
            2.1. Need to convert data into a 28x28x1 format for ConvNet 
            2.2. Build keras ConvNet model
            2.3. Figure out how to import keras model into sklearn, try
                 from keras.wrappers.scikit_learn import KerasClassifier
        

"""

#data preparation
def load_data(training_file, testing_file):
    train = pd.read_csv(training_file, header = 0)
    test = pd.read_csv(testing_file, header = 0)
    y_train = train['label']
    
    #check class balance in training set
    #print(train['label'].value_counts())

    #create copies to not alter the originals
    train_c = train.copy().drop('label', axis=1)
    test_c = test.copy()
    
    return [train_c, test_c], y_train



#Models
def MLP_model(n_layers, n_neurons):
    return  MLPClassifier(hidden_layer_sizes=(n_layers, n_neurons), verbose=True, random_state=1)

def linearSVM():
    pass

def SVM():
    pass

def RF():
    pass


#predict and get file
def predict(clf, test_data):
    def create_submission_csv(predicted_labels, test_data):
        test_data['index'] = test_data['index'] + 1
        test_data['Label'] = predicted_labels
        test_data.to_csv('mnist_predictions_mlp.csv', sep=',', columns=['index', 'Label'], index=False)

    predicted_labels = clf.predict(test_data)
    create_submission_csv(predicted_labels, test_data)



#driver
def main():
    datasets, y_train = load_data('train.csv', 'test.csv')
    mlp = MLP_model(3, 784)
    mlp.fit(datasets[0], y_train)
    scores = cross_val_score(mlp, datasets[0], y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, (scores.std() *100)))
    predict(mlp, datasets[1])


if __name__ == '__main__':
    main()