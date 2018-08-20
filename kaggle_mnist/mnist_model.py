import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import warnings

"""
    To Do:
        1. Try different parameters, could use gridCV but 
           not many parameters to try so will just stick with manual tuning (done) 
           Best model:
                accuracy = .9677
                layers = input, 512,256,128,64,32, output
                alpha = 0.0001

        2. Try a convolutional neural network (needs to be done on desktop, no CNTK for mac)
            2.1. Need to convert data into a 28x28x1 format for ConvNet 
            2.2. Build keras ConvNet model
            2.3. Figure out how to import keras model into sklearn, try
                 from keras.wrappers.scikit_learn import KerasClassifier
        

"""

#data preparation
def load_data(training_file, testing_file):
    #instances consists of 784 features (28x28x1 image if we convert to matrix form)
    train = pd.read_csv(training_file, header = 0)
    test = pd.read_csv(testing_file, header = 0)
    
    y_train = train['label']
    train = train.drop('label', axis=1)
    
    return [train, test], y_train



#Model
def MLP_model(n_neurons):
    #random_state is set for reproducibility, sets all weights to the same starting point
    return  MLPClassifier(hidden_layer_sizes=(n_neurons), verbose=True, random_state=1, alpha=.0001)


#predict and get file
def predict(clf, test_data, accuracy):
    def create_submission_csv(predicted_labels, test_data):
        test_data = test_data.reset_index()
        test_data['index'] += 1
        test_data['Label'] = predicted_labels
        test_data.to_csv('mnist_predictions_mlp_{0:.{1}f}.csv'.format(accuracy, 4), sep=',', columns=['index', 'Label'], index=False)

    predicted_labels = clf.predict(test_data)
    create_submission_csv(predicted_labels, test_data)



#driver
def main():
    datasets, y_train = load_data('train.csv', 'test.csv')

    #hidden layers should be between input and output layer sizes (785 to 10) and 
    #should only ever decrease or stay the same so you don't lose information
    mlp = MLP_model((512,256,128,64,32))
    mlp.fit(datasets[0], y_train)
    
    #test model using cross-validation, 5-fold should be enough
    scores = cross_val_score(mlp, datasets[0], y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, (scores.std() *100)))
    
    #predict test instances and create submission file
    predict(mlp, datasets[1], scores.mean())


if __name__ == '__main__':
    main()