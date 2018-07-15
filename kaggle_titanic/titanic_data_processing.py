from sklearn.feature_extraction.text import CountVectorizer
import csv
import scipy as sp
import numpy as np

#reads in the training and testing csvs into lists
with open('train.csv', 'r') as inf:
    f = csv.reader(inf)
    train = list(f)

with open('test.csv', 'r') as inf:
    f = csv.reader(inf)
    test = list(f)

#original training and testing dataset converted to np array
o_train = np.array(train)
o_test = np.array(test)


#extracts the text and converts it to a bag of words (frequency), then converts bag of words to a np array
text = []
for instance in train[1:]:
    text.append(instance[3])
vectorizer = CountVectorizer(stop_words=['Mr.', 'Mrs.','Miss.','Dr.'])
matrix =  vectorizer.fit_transform(text)
np_matrix = matrix.toarray()
#print(np_matrix)

#creates new dataset headers by getting feature names, then maps it to the matrix
X_col = []
X_col.append(vectorizer.get_feature_names())
X_col = np.array(X_col)
text_train = np.append(X_col, np_matrix, 0)

#creates the no_text datasets by converting the original training set into an np array then deleting the text column
train_no_text = np.array(train)
train_no_text = np.delete(train_no_text, 3, 1)
test_no_text = np.delete(o_test, 2,1)

#convert the original list into an np array and then combine it with the bag of words
full_train = np.append(train_no_text, text_train, 1)

print(o_train.shape)
np.savetxt("o_train.csv", o_train, fmt = "%s", delimiter=",")
print(o_test.shape)
np.savetxt("test_no_text.csv", test_no_text, fmt = "%s", delimiter=",")
print(train_no_text.shape)
np.savetxt("train_no_text.csv", train_no_text, fmt = "%s", delimiter=",")
print(full_train.shape)
np.savetxt("full_train.csv", full_train, fmt = "%s", delimiter=",")