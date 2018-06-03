from sklearn.feature_extraction.text import CountVectorizer
import csv
import scipy as sp
import numpy as np

#reads in the training csv to a list
with open('train.csv', 'r') as inf:
    f = csv.reader(inf)
    train = list(f)

#original training dataset converted to np array
o_train = np.array(train)

#extracts the text and converts it to a bag of words (frequency), then converts bag of words to a np array
text = []
for instance in train[1:]:
    text.append(instance[3])
vectorizer = CountVectorizer(stop_words=['Mr.', 'Mrs.','Miss.','Dr.'])
matrix =  vectorizer.fit_transform(text)
np_matrix = matrix.toarray()

#creates new dataset headers by getting feature names, then maps it to the matrix
X_col = []
X_col.append(vectorizer.get_feature_names())
X_col = np.array(X_col)
text_train = np.append(X_col, np_matrix, 0)

#removes the name column from the original list as we converted it to bag of words
for row in train:
    del row[3]

#convert the original list into an np array and then combine it with the bag of words
train_no_text = np.array(train)
full_train = np.append(train_no_text, text_train, 1)

print(o_train.shape)
print(train_no_text.shape)
print(full_train.shape)
