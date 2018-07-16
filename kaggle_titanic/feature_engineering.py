import numpy as np
import pandas as pd
import random
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

#read in the train and test datasets
train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)
test['Survived'] = 0 #add a survived column for dimensional consistency
y_train = train['Survived']

#check class balance in training set
print(train['Survived'].value_counts())

#create copies to not alter the originals
train_c = train.copy()
test_c = test.copy()

#combine test and training to make feature changes to both datasets
#concat = train.append(test, ignore_index=True)

#extract text from the training and testing datasets, will be used later
text_train = train_c['Name'].values
text_test = test_c['Name'].values

#confirm shape sizes
print(train_c.shape) #891 instances
print(test_c.shape) #418 instances

#drop the label and text based features deemed unnecessary
train_c = train_c.drop(['Name','Cabin','Ticket','Survived'], 1)
test_c = test_c.drop(['Name','Cabin','Ticket','Survived'], 1)

#fill in any remaining NaN values with 0
train_c = train_c.fillna(0)
test_c = test_c.fillna(0)

#convert the categorical features into a one-hot encoding space for the training and testing datasets
train_c = pd.get_dummies(train_c, columns=['Sex','Embarked', 'Pclass','Parch'])
test_c = pd.get_dummies(test_c, columns=['Sex','Embarked', 'Pclass','Parch'])

#add the bag of words representation of the text features in the following lines of code
vectorizer = CountVectorizer(ngram_range=(1, 1))

#fit and transform the vocabulary for the training text, then transform the test text based on this vocabulary
text_b_train = vectorizer.fit_transform(text_train)
text_b_test = vectorizer.transform(text_test)

#add the bag of words features back to the numeric features
X_train = np.hstack([text_b_train.toarray(), train_c])
X_train_col = vectorizer.get_feature_names()+list(train_c)
X_test = np.hstack([text_b_test.toarray(), test_c])
X_test_col = vectorizer.get_feature_names()+list(test_c)

#create a logistic regression model and check performance using 5-fold cross-validation
clf = LogisticRegression(C=5.0)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, (scores.std() * 2*100)))

#predict and add predictions back to the test copy, then output the results to file
predicted = clf.predict(X_test)
test_c['Survived'] = predicted
test_c.to_csv('titanic_predictions_LR.csv', sep=',', columns=['PassengerId', 'Survived'], index=False)
