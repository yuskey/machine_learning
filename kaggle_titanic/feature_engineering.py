import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import metrics


train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)
test['Survived'] = 0

#combine test and training to make feature changes to both datasets
concat = train.append(test, ignore_index=True)

print(train.shape)  #891 instances
print(test.shape)   #418 instances
print(concat.shape)


#train['Embarked'] = train['Embarked'].fillna('A')

#convert one-hot features like sex and embarked
#encoder = preprocessing.LabelBinarizer()
#new_sex = encoder.fit_transform(train['Sex'])
#new_embarked = encoder.fit_transform(train['Embarked'])
#train['Sex'] = new_sex
#train['Embarked'] = new_embarked

#clf = LogisticRegression()
#clf.fit(X_train, y_train)
#scores = cross_val_score(clf, X_train, y_train, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#clf.predict(X_test)