from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
# Calculated the accuracy using Adaboost,RandomForest,DecisionTree..........Adaboost was found to be the best model
# classifier

data = read_csv("C:\\Users\\kunal\\Downloads\\breast_cancer_prediction\\data.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

real_y = data.iloc[:, 0:1].values
real_x = data.iloc[:, 1:].values

X_train, X_test, Y_train, Y_test = train_test_split(real_x, real_y, test_size=0.2, random_state=0)
# model=DecisionTreeClassifier(criterion="gini")
model = AdaBoostClassifier(n_estimators=700)
model.fit(X_train, Y_train)
print(accuracy_score(Y_train, model.predict(X_train)))
print(accuracy_score(Y_test, model.predict(X_test)))
