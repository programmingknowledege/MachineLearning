from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
data = read_csv("C:\\Users\\kunal\\Downloads\\hr-analytics\\HR_comma_sep.csv")

datasalary = pd.get_dummies(data["salary"], prefix="salary")
datadepartment = pd.get_dummies(data["Department"], prefix="Department")

data.drop(["salary","Department"], inplace=True, axis=1)
datafinal = pd.concat([data, datasalary, datadepartment], axis=1)

print(datafinal.columns)

real_x = datafinal.drop(["left"], axis=1)
real_x = real_x.values
real_y = datafinal.left.values


X_train, X_test, Y_train, Y_test = train_test_split(real_x, real_y, random_state=0, test_size=0.2)

decisionmodel=DecisionTreeClassifier()
decisionmodel.fit(X_train,Y_train)
print(accuracy_score(Y_train,decisionmodel.predict(X_train)))
print(accuracy_score(Y_test,decisionmodel.predict(X_test)))

randommodel=RandomForestClassifier(n_estimators=500)
randommodel.fit(X_train,Y_train)
print(accuracy_score(Y_train,randommodel.predict(X_train)))
print(accuracy_score(Y_test,randommodel.predict(X_test)))

extratree=ExtraTreesClassifier(n_estimators=900)
extratree.fit(X_train,Y_train)
print(accuracy_score(Y_train,extratree.predict(X_train)))
print(accuracy_score(Y_test,extratree.predict(X_test)))

adaboosttree=AdaBoostClassifier(base_estimator=RandomForestClassifier(),n_estimators=900)
adaboosttree.fit(X_train,Y_train)
print(accuracy_score(Y_train,adaboosttree.predict(X_train)))
print(accuracy_score(Y_test,adaboosttree.predict(X_test)))


#Got max accuracy with adaboost of 99%+