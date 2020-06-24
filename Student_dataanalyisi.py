from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from  sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("student-data.csv")

le_school = LabelEncoder()
data["school"] = le_school.fit_transform(data["school"])
le_sex = LabelEncoder()
data["sex"] = le_school.fit_transform(data["sex"])
le_address = LabelEncoder()
data["address"] = le_school.fit_transform(data["address"])
le_famsize = LabelEncoder()
data["famsize"] = le_school.fit_transform(data["famsize"])
le_Pstatus = LabelEncoder()
data["Pstatus"] = le_school.fit_transform(data["Pstatus"])
le_Mjob = LabelEncoder()
data["Mjob"] = le_school.fit_transform(data["Mjob"])
le_Fjob = LabelEncoder()
data["Fjob"] = le_school.fit_transform(data["Fjob"])
le_reason = LabelEncoder()
data["reason"] = le_school.fit_transform(data["reason"])
le_guardian = LabelEncoder()
data["guardian"] = le_school.fit_transform(data["guardian"])
le_schoolsup = LabelEncoder()
data["schoolsup"] = le_school.fit_transform(data["schoolsup"])
le_famsup = LabelEncoder()
data["famsup"] = le_school.fit_transform(data["famsup"])
le_paid = LabelEncoder()
data["paid"] = le_school.fit_transform(data["paid"])
le_activities = LabelEncoder()
data["activities"] = le_school.fit_transform(data["activities"])
le_nursery = LabelEncoder()
data["nursery"] = le_school.fit_transform(data["nursery"])
le_higher = LabelEncoder()
data["higher"] = le_school.fit_transform(data["higher"])
le_internet = LabelEncoder()
data["internet"] = le_school.fit_transform(data["internet"])
le_romantic = LabelEncoder()
data["romantic"] = le_school.fit_transform(data["romantic"])
le_passed = LabelEncoder()
data["passed"] = le_school.fit_transform(data["passed"])

real_x = data.drop(["passed"], axis=1)
real_x = real_x.values
real_y = data.passed.values


X_train, X_test, Y_train, Y_test = train_test_split(real_x, real_y, test_size=0.2, random_state=0)
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, Y_train)
print(accuracy_score(Y_train, decisiontree.predict(X_train)))
print(accuracy_score(Y_test, decisiontree.predict(X_test)))

randomtree = RandomForestClassifier(n_estimators=900)
randomtree.fit(X_train, Y_train)
print(accuracy_score(Y_train, randomtree.predict(X_train)))
print(accuracy_score(Y_test, randomtree.predict(X_test)))

adaboost = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=900)
adaboost.fit(X_train, Y_train)
print(accuracy_score(Y_train, adaboost.predict(X_train)))
print(accuracy_score(Y_test, adaboost.predict(X_test)))

logistic = LogisticRegression(random_state=0)
logistic.fit(X_train, Y_train)
print(accuracy_score(Y_train, logistic.predict(X_train)))
print(accuracy_score(Y_test, logistic.predict(X_test)))

gradientboost = GradientBoostingClassifier(n_estimators=500)
gradientboost.fit(X_train, Y_train)
print(accuracy_score(Y_train, gradientboost.predict(X_train)))
print(accuracy_score(Y_test, gradientboost.predict(X_test)))

kneighbour = KNeighborsClassifier(n_neighbors=99)
kneighbour.fit(X_train, Y_train)
print(accuracy_score(Y_train, kneighbour.predict(X_train)))
print(accuracy_score(Y_test, kneighbour.predict(X_test)))

extraclassifier = ExtraTreesClassifier(n_estimators=900)
extraclassifier.fit(X_train, Y_train)
print(accuracy_score(Y_train, extraclassifier.predict(X_train)))
print(accuracy_score(Y_test, extraclassifier.predict(X_test)))

