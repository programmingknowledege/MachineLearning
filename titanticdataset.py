import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("titanic_data.csv")

data.drop(["PassengerId", "Cabin", "Ticket", "Name"], axis=1, inplace=True)

data['Embarked'] = data['Embarked'].replace(np.nan, 'S')
median = data['Age'].median()
data['Age'] = data['Age'].replace(np.nan, median)

le_Sex = LabelEncoder()
data["Sex"] = le_Sex.fit_transform(data["Sex"])

le_Embarked = LabelEncoder()
data["Embarked"] = le_Embarked.fit_transform(data["Embarked"])

real_x = data.drop(["Survived"], axis=1)
real_x = real_x.values
real_y = data.Survived.values

X_train, X_test, Y_train, Y_test = train_test_split(real_x, real_y, test_size=0.2, random_state=0)
kneigbour = KNeighborsClassifier(n_neighbors=51)
kneigbour.fit(X_train, Y_train)
print(accuracy_score(Y_train, kneigbour.predict(X_train)))
print(accuracy_score(Y_test, kneigbour.predict(X_test)))

logistic = LogisticRegression()
logistic.fit(X_train, Y_train)
print(accuracy_score(Y_train, logistic.predict(X_train)))
print(accuracy_score(Y_test, logistic.predict(X_test)))

deciontree = DecisionTreeClassifier()
deciontree.fit(X_train, Y_train)
print(accuracy_score(Y_train, deciontree.predict(X_train)))
print(accuracy_score(Y_test, deciontree.predict(X_test)))

random = RandomForestClassifier(n_estimators=500)
random.fit(X_train, Y_train)
print(accuracy_score(Y_train, random.predict(X_train)))
print(accuracy_score(Y_test, random.predict(X_test)))

randomtree = ExtraTreesClassifier(n_estimators=500)
randomtree.fit(X_train, Y_train)
print(accuracy_score(Y_train, randomtree.predict(X_train)))
print(accuracy_score(Y_test, randomtree.predict(X_test)))

adaboost = AdaBoostClassifier(n_estimators=300)
adaboost.fit(X_train, Y_train)
print(accuracy_score(Y_train, adaboost.predict(X_train)))
print(accuracy_score(Y_test, adaboost.predict(X_test)))

gardientboost = GradientBoostingClassifier(n_estimators=500)
gardientboost.fit(X_train, Y_train)
print(accuracy_score(Y_train, gardientboost.predict(X_train)))
print(accuracy_score(Y_test, gardientboost.predict(X_test)))

kfold = KFold(n_splits=10, random_state=0)
result = cross_val_score(RandomForestClassifier(), X_train, Y_train, cv=10, scoring="accuracy")
print(result.mean())

# RandomForest gave the best accuracy
