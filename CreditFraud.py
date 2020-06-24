import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

data = pd.read_csv("C:\\Users\\kunal\\Downloads\\creditcardfraud\\creditcard.csv")
data.drop(["Time"], axis=1, inplace=True)

real_x = data.drop(["Class"], axis=1)
real_x = real_x.values
real_y = data.Class.values

scaler = StandardScaler()
real_x = scaler.fit_transform(real_x)

X_train, X_test, Y_train, Y_test = train_test_split(real_x, real_y, test_size=0.2, random_state=0)
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