import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("C:\\Users\\kunal\\Downloads\\iris.csv")

le_variety = LabelEncoder()
data["variety"] = le_variety.fit_transform(data["variety"])
real_x = data.iloc[:, 1:4].values
real_y = data.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(real_x, real_y, test_size=0.2, random_state=0)
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)
print(accuracy_score(Y_train, model.predict(X_train)))
y_pred = model.predict(X_test)
print(accuracy_score(Y_test, model.predict(X_test)))

print(confusion_matrix(Y_test, y_pred))
