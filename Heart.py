from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

data = read_csv("C:\\Users\\kunal\\Downloads\\heartcsv\\Heart.csv")
data.drop(["Unnamed: 0"], axis=1, inplace=True)
data.Thal.fillna(value="normal", inplace=True)

data.fillna(value=0, inplace=True)
le_chestpain = LabelEncoder()
le_ahd = LabelEncoder()
le_thal = LabelEncoder()
data["ChestPain"] = le_chestpain.fit_transform(data["ChestPain"])
data["AHD"] = le_ahd.fit_transform(data["AHD"])
data["Thal"] = le_thal.fit_transform(data["Thal"])

real_x = data.iloc[:, 0:13].values
real_y = data.iloc[:, 13].values

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

# Out of all the model training, Randomforest and ExtraTreeClassifier were found to be the best classifier
