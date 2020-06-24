from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
print(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print("LinearRegression")
model = LinearRegression()
model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))
print(model.score(X_test,Y_test))

print("RidgeRegression")
modelRidge=Ridge(alpha=0.01)
modelRidge.fit(X_train, Y_train)
print(modelRidge.score(X_train, Y_train))
print(modelRidge.score(X_test,Y_test))


print("LassoRegression")
lasso = Lasso()
lasso.fit(X_train, Y_train)

print(lasso.score(X_train, Y_train))
print(lasso.score(X_test,Y_test))

print("RandomForestRegression")
randomreg=RandomForestRegressor()
randomreg.fit(X_train, Y_train)
print(randomreg.score(X_train, Y_train))
print(randomreg.score(X_test,Y_test))

print("AdaBoostRegressor")
AdaBoostRegressorr=AdaBoostRegressor()
AdaBoostRegressorr.fit(X_train, Y_train)
print(AdaBoostRegressorr.score(X_train, Y_train))
print(AdaBoostRegressorr.score(X_test,Y_test))


print("DecisionTreeRegressor")
DecisionTreeRegressorr=DecisionTreeRegressor()
DecisionTreeRegressorr.fit(X_train, Y_train)
print(DecisionTreeRegressorr.score(X_train, Y_train))
print(DecisionTreeRegressorr.score(X_test,Y_test))