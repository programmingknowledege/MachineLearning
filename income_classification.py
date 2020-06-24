import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("C:\\Users\\kunal\\Downloads\\income-classification\\income_evaluation.csv")
print(data.columns)

le_income=LabelEncoder()
data[" income"]=le_income.fit_transform(data[" income"])
print(data[" income"].unique())