# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
# Step-1
Import the libraries and read the data frame using pandas.
# Step-2
Calculate the null values from dataframe and apply label encoder.
# Step-3
Apply decision tree classifier on the dataframe.
# Step-4
Obtain the value of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:IYYANAR S
RegisterNumber:212222240036
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
# data.head():
![head](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/e5bd665a-f1d2-43be-9f14-5a4645ecc8a5)

# data.info():
![info](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/cc941eec-cb42-4983-8d69-3445e894c15a)

# is null() and sum():
![null and sum](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/ef3e5763-c417-4ca2-86a3-2e153126db44)

# data value counts():
![value count](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/ce2c403a-d16c-4001-9e25-7d4df391a379)

# data.head() for salary:
![head for salary](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/a55b128b-86ad-4c45-a4d1-fb4f737c7fcf)

# x.head():
![x head](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/b064a4ba-999e-43b1-8d5c-e51b7530aa78)

# Accuracy value:
![accuracy value](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/5f485532-2a00-44e3-8f9d-23ee209a096e)

# Data prediction:
![data prediction](https://github.com/Iyyanar22009120/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118680259/783c60c2-c34f-442e-8225-0368c09b2d58)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
