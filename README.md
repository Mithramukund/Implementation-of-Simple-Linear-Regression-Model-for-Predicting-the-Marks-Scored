# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kothai K
RegisterNumber:  212222240051
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
df.head()
df.tail()

## segregating data to variables

X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y

## graph plotting for training data

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

## Displaying predicted values

Y_pred

## Displaying actual values

Y_test

## Graph plot for training data

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## Graph plot for test data

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```
## Output:
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/8005dd99-5a09-4708-aaa2-0869a39e51ad)


![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/64ecbe05-5cd2-4719-8d41-71933e06d3a6)
# Array values of X
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/bd10a07b-9dfa-4b25-bbd3-0fe8e9c35e01)
# Array values of Y
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/c04e68b0-f11e-472b-a165-a7a688ad4d3f)
# Values of Y prediction
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/ad3d2f27-96e5-4e5c-8646-4d0071d68b33)
# Values of Y test
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/cb0eee3e-046b-4ddd-bdbd-c5e4f0322b4c)
# Training set graph
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/d4bbb6ea-76d7-4827-aa5a-8196ac72ae83)
# Testing set graph
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/8b3b3ebc-5020-4f2f-9e76-687a17f1c91c)
# Value of MSE,MAE & RMSE
![image](https://github.com/KothaiKumar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121215739/eab2546b-aa47-4bc6-b359-b8bd6bf5b2ea)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
