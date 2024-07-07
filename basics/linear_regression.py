import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

boston=fetch_california_housing()
dataset=pd.DataFrame(boston.data,columns=boston.feature_names)
dataset['PRICE']=boston.target
#print(dataset.head())
#print(dataset.describe())
#print(dataset.isnull().sum())
'''
sns.histplot(dataset['PRICE'],bins=30,kde=True)
plt.xlabel('Price')
plt.ylabel('frequency')
plt.title('housing price distribution')
plt.show()
'''
X=dataset.drop('PRICE',axis=1)
Y=dataset['PRICE']
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LinearRegression()

model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
calculate_error=mean_squared_error(Y_test,Y_pred)
#print(calculate_error)

plt.scatter(Y_test,Y_pred)
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('comaprison')
#plt.show()


residuals = Y_test - Y_pred
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()


