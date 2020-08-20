import matplotlib.pyplot as plt                             
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston=load_boston()
boston

df_X=pd.DataFrame(boston.data,columns=boston.feature_names)
df_X.head()

df_Y=pd.DataFrame(boston.target)
df_Y.head()

X_train,X_test,Y_train,Y_test=train_test_split(df_X,df_Y,test_size=0.33,random_state=42)

reg=linear_model.LinearRegression()

reg.fit(X_train,Y_train)

print(reg.coef_) 

y_pred=reg.predict(X_test)

print(mean_squared_error(Y_test,y_pred))

print(reg.intercept_)


