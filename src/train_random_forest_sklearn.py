import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../data/processed/bitstampUSD.csv")
df.head()

train = df[df["Timestamp"]<= 1529899200]
test = df[df["Timestamp"]> 1529899200]

X_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values
X_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values

X_train_scaled = mms.fit_transform(X_train)
X_test_scaled = mms.transform(X_test)

X_train_scaled

from sklearn import linear_model

kf = KFold(n_splits=10)
coef=[]
MSE =[]
for train_index, test_index in kf.split(X_train_scaled):
    Xtrain, Xval = X_train_scaled[train_index], X_train_scaled[test_index]
    ytrain, yval = y_train[train_index], y_train[test_index]
    reg = linear_model.LinearRegression()
    
    start = time.time()
    reg.fit(Xtrain,ytrain)
    end = time.time() 
    print("Time Taken {:.2f}".format(end-start))
    y_pred_val = reg.predict(Xval)
    mse = mean_squared_error(yval, y_pred_val)
    MSE.append(mse)
    coef.append(reg)
    
print("Average Error is {:.2f}".format(np.array(MSE).mean()))

alpha=[0.01,0.1,10,100,1000,10000]

MSE=[]
for val in alpha:
    temp=[]
    for train_index, test_index in kf.split(X_train_scaled):
        Xtrain, Xval = X_train_scaled[train_index], X_train_scaled[test_index]
        ytrain, yval = y_train[train_index], y_train[test_index]
        reg = linear_model.Ridge(alpha=val)

        #start = time.time()
        reg.fit(Xtrain,ytrain)
        #end = time.time() 
        #print("Time Taken {:.2f}".format(end-start))
        y_pred_val = reg.predict(Xval)
        mse = mean_squared_error(yval, y_pred_val)
        temp.append(mse)
    MSE.append(np.array(temp).mean())
	
MSE

MSE=[]
for val in alpha:
    temp=[]
    for train_index, test_index in kf.split(X_train_scaled):
        Xtrain, Xval = X_train_scaled[train_index], X_train_scaled[test_index]
        ytrain, yval = y_train[train_index], y_train[test_index]
        reg = linear_model.Lasso(alpha=val)

        #start = time.time()
        reg.fit(Xtrain,ytrain)
        #end = time.time() 
        #print("Time Taken {:.2f}".format(end-start))
        y_pred_val = reg.predict(Xval)
        mse = mean_squared_error(yval, y_pred_val)
        temp.append(mse)
    MSE.append(np.array(temp).mean())
	
MSE

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

kf = KFold(n_splits=5)
MSE=[]
for val in [1,2,3,4,5]:
    temp=[]
    for train_index, test_index in kf.split(X_train_scaled):
        
        trans = PolynomialFeatures(degree=val)
        data = trans.fit_transform(X_train_scaled)
        
        Xtrain, Xval = data[train_index], data[test_index]
        ytrain, yval = y_train[train_index], y_train[test_index]
        reg = linear_model.LinearRegression()


        #start = time.time()
        reg.fit(Xtrain,ytrain)
        #end = time.time() 
        #print("Time Taken {:.2f}".format(end-start))
        y_pred_val = reg.predict(Xval)
        mse = mean_squared_error(yval, y_pred_val)
        temp.append(mse)
    MSE.append(np.array(temp).mean())
	
MSE

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': [4,5,6,7],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
						  
grid_search.fit(X_train_scaled, y_train)
grid_search.best_params_
{'bootstrap': True,
 'max_depth': 80,
 'max_features': 3,
 'min_samples_leaf': 5,
 'min_samples_split': 12,
 'n_estimators': 100}
best_grid = grid_search.b

y_pred_test = coef[0].predict(X_test_scaled)
print('Mean squared error test set: %.2f'
      % mean_squared_error(y_test, y_pred_test))
	  
y_pred_test = coef[-1].predict(X_test_scaled)
print('Mean squared error test set: %.2f'
      % mean_squared_error(y_test, y_pred_test))
	  
y_pred_train = reg.predict(X_train_scaled)
y_pred_test = reg.predict(X_test_scaled)

print('Mean squared error train set: %.2f'
      % mean_squared_error(y_train, y_pred_train))
print('Mean squared error test set: %.2f'
      % mean_squared_error(y_test, y_pred_test))
# The coefficient of determination: 1 is perfect prediction

X_train.shape

reg.coef_

reg = linear_model.LinearRegression()
aa = cross_val_score(reg, X_train_scaled, y_train, cv=10 ,scoring = "neg_mean_squared_error")

import numpy as np

aa = np.array(aa)

aa = aa*-1

aa
