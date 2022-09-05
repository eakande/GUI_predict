
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from pandas import DataFrame
from matplotlib import pyplot

# data=pd.read_csv('Data.csv')
# X_data = data.drop(['Headline', 'Date'], axis=1)

data = pd.read_excel('Data.xlsx', sheet_name="2011_insecurity").dropna()

X = data.drop(['Headline Inflation', 'Date'], axis=1)
y = data['Headline Inflation']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#model = RandomForestRegressor(n_estimators=50, max_depth=5)


rf = RandomForestRegressor(n_estimators=4000,
                              n_jobs=-1,
                              oob_score=True,
                              bootstrap=True,
                              max_depth=5,
                              random_state=42)


rf.fit(X,y)
y_pred=rf.predict(X)
rf.score (X_train,y_train), rf.score(X_test,y_test)

print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'
      .format(rf.score(X_train, y_train), 
             rf.oob_score_,rf.score(X_test, y_test)))


Time=list(range(0,134,1))


# plotting the points  

pyplot.plot(Time, y, label='Expected')
pyplot.plot(Time, y_pred, label='Predicted')
pyplot.legend()
pyplot.show()

################## Feature Importance ##############


#### avoid the message error: 'numpy.ndarray' object has no attribute 'columns'
#### we need to obtain the coefficients and the x columns


############### Select the most important features ######


feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(5)

feat_importances.nlargest(5).plot(kind='barh')
plt.show()



X2= data[['Food Prices','Transportation Prices','Housing and Utility Prices']]


y2 = y
###### Split Data set

X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2,test_size=0.3,random_state=0)



############# Fit based on feature importance #######

#X2_train = np.nan_to_num(X2_train)
#X2_test = np.nan_to_num(X2_test)


rf2 = RandomForestRegressor(n_estimators=4000,
                              n_jobs=-1,
                              oob_score=True,
                              bootstrap=True,
                              max_depth=5,
                              random_state=42)

rf2.fit(X2,y2)
y2_pred=rf2.predict(X2)
rf2.score (X2_train,y2_train), rf2.score(X2_test,y2_test),rf2.oob_score_

y2_test =y2_test.sort_index()

y2_pred = pd.Series(y2_pred).sort_index()







Time=list(range(0, 134,1))

# plotting the points  

pyplot.plot(Time, y2, label='Expected')
pyplot.plot(Time, y2_pred, label='Predicted')
pyplot.legend()
pyplot.show()


pickle.dump(rf2, open('modeltest1.pkl','wb'))

modeltest1 = pickle.load(open('modeltest1.pkl','rb'))



