#!/usr/bin/env python
# coding: utf-8

# In[66]:
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder 

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('gender.csv')

le=LabelEncoder()
df["FavoriteColor"]= le.fit_transform(df["FavoriteColor"])
df["FavoriteMusicGenre"]= le.fit_transform(df["FavoriteMusicGenre"])
df["FavoriteBeverage"]= le.fit_transform(df["FavoriteBeverage"])
df["FavoriteSoftDrink"]= le.fit_transform(df["FavoriteSoftDrink"])
df["Gender"]= le.fit_transform(df["Gender"])

y= df["Gender"]
X = df.drop(["Gender"], axis=1)


# # Logistic Regression# 

loj_model = LogisticRegression(solver="liblinear").fit(X,y)

loj_model.intercept_ #katsayılar

loj_model.coef_

loj_model.predict(X)[0:10]

y_pred = loj_model.predict(X)

confusion_matrix(y,y_pred)

accuracy_score(y,y_pred) #doğruluk oranı

#eğitim seti ve train seti 
X_train , X_test , y_train, y_test =train_test_split( X,y,test_size=0.70, random_state=20)

loj_model = LogisticRegression(solver="liblinear").fit(X_train,y_train)

y_pred= loj_model.predict(X_test)
accuracy_score(y_test , y_pred)


# # KNN

X_train , X_test , y_train, y_test =train_test_split( X,y,test_size=0.70, random_state=20)

knn_model = KNeighborsClassifier().fit(X_train, y_train)

knn_model

y_pred = knn_model.predict(X_test)

accuracy_score(y_test,y_pred)


# # SVM

svm_model = SVC(kernel= "linear").fit(X_train, y_train )

svm_model


y_pred = svm_model.predict(X_test)

accuracy_score(y_test,y_pred)

