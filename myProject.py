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


# In[67]:


import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[68]:


df = pd.read_csv('gender.csv')


# In[69]:


le=LabelEncoder()
df["FavoriteColor"]= le.fit_transform(df["FavoriteColor"])
df["FavoriteMusicGenre"]= le.fit_transform(df["FavoriteMusicGenre"])
df["FavoriteBeverage"]= le.fit_transform(df["FavoriteBeverage"])
df["FavoriteSoftDrink"]= le.fit_transform(df["FavoriteSoftDrink"])
df["Gender"]= le.fit_transform(df["Gender"])


# In[70]:


y= df["Gender"]
X = df.drop(["Gender"], axis=1)


# # Logistic Regression# 

# In[71]:


loj_model = LogisticRegression(solver="liblinear").fit(X,y)


# In[72]:


loj_model.intercept_ #katsayılar


# In[73]:


loj_model.coef_


# In[74]:


loj_model.predict(X)[0:10]


# In[75]:


y_pred = loj_model.predict(X)


# In[76]:


confusion_matrix(y,y_pred)


# In[77]:


accuracy_score(y,y_pred) #doğruluk oranı


# In[78]:


#eğitim seti ve train seti 


# In[79]:


X_train , X_test , y_train, y_test =train_test_split( X,y,test_size=0.70, random_state=20)


# In[80]:


loj_model = LogisticRegression(solver="liblinear").fit(X_train,y_train)


# In[81]:


y_pred= loj_model.predict(X_test)


# In[82]:


accuracy_score(y_test , y_pred)


# # KNN

# In[83]:


X_train , X_test , y_train, y_test =train_test_split( X,y,test_size=0.70, random_state=20)


# In[84]:


knn_model = KNeighborsClassifier().fit(X_train, y_train)


# In[85]:


knn_model


# In[86]:


y_pred = knn_model.predict(X_test)


# In[87]:


accuracy_score(y_test,y_pred)


# # SVM

# In[92]:


svm_model = SVC(kernel= "linear").fit(X_train, y_train )


# In[93]:


svm_model


# In[95]:


y_pred = svm_model.predict(X_test)


# In[96]:


accuracy_score(y_test,y_pred)

