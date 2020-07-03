#Here we will use the preprocessed and feature engineered datasets from the 'exploration' document
#to be sure they are the same show the first few rows of each
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print(train_df.head())
print(test_df.head())

#For now we will only work with the training data. Split it into features and labels
X = train.drop('Survived', axis = 1)
y = train['Survived']

#To be sure our variables are sufficiently uncorrelated we view the correlation plot
X_corr = X.corr()
X_corr.style.background_gradient(cmap = 'coolwarm').set_precision(2)

#the features are sufficiently uncorrelated so we continue 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Time to model. Let's try Logistic Regression first
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()
logreg_cv = cross_val_score(estimator = logreg, X_scaled, y, cv = 5)
print(logreg_cv)

#We can also try a Random Forest model
from skearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_cv = cross_val_score(estimator = rf, X_scaled, y, cv = 5)
print(rf_cv)

#Further models could be explored, but for now these two will suffice
##Since Random Forest performed better out of the box. Let's tune its hyperparameters
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 12, stratify = y
estimator_space = np.arange(50, 500, 50)
depth_space = np.arange(10,110,10)
features_space = ['sqrt']

params = {'n_estimators' : estimator_space,
         'max_depth' : depth_space,
         'max_features' : features_space}
rfmodel = RandomForestClassifier()
rfmodel_search = RandomizedSearchCV(rfmodel, params, cv = 5)
rfmodel_search.fit(X_train, y_train)
print("Test score is {}".format(rfmodel_search.score(X_test, y_test)))
print("Tuned Random Forest Parameters: {}".format(rfmodel_search.best_params_))
print("Best score is {}".format(rfmodel_search.best_score_))

#saving this best model we verify its performance again with cross validation
best_rf = rfmodel_search.best_estimator_
best_rf_cv = cross_val_score(estimator = best_rf, X_scaled, y, cv = 5)
print(best_rf_cv)
print('Predicting on new data we should expect approximately ', np.mean(best_rf_cv), '% accuracy from this model)
