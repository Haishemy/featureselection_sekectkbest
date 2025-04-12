import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV



def selectkbest(indep_X,dep_Y,n):
        test = SelectKBest(score_func=chi2, k=n)
        fit1= test.fit(indep_X,dep_Y)
        # summarize scores       
        selectk_features = fit1.transform(indep_X)
        return selectk_features
 
def selected_feature_names(indep_X, dep_Y, n):
    selector = SelectKBest(score_func=chi2, k=n)
    selector.fit(indep_X, dep_Y)
    
    # Get the selected feature names
    selected_columns = indep_X.columns[selector.get_support()]
    
   # print(f'Selected feature columns: {selected_columns.tolist()}')
    
    return selected_columns.tolist()

    
def split_scalar(indep_X,dep_Y):
        X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)    
        return X_train, X_test, y_train, y_test
    
def r2_prediction(regressor,X_test,y_test):
     y_pred = regressor.predict(X_test)
     from sklearn.metrics import r2_score
     r2=r2_score(y_test,y_pred)
     return r2
 
def gridsearch_linear(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    # LinearRegression has no hyperparameters to tune usually, but we can still use GridSearchCV
    params = {}  # No hyperparameters for basic LinearRegression
    grid = GridSearchCV(model, params, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    r2 = r2_prediction(best_model, X_test, y_test)
    return r2, grid.best_params_

def gridsearch_svm_linear(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVR
    model = SVR(kernel='linear')
    params = {
        'C': [0.1, 1, 10, 100]
    }
    grid = GridSearchCV(model, params, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    r2 = r2_prediction(best_model, X_test, y_test)
    return r2, grid.best_params_

def gridsearch_svm_rbf(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVR
    model = SVR(kernel='rbf')
    params = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1, 10]
    }
    grid = GridSearchCV(model, params, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    r2 = r2_prediction(best_model, X_test, y_test)
    return r2, grid.best_params_

def gridsearch_decision_tree(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=0)
    params = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(model, params, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    r2 = r2_prediction(best_model, X_test, y_test)
    return r2, grid.best_params_

def gridsearch_random_forest(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=0)
    params = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    r2 = r2_prediction(best_model, X_test, y_test)
    return r2, grid.best_params_
    
    
def selectk_regression(acclin,accsvml,accsvmnl,accdes,accrf): 
    
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Linear','SVMl','SVMnl','Decision','Random'
                                                                                     ])

    for number,idex in enumerate(dataframe.index):
        
        dataframe['Linear'][idex]=acclin[number]       
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVMnl'][idex]=accsvmnl[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe
    
dataset1=pd.read_csv("prep.csv",index_col=None)

df2=dataset1

df2 = pd.get_dummies(df2, drop_first=True)

indep_X=df2.drop('classification_yes', 1)
dep_Y=df2['classification_yes']


kbest=selectkbest(indep_X,dep_Y,5)      

acclin = []
accsvml = []
accsvmnl = []
accdes = []
accrf = []

X_train, X_test, y_train, y_test = split_scalar(kbest, dep_Y)

r2_lin, best_params_lin = gridsearch_linear(X_train, y_train, X_test, y_test)
acclin.append(r2_lin)

r2_svml, best_params_svml = gridsearch_svm_linear(X_train, y_train, X_test, y_test)
accsvml.append(r2_svml)

r2_svmnl, best_params_svmnl = gridsearch_svm_rbf(X_train, y_train, X_test, y_test)
accsvmnl.append(r2_svmnl)

r2_dtree, best_params_dtree = gridsearch_decision_tree(X_train, y_train, X_test, y_test)
accdes.append(r2_dtree)

r2_rf, best_params_rf = gridsearch_random_forest(X_train, y_train, X_test, y_test)
accrf.append(r2_rf)

    
result = selectk_regression(acclin, accsvml, accsvmnl, accdes, accrf)



result