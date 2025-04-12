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
    
 
def cm_prediction(classifier,X_test):
     y_pred = classifier.predict(X_test)
        
        # Making the Confusion Matrix
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(y_test, y_pred)
        
     from sklearn.metrics import accuracy_score 
     from sklearn.metrics import classification_report 
        #from sklearn.metrics import confusion_matrix
        #cm = confusion_matrix(y_test, y_pred)
        
     Accuracy=accuracy_score(y_test, y_pred )
        
     report=classification_report(y_test, y_pred)
     return  classifier,Accuracy,report,X_test,y_test,cm

def logistic(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv=5)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, X_test, y_test, cm

def svm_linear(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(SVC(kernel='linear', random_state=0), param_grid, cv=5)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, X_test, y_test, cm
    
def svm_NL(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [1, 10], 'gamma': [0.01, 0.1, 1]}
    grid = GridSearchCV(SVC(kernel='rbf', random_state=0), param_grid, cv=5)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, X_test, y_test, cm

   
def knn(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {'n_neighbors': [3, 5, 7]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, X_test, y_test, cm

    
def Navie(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, X_test, y_test, cm

def Decision(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {'max_depth': [None, 5, 10], 'criterion': ['gini', 'entropy']}
    grid = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, cv=5)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, X_test, y_test, cm


def random(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [None, 5, 10],
        'criterion': ['entropy', 'gini']
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5, n_jobs=1)
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_
    classifier, Accuracy, report, X_test, y_test, cm = cm_prediction(classifier, X_test, y_test)
    return classifier, Accuracy, report, X_test, y_test, cm

    
def selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
    
    dataframe=pd.DataFrame(index=['ChiSquare'],columns=['Logistic','SVMl','SVMnl','KNN','Navie','Decision','Random'])
    for number,idex in enumerate(dataframe.index):      
        dataframe['Logistic'][idex]=acclog[number]       
        dataframe['SVMl'][idex]=accsvml[number]
        dataframe['SVMnl'][idex]=accsvmnl[number]
        dataframe['KNN'][idex]=accknn[number]
        dataframe['Navie'][idex]=accnav[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
    return dataframe
    
dataset1=pd.read_csv("prep.csv",index_col=None)

df2=dataset1

df2 = pd.get_dummies(df2, drop_first=True)

indep_X=df2.drop('classification_yes', 1)
dep_Y=df2['classification_yes']


kbest=selectkbest(indep_X,dep_Y,5)       

acclog=[]
accsvml=[]
accsvmnl=[]
accknn=[]
accnav=[]
accdes=[]
accrf=[]


X_train, X_test, y_train, y_test=split_scalar(kbest,dep_Y)   
    
        
classifier,Accuracy,report,X_test,y_test,cm=logistic(X_train,y_train,X_test, y_test)
acclog.append(Accuracy)

classifier,Accuracy,report,X_test,y_test,cm=svm_linear(X_train,y_train,X_test, y_test)  
accsvml.append(Accuracy)
    
classifier,Accuracy,report,X_test,y_test,cm=svm_NL(X_train,y_train,X_test, y_test)  
accsvmnl.append(Accuracy)
    
classifier,Accuracy,report,X_test,y_test,cm=knn(X_train,y_train,X_test, y_test)  
accknn.append(Accuracy)
    
classifier,Accuracy,report,X_test,y_test,cm=Navie(X_train,y_train,X_test, y_test)  
accnav.append(Accuracy)
    
classifier,Accuracy,report,X_test,y_test,cm=Decision(X_train,y_train,X_test, y_test)  
accdes.append(Accuracy)
    
classifier,Accuracy,report,X_test,y_test,cm=random(X_train,y_train,X_test, y_test)  
accrf.append(Accuracy)
    
result=selectk_Classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf)

result