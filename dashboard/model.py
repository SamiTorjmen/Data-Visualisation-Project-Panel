import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class History:
    def __init__(self, model, X_train, y_train, X_test, y_test, y_pred, residuals):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.residuals = residuals
        
    def to_dict(self):
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'model': self.model,
            'y_pred': self.y_pred,
            'residuals': self.residuals
        }

from sklearn.model_selection import train_test_split
def model_history(df, target, model):
    
    Y = df[target]

    columns_to_drop = ['pass', 'target_name', 'id', target]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns_to_drop, axis=1)
    
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.3, random_state=123)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_instance = model()
    model_instance.fit(X_train, y_train)

    y_pred = model_instance.predict(X_test)

    residuals = y_test - y_pred

    history = History(str(model_instance)[:-2], X_train, y_train, X_test, y_test, y_pred, residuals)
    return history


def df_reg(df):
    df_copy = df.copy()
    df_copy = pd.get_dummies(df_copy, prefix='gender_', columns=['gender'])
    df_copy = pd.get_dummies(df_copy, prefix='race_', columns=['race/ethnicity'])
    df_copy = pd.get_dummies(df_copy, prefix='lunch_', columns=['lunch'])
    edu_dict = {"some high school": 0, "high school": 1, "some college": 2, "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5}
    df_copy['parental level of education'] = df_copy['parental level of education'].replace(edu_dict)
    edu_dict = {"none": 0, "completed": 1}
    df_copy['test preparation course'] = df_copy['test preparation course'].replace(edu_dict)
    df_copy["average_score"] = df_copy[["math score", "reading score", "writing score"]].mean(axis=1)
    df_copy.drop(['math score', 'reading score', 'writing score','gender__female','lunch__free/reduced','pass','target_name'], axis=1,inplace=True)
    return df_copy
    

# Classification

class Classification:
    def __init__(self, model_cl, X_train_cl, y_train_cl, X_test_cl, y_test_cl, y_pred_cl, y_score_cl,classes):
        self.model_cl = model_cl
        self.X_train_cl = X_train_cl 
        self.y_train_cl = y_train_cl
        self.X_test_cl = X_test_cl
        self.y_test_cl = y_test_cl
        self.y_pred_cl = y_pred_cl
        self.y_score_cl = y_score_cl
        self.classes = classes
        
    def to_dict(self):
        return {
            'X_train': self.X_train_cl,
            'y_train': self.y_train_cl,
            'X_test': self.X_test_cl,
            'y_test': self.y_test_cl,
            'model': self.model_cl,
            'y_pred': self.y_pred_cl,
            'y_score':self.y_score_cl,
            'classes': self.classes
        }
    
def model_cl_history(df, model_cl):

    Y = df['pass']
    
    columns_to_drop = ['pass', 'target_name', 'id','math score','reading score','writing score']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns_to_drop, axis=1)
    
    X = pd.get_dummies(X)

    X_train_cl, X_test_cl, y_train_cl, y_test_cl  = train_test_split(X, Y, test_size=0.15, random_state=42)

    if model_cl == SVC:
        model_instance = model_cl(probability=True)

    else:
        model_instance = model_cl()
        
    model_instance.fit(X_train_cl, y_train_cl)

    y_pred_cl = model_instance.predict(X_test_cl)
    y_score_cl = model_instance.predict_proba(X_test_cl)[:,1]

    classes = np.unique(Y)

    classification = Classification(str(model_instance)[:-2], X_train_cl, y_train_cl, X_test_cl, y_test_cl, y_pred_cl,y_score_cl, classes)
    return classification