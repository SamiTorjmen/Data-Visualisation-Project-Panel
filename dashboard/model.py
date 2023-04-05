import pandas as pd
import numpy as np

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
    