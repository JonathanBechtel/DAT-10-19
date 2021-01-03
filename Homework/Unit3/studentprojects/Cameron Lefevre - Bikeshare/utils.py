# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:39:06 2020

@author: Jonathan
"""
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd


def extract_dates(df, cols=None, drop_cols=False, date_parts=None, get_history=True, history_type=['days']):
    if cols is not None:
        assert type(cols) == list, "Please make sure argument for drop_cols is a list with the column labels you would like to encode"
        date_cols = cols
    else:
        date_cols = df.select_dtypes(include='datetime64').columns.tolist()
        if not date_cols:
            raise TypeError("This dataframe does not have any datetime columns within it")
    if date_parts is not None:
        assert type(date_parts) == list, "Please make sure argument for date_parts is a list with the date attributes you want to encode"
    else:
        date_parts = ['dayofweek', 'dayofyear', 'days_in_month', 'is_leap_year', 'is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start', 'quarter', 'week', 'weekofyear', 'day', 'hour', 'minute', 'month', 'year']
    
    
    for col in date_cols:
        if hasattr(df[col], 'dt'):
            for part in date_parts:
                col_name = f"{col}_{part}"
                df[col_name] = getattr(df[col].dt, part)
                
    if get_history:
        for col in date_cols:
           if hasattr(df[col], 'dt'):
            for type_ in history_type:
                if hasattr((df[col] - df[col].min()).dt, type_):
                    col_name  = f"{col}_history_{type_}"
                    col_value = getattr((df[col] - df[col].min()).dt, type_)
                    df[col_name] = col_value
                
    if drop_cols:
        df.drop(date_cols, axis=1, inplace=True)
        return df
    else:
        return df
    
def get_val_scores(model, X, y, return_test_score=False, return_training_score=False, return_importances=False, random_state=None, randomize=True, cv=5, test_size=0.2, val_size=0.2, use_kfold=True, return_folds=False, stratify=False):
        
    if randomize:
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=True, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)
    else:
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=False)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    if use_kfold:
        val_scores = cross_val_score(model, X=X_train, y=y_train, cv=cv)
    else:
        if randomize:
            if stratify:
                X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=True)
            else:
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=True)
        else:
            if stratify:
                print("Warning! You opted to both stratify your training data and to not randomize it.  These settings are incompatible with scikit-learn.  Stratifying the data, but shuffle is being set to True")
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size,  shuffle=True)
            else:
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=False) 
        val_score = model.fit(X_train, y_train).score(X_val, y_val)
        
    if return_importances:
        if hasattr(model, 'steps'):
            try:
                feats = pd.DataFrame({
                    'Columns': X.columns,
                    'Importance': model.steps[-1][1].feature_importances_
                }).sort_values(by='Importance', ascending=False)
            except:
                model.fit(X_train, y_train)
                feats = pd.DataFrame({
                    'Columns': X.columns,
                    'Importance': model.steps[-1][1].feature_importances_
                }).sort_values(by='Importance', ascending=False)
        else:
            try:
                feats = pd.DataFrame({
                    'Columns': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            except:
                mod.fit(X_train, y_train)
                feats = pd.DataFrame({
                    'Columns': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            
    mod_scores = {}
    try:
        mod_scores['validation_score'] = val_scores.mean()
        if return_folds:
            mod_scores['fold_scores'] = val_scores
    except:
        mod_scores['validation_score'] = val_score

    if return_training_score:
        model.fit(X_train, y_train)
        mod_scores['training_score'] =  model.score(X_train, y_train)
        
    if return_test_score:
        model.fit(X_train, y_train)
        mod_scores['test_score'] =  model.score(X_test, y_test)
           
    if return_importances:
        return mod_scores, feats
    else:
        return mod_scores