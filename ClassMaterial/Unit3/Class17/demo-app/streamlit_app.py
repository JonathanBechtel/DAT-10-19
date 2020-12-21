# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:30:58 2020

@author: Jonathan
"""

import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import xgboost as xgb
from matplotlib.pyplot import style
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.pipeline import make_pipeline
from pdpbox import pdp
import plotly.express as px


style.use('ggplot')
st.title("Our First Data Application")

@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/JonathanBechtel/dat-data-demo/master/iowa_train2.csv")
    return df

@st.cache
def create_groupby_object(x_axis, y_axis):
    data = df.groupby(x_axis)[y_axis].mean()
    return data

@st.cache
def create_plotly_graph_data(x_axis, y_axis):
    data = df.groupby(x_axis)[y_axis].mean().to_frame().reset_index()
    return data

df = load_data()



page = st.sidebar.radio('Section',
                        ['Data Explorer', 'Model Explorer', 'Causal Impact'])

if page == 'Data Explorer':

    st.write(df)
    
    x_axis = st.sidebar.selectbox(
        'X-Axis',
         df.columns.tolist(),
         index=1)
    
    y_axis = st.sidebar.selectbox(
             'Y-Axis',   
            df.select_dtypes(include=np.number).columns.tolist())
    
    chart_type = st.sidebar.selectbox(
            'What Type of Chart Would You Like to Create?',
            ['line', 'bar', 'box'])
    
    st.subheader(f"Breaking Down {y_axis} by: {x_axis}")
    
    if chart_type == 'line':
        data = create_groupby_object(x_axis, y_axis)
        st.line_chart(data)
    elif chart_type == 'bar':
        data = create_groupby_object(x_axis, y_axis)
        st.bar_chart(data)
    elif chart_type == 'box':
        chart = px.box(df, x=x_axis, y=y_axis)
        st.write(chart)
        #if df[x_axis].nunique() > 8:
        #    chart.set_xticklabels(rotation=90)
        #st.pyplot(chart)
   
if page in ['Model Explorer', 'Causal Impact']:

    st.cache()
    pipe = make_pipeline(OneHotEncoder(use_cat_names=True), xgb.XGBRegressor())
    
    st.cache()
    X_train, X_val, y_train, y_val = train_test_split(df.drop('SalePrice', axis=1), df['SalePrice'], test_size=0.2, random_state=1985)    
    
if page == 'Model Explorer':
    num_rounds      = st.sidebar.number_input('Number of Boosting Rounds',
                                 min_value=100, max_value=5000, step=100)
    
    tree_depth      = st.sidebar.number_input('Tree Depth',
                                 min_value=2, max_value=8, step=1, value=3)
    
    learning_rate   = st.sidebar.number_input('Learning Rate',
                                    min_value=.001, max_value=1.0, step=.05, value=0.1)
    
    validation_size = st.sidebar.number_input('Validation Size',
                                      min_value=.1, max_value=.5, step=.1, value=0.2)
    
    random_state    = st.sidebar.number_input('Random State', value=1985)
        
    pipe[1].set_params(n_estimators=num_rounds, max_depth=tree_depth, learning_rate=learning_rate)
    
    X_train, X_val, y_train, y_val = train_test_split(df.drop('SalePrice', axis=1), df['SalePrice'], test_size=validation_size, random_state=random_state)
    
    pipe.fit(X_train, y_train)
    
    mod_results = pd.DataFrame({
            'Train Size': X_train.shape[0],
            'Validation Size': X_val.shape[0],
            'Boosting Rounds': num_rounds,
            'Tree Depth': tree_depth,
            'Learning Rate': learning_rate,
            'Training Score': pipe.score(X_train, y_train),
            'Validation Score': pipe.score(X_val, y_val)
            }, index=['Values'])
    
    st.subheader("Model Results")
    st.table(mod_results)
    

    st.subheader("Real vs Predicted Validation Values")

    chart = sns.regplot(x=pipe.predict(X_val), y=y_val)
    st.pyplot(chart.figure)
    
if page == 'Causal Impact':
        
        pipe.fit(X_train, y_train)
    
        # transform the dataset for PDPBox -- necessary step
        dataset = pipe[0].fit_transform(X_train)
        columns = dataset.columns.tolist()
        pipe.fit(X_train, y_train)
    
        pdp_col = st.sidebar.selectbox(
        'Column',
         df.columns.tolist(),
         index=1)
        
        st.subheader("Choose A Column To Explore Its Impact on SalePrice")
        
        # if the column is numeric
        if is_numeric_dtype(X_train[pdp_col]):
            # create the partial dependence figure for a numeric column
            pdp_fig = pdp.pdp_isolate(
                    model=pipe[1], dataset=dataset, model_features=columns, 
                    feature=pdp_col)
            fig, axes = pdp.pdp_plot(pdp_fig, pdp_col, plot_lines=True, frac_to_plot=100)
            
            st.pyplot(fig)
            
        else:
            # grab the columns to use for one hot encoding
            cols_to_use = [col for col in columns if pdp_col in col]
            pdp_fig = pdp.pdp_isolate(
                    model=pipe[1], dataset=dataset, model_features=columns,
                    feature=cols_to_use)
            
            fig, axes = pdp.pdp_plot(pdp_fig, pdp_col, plot_lines=True, frac_to_plot=100)
            
            # format tick labels if there are more than 9 unique values
            if len(cols_to_use) > 8:
                xtick_labels = [col.split('_')[1] for col in cols_to_use]
                axes['pdp_ax'].set_xticklabels(xtick_labels, rotation='vertical');
                st.pyplot(fig)
            else:
                st.pyplot(fig)
        
    