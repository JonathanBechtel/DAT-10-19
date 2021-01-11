# -*- coding: utf-8 -*-
"""
Homework 4 Application 
"""
import streamlit as st
import pandas as pd 
import numpy as np 
import xgboost as xgb 
import plotly.express as px 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
import category_encoders as ce
from sklearn.model_selection import train_test_split


#saves data in memory so doesn't load everytime we make changes 
@st.cache 
def load_data():
    df = pd.read_csv('Diabetes.csv')
    return df 
#cahces values 
df = load_data()



#Sidebar
page = st.sidebar.radio('Section', ['INTRODUCTION', 'DATA EXPLORER', 'MODEL EXPLORER']) # 'Causal Impact'])
#print(page)


#INTRO 
if page == 'INTRODUCTION': 
    st.title("PREDICTING DIABETES WITH NHANES FROM 2017")
    st.text('The NHANES program began in the early 1960s and has been conducted as a series')
    st.text('of surveys focusing on different population groups or health topics.')
    st.header('DATA AT A GLANCE')
#    cols = ['Age', 'BLOODSUGAR', 'CHOLESTEROL', 'BMI', 'WAIST', 'RACE', 'GENDER', 'DIABETES', 'SMOKE', 'HISTORY', 'INCOME']
#    st_ms = st.multiselect("Columns", df.columns.tolist(), default=cols)
    st.dataframe(df.head(50))
    st.header('MISSING VALUES')
    missing = df.isnull().sum()
    st.bar_chart(missing) 

#First tab on page 
if page == 'DATA EXPLORER':
    st.title("DATA EXPLORER")
    x_axis = st.sidebar.selectbox('CHOOSE X-AXIS', df.columns.tolist(), index=1) #What item do you want to show up first? Makes 1 default 
    y_axis = st.sidebar.selectbox('CHOOSE Y-AXIS', df.select_dtypes(include=np.number).columns.tolist(), index=3)
    graph_type = st.sidebar.selectbox('CHOOSE CHART TYPE', ['LINE', 'BAR', 'BOX'])
    st.header(f"CHART FOR: {x_axis} VS {y_axis}")
    
    if graph_type == 'LINE':
        grouping = df.groupby(x_axis)[y_axis].mean()  #Why mean? 
        st.line_chart(grouping)   
    elif graph_type == 'BAR':
        grouping = df.groupby(x_axis)[y_axis].mean()  #Why mean? 
        st.bar_chart(grouping) 
    elif graph_type == 'BOX': 
        chart = px.box(df, x=x_axis, y=y_axis)#Don't use grouping, use entire dataset
        st.plotly_chart(chart)
  
    
#Second tab on page 
if page == 'MODEL EXPLORER':
    st.title("MODEL EXPLORER")
    df = df.fillna(0)
    num_rounds = st.sidebar.number_input('Number of Boosting Rounds', min_value=100, max_value=5000, step=100)
    tree_depth = st.sidebar.number_input('Tree Depth', min_value=2, max_value=8, step=1, value=3)
    learning_rate = st.sidebar.number_input('Learning Rate', min_value=.001, max_value=1.0, step=0.5, value=0.1)
    validation_size = st.sidebar.number_input('Validation Size', min_value=.1, max_value=.5, step=.1, value=.2)
    random_state = st.sidebar.number_input('Random State', value=1985)
    
    @st.cache
    def split_data_and_fit_model(num_rounds, tree_depth, learning_rate, validation_size, random_state):
        
        pipe = make_pipeline(ce.OneHotEncoder(use_cat_names=True), xgb.XGBClassifier())
        X_train, X_val, y_train, y_val = train_test_split(df.drop('DIABETES', axis=1), df['DIABETES'], test_size=validation_size, random_state=random_state)
        pipe[1].set_params(n_estimators=num_rounds, max_depth=tree_depth, learning_rate=learning_rate)
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
        
        plot = plot_confusion_matrix(pipe, X_val, y_val)

        return mod_results , plot
            
    
    st.subheader("Model Results")
    
    mod_results, plot = split_data_and_fit_model(num_rounds, tree_depth, learning_rate, validation_size, random_state)
    st.table(mod_results)
    st.subheader('Confusion Matrix')
    st.pyplot(plot)











