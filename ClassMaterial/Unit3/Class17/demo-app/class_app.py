# -*- coding: utf-8 -*-
"""
Streamlit application made for DAT-10-19
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px

# we're going to load our data set in as a FUNCTION
@st.cache
def load_data():
    df = pd.read_csv('iowa_train2.csv')
    return df

# now we'll call the function -- this cache's the values
df = load_data()

st.title("Our First Data Application")
st.write(df)

page = st.sidebar.radio('Section', ['Data Explorer', 'Model Explorer'])

@st.cache
def print_page_title(page_val):
    print(f"The value of the page is: {page_val}")

print_page_title(page)

if page == 'Data Explorer':
    
    x_axis = st.sidebar.selectbox('Choose A Value for the X-Axis', df.columns.tolist(), index=1)
    y_axis = st.sidebar.selectbox('Choose A Value for the Y-Axis', df.select_dtypes(include=np.number).columns.tolist(), index=3)
    chart_type = st.sidebar.selectbox(
            'What Type of Chart Would You Like to Create?', ['line', 'bar', 'box'])
    
    st.header(f"Chart For: {x_axis} vs {y_axis}")
    
    if chart_type == 'line':
        grouping = df.groupby(x_axis)[y_axis].mean()
        st.line_chart(grouping)
        
    elif chart_type == 'bar':
        grouping = df.groupby(x_axis)[y_axis].mean()
        st.bar_chart(grouping) 
        
    elif chart_type == 'box':
        chart = px.box(df, x=x_axis, y=y_axis)
        st.plotly_chart(chart)