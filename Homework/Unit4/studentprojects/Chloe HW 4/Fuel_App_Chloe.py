# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:06:19 2020

@author: chloe 
"""



title = 'Fuel Economy Data'



import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import xgboost as xgb
from matplotlib.pyplot import style
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.pipeline import make_pipeline
from pdpbox import pdp
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
style.use('ggplot')
st.title("Fuel Economy Data")

@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/chloemd/DAT-1019-Chloe/main/Homework/Unit4/data/fuel_eco_clean.csv")
    return df

@st.cache
def load_avgs():
    df = pd.read_csv("https://raw.githubusercontent.com/chloemd/DAT-1019-Chloe/main/Homework/Unit4/data/fuel_eco_clean_averages.csv")
    return df

@st.cache
def load_urls():
    urls = pd.read_csv("https://raw.githubusercontent.com/chloemd/DAT-1019-Chloe/main/Homework/Unit4/data/fuelgov_img_urls.csv")
    return urls

@st.cache
def create_groupby_object(x_axis, y_axis):
    data = df.groupby(x_axis)[y_axis].mean()
    return data

@st.cache
def create_plotly_graph_data(x_axis, y_axis):
    data = df.groupby(x_axis)[y_axis].mean().to_frame().reset_index()
    return data

@st.cache
def return_stats(df, y_cols):
    avg_stats = df.groupby(['Year', 'Make', 'Model'])[y_cols]
    avg_stats = avg_stats.mean().reset_index()
    return avg_stats

st.cache()
df = load_data()
st.cache() 
urls = load_urls()
st.cache()
preds = load_avgs()



page = st.sidebar.radio('Section',
                        ['Explore Data', 'Explore Model', 'Browse Vehicles', 'Compare Vehicles'])



if page == 'Explore Data':
    st.markdown("""
                <style>
                table td:nth-child(1) {
                    display: none
                    }
                table th:nth-child(1) {
                    display: none
                    }
                </style>
                """, unsafe_allow_html=True)
    
    
    x_cols = ['Year', 'Class', 'Drive', 'Transmission', 'Fuel Type', 'Engine Cylinders', 'Engine Displacement']
    y_cols = ['Combined MPG (FT1)', 'City MPG (FT1)', 'Fuel Economy Score',
              'Highway MPG (FT1)', 'Annual Fuel Cost (FT1)', 'Tailpipe CO2 (FT1)']
    
    st.cache()
    avg_stats = return_stats(df, y_cols)
    
    sort_table = st.sidebar.selectbox('Sort Data',
                        ['Ascending', 'Descending'])
    sort_bool = sort_table == 'Ascending'
    
    x_axis = st.sidebar.selectbox(
        'X-Axis',
         x_cols,
         index=0)
    
    y_axis = st.sidebar.selectbox(
             'Y-Axis',
             y_cols
            
            )
    
    st.cache()
    data = df.groupby(x_axis)[y_axis].mean().reset_index()
    
    want_min = ['Annual Fuel Cost (FT1)', 'Tailpipe CO2 (FT1)']
    
    
    
    if y_axis in want_min:
        if sort_bool:
            st.subheader(f"5 Best Vehicles for {y_axis}")
        else:
            st.subheader(f"5 Worst Vehicles for {y_axis}")
    elif y_axis not in want_min:
        if sort_bool:
            st.subheader(f"5 Worst Vehicles for {y_axis}")
        else:
            st.subheader(f"5 Best Vehicles for {y_axis}")
            
    st.table(avg_stats[avg_stats[y_axis] > 0].sort_values(y_axis, ascending=sort_bool).iloc[:5][['Year', 'Make', 'Model', y_axis]])
    
    chart_type = st.sidebar.selectbox(
            'Select a chart type:',
            ['Line', 'Bar', 'Box', 'Violin'])
    
    
    st.subheader(f"Breaking Down {y_axis} by: {x_axis}")
    if chart_type == 'Line':
        chart = px.line(data, x=x_axis, y=y_axis)
        st.write(chart)
    elif chart_type == 'Bar':
        chart = px.bar(data, x=x_axis, y=y_axis)
        st.write(chart)
    elif chart_type == 'Box':
        chart = px.box(df, x=x_axis, y=y_axis)
        st.write(chart)
    elif chart_type == 'Violin':
        chart = px.violin(df, x=x_axis, y=y_axis)
        st.write(chart)

if page == 'Explore Model':
    st.cache()
    cols_to_drop = ['City MPG (FT1)', 'Unrounded City MPG (FT1)','City MPG (FT2)','Unrounded City MPG (FT2)',
 'Highway MPG (FT1)','Unrounded Highway MPG (FT1)','Highway MPG (FT2)','Unrounded Highway MPG (FT2)',
 'Unadjusted City MPG (FT1)','Unadjusted Highway MPG (FT1)','Unadjusted City MPG (FT2)',
 'Unadjusted Highway MPG (FT2)','Combined MPG (FT1)','Unrounded Combined MPG (FT1)','Combined MPG (FT2)',
 'Unrounded Combined MPG (FT2)',
 'My MPG Data','Composite City MPG','Composite Highway MPG','Composite Combined MPG','City Range (FT1)',
 'Range (FT1)','City Range (FT1)','Highway Range (FT1)','City Range (FT2)','Highway Range (FT2)',
 'Range (FT2) Clean','Save or Spend (5 Year)','Tailpipe CO2 (FT1)','Annual Fuel Cost (FT1)',
 'Annual Consumption in Barrels (FT1)','Tailpipe CO2 in Grams/Mile (FT1)','Fuel Economy Score',
 'GHG Score','City Gasoline Consumption (CD)','City Electricity Consumption',
 'Highway Gasoline Consumption (CD)','Highway Electricity Consumption','Combined Electricity Consumption',
 'Combined Gasoline Consumption (CD)','Annual Consumption in Barrels (FT1)','Annual Consumption in Barrels (FT2)',
 'Fuel Type','Fuel Type 1','Fuel Type 2','Alternative Fuel/Technology','Gas Guzzler Tax']
    st.cache()
    pipe = make_pipeline(TargetEncoder(), xgb.XGBRegressor())
    
    
    
    num_rounds      = st.sidebar.number_input('Number of Boosting Rounds',
                                 min_value=50, max_value=500, step=50, value=150)
    
    tree_depth      = st.sidebar.number_input('Tree Depth',
                                 min_value=2, max_value=6, step=1, value=5)
    
    learning_rate   = st.sidebar.number_input('Learning Rate',
                                    min_value=.001, max_value=1.0, step=.05, value=0.15)
    
    validation_size = st.sidebar.number_input('Validation Proportion',
                                      min_value=.1, max_value=.5, step=.1, value=0.2)
    
    random_state    = st.sidebar.number_input('Random State', value=2021)
    
    
    st.cache()
    X_train, X_val, y_train, y_val = train_test_split(df.drop(cols_to_drop, axis=1), df['Combined MPG (FT1)'], test_size=validation_size, random_state=random_state) 
    
    
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
 
    st.subheader("Model Results")
    st.table(mod_results)
    
    st.write('')
    st.subheader("Real vs Predicted Validation Values")
    
    
    
    
    chart = sns.regplot(x=pipe.predict(X_val), y=y_val)
    st.pyplot(chart.figure)
   
if page == 'Browse Vehicles':
    
    st.markdown("""
                <style>
                table td:nth-child(1) {
                    display: none
                    }
                table th:nth-child(1) {
                    display: none
                    }
                </style>
                """, unsafe_allow_html=True)
    car_info_cols = ['Class', 'Transmission','Combined MPG (FT1)', 'City MPG (FT1)', 'Highway MPG (FT1)', 'Annual Fuel Cost (FT1)', 'Fuel Economy Score']
    unique_years = df['Year'].unique().tolist()
    year = st.sidebar.selectbox(
        'Model Year',
        unique_years,
        index=len(unique_years)-1)
    
    make = st.sidebar.selectbox(
        'Make',
        df[df['Year'] == year]['Make'].unique().tolist(), 
        index=0)
    
    model = st.sidebar.selectbox(
        'Model',
        df[(df['Year'] == year) & (df['Make'] == make)]['Model'].unique().tolist(),
        index=0)
    
    pred = preds[(preds['Year'] == year) & (preds['Make'] == make) & (preds['Model'] == model)]
    
    
    year_avg = pred['MPG (Year Average)'].mean()
    class_avg = pred['MPG (Class Average)'].mean()
    vehicle_class = pred['Class'].iloc[0]
    make_avg = pred['MPG (Make Average)'].mean()
    vehicle_ft = pred['Fuel Type'].iloc[0]
    ft_avg = pred['MPG (Fuel Type Average)'].mean()
    
    
    url = urls[(urls['Year'] == year) & (urls['Make'] == make) & (urls['Model'] == model)]['Image Url'].item()
    st.sidebar.markdown(f"![Picture of: {year} {make} {model}]({url})", unsafe_allow_html=True)
    st.subheader(f"{year} {make} {model}")
    st.table(df[(df['Year'] == year) & (df['Make'] == make) & (df['Model'] == model)][car_info_cols])
    
    st.write("")
    with st.beta_expander(label=f"View Combined MPG Averages for {year} Vehicles"):
        
        
        average_vals = pd.DataFrame({
            vehicle_class: class_avg,
            f"{vehicle_ft} Fuel Type": ft_avg,
            f"{make}s": make_avg, 
            f'All {year} Vehicles': year_avg},
            index=['Values'])
        st.table(average_vals)
    
if page == 'Compare Vehicles':
   
    st.markdown("""
                <style>
                table td:nth-child(1) {
                    display: none
                    }
                table th:nth-child(1) {
                    display: none
                    }
                </style>
                """, unsafe_allow_html=True)
    unique_years = df['Year'].unique().tolist()
    x_cols = ['Year', 'Class', 'Drive', 'Transmission', 'Fuel Type', 'Engine Cylinders', 'Engine Displacement']
    car_info_cols = ['Class', 'Transmission','Combined MPG (FT1)', 'Annual Fuel Cost (FT1)']
    with st.beta_expander(label="Select Vehicles to Compare"):
        col1, col2 = st.beta_columns(2)
        with col1:
            v1_year = st.selectbox('Vehicle 1 Year', unique_years, index=len(unique_years)-1)
            v1_make = st.selectbox('Vehicle 1 Make', df[df['Year'] == v1_year]['Make'].unique().tolist(), index=0)
            v1_model = st.selectbox('Vehicle 1 Model', df[(df['Year'] == v1_year) & (df['Make'] == v1_make)]['Model'].unique().tolist(), index=0)
            
            
        
        with col2:
            v2_year = st.selectbox('Vehicle 2 Year', unique_years, index=len(unique_years)-2)
            v2_make = st.selectbox('Vehicle 2 Make', df[df['Year'] == v2_year]['Make'].unique().tolist(), index=0)
            v2_model = st.selectbox('Vehicle 2 Model', df[(df['Year'] == v2_year) & (df['Make'] == v2_make)]['Model'].unique().tolist(), index=0)
    
    
    col3, col4 = st.beta_columns(2)
    
    with col3:
        v1_url = urls[(urls['Year'] == v1_year) & (urls['Make'] == v1_make) & (urls['Model'] == v1_model)]['Image Url'].item()
        st.text("")
        st.text("")
        st.markdown(f"![Picture of: {v1_year} {v1_make} {v1_model}]({v1_url})", unsafe_allow_html=True)
        st.subheader(f"{v1_year} {v1_make} {v1_model}")
        st.table(df[(df['Year'] == v1_year) & (df['Make'] == v1_make) & (df['Model'] == v1_model)][car_info_cols])
        
    with col4:
        v2_url = urls[(urls['Year'] == v2_year) & (urls['Make'] == v2_make) & (urls['Model'] == v2_model)]['Image Url'].item()
        st.text("")
        st.text("")
        st.markdown(f"![Picture of: {v2_year} {v2_make} {v2_model}]({v2_url})", unsafe_allow_html=True)
        st.subheader(f"{v2_year} {v2_make} {v2_model}")
        st.table(df[(df['Year'] == v2_year) & (df['Make'] == v2_make) & (df['Model'] == v2_model)][car_info_cols])
        