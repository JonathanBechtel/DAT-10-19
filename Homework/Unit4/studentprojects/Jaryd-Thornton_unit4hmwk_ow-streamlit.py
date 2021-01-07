#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicle analysis project
Created on Jan 5 2021
@author: Jaryd
heroku url: https://shielded-woodland-03438.herokuapp.com/
"""

import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from pdpbox import pdp
import plotly.express as px
import requests

# format tables to not show index
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

@st.cache
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/database.csv',
                     usecols=['Make', 'Model', 'Year', 'Class', 'Drive', 'Transmission',
                              'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
                              'Supercharger', 'Fuel Type 1', 'Annual Fuel Cost (FT1)', 'Combined MPG (FT1)',
                              'Tailpipe CO2 in Grams/Mile (FT1)'], 
                     dtype={'Year': 'str'})
    return df 

df = load_data()

# rename columns to easier use
df = df.rename({'Tailpipe CO2 in Grams/Mile (FT1)':'CO2 Grams/Mile',
                'Combined MPG (FT1)': 'MPG',
                'Fuel Type 1': 'Fueled by',
                'Annual Fuel Cost (FT1)': 'Annual Fuel Cost'}, axis=1)

# cylinders and displacment should only be in int format
df['Engine Cylinders'] = df['Engine Cylinders'].fillna(0)
df['Engine Cylinders'] = df['Engine Cylinders'].astype(int)
df['Engine Displacement'] = df['Engine Displacement'].fillna(0)
df['Engine Displacement'] = df['Engine Displacement'].astype(int)

# Merge air intake systems
df['Turbocharger'] = df['Turbocharger'].fillna("F")
df['Supercharger'] = np.where(df['Supercharger'] == "S", "T", "F")
df['Turbocharged'] = np.where((df['Turbocharger'] == "T") | (df['Supercharger'] == "T"), "True", "False")

# cut dataset to desired cols
X_cols = ['Fueled by', 'Make', 'Year', 'Class', 'Drive', 'Transmission',
          'Engine Cylinders', 'Engine Displacement', 'Turbocharger', 'Supercharger', 'Turbocharged',
          'Annual Fuel Cost', 'MPG', 'CO2 Grams/Mile']

y_cols = ['Annual Fuel Cost', 'MPG', 'CO2 Grams/Mile']

# chart types
charts = ['Scatter', 'Box', 'Bar', 'Line']

# create pages
page = st.sidebar.radio('Section',
                        ['Summary', 'Interactive explorer', 'ML: Vehicle selector', 'ML: Engine specs'])

if page  == 'Summary':
    
    st.title('Vehicle Analysis')
    st.text('The following report analyzes the fuel efficency of vehicles and engine types.')
    st.header('Summary')
    st.subheader('Miles Per Gallon (MPG) and CO2 emissions')
    
    chart_mpg_co2 = px.scatter(df, x='MPG', y='CO2 Grams/Mile',
    title='Correlation between MPG and CO2')
    st.write(chart_mpg_co2)
    
    df2 = df.copy()
    df2 = df2.rename({'Year': 'Year Manufactured'}, axis=1)
    year_co2 = df2.groupby('Year Manufactured')['CO2 Grams/Mile'].mean().reset_index()
    chart_year_co2 = px.line(year_co2, x='Year Manufactured', y='CO2 Grams/Mile',
    title='CO2 emissions by year manufactured')
    st.write(chart_year_co2)
    
    mpg_year = df.groupby('Year')['MPG'].mean().reset_index()
    mpg_year.columns=['Year Manufactured', 'MPG']
    px.line(mpg_year, x='Year Manufactured', y='MPG',
    title='MPG by Year Manufactured')
    
    st.text("The improvement in MPG and CO2 levels is largely due to the rise in electric vehicles.")
    st.text("However there have still been improvments made with fossil fueled engines.")
    
    st.subheader('Excluding electric vehicles...')
    
    no_elec = df.loc[df['Fueled by'] != 'Electricity']
    mpg_year_fuel = no_elec.groupby(['Year', 'Fueled by'])['MPG'].mean().reset_index()
    mpg_year_fuel.columns=['Year Manufactured','Fueled by', 'MPG']
    chart_mpg_year_fuel = px.line(mpg_year_fuel, x='Year Manufactured', y='MPG',
    title='MPG by Year Manufactured', color='Fueled by')
    st.write(chart_mpg_year_fuel)
    
    co2_year_fuel = no_elec.groupby(['Year', 'Fueled by'])['CO2 Grams/Mile'].mean().reset_index()
    co2_year_fuel.columns=['Year Manufactured','Fueled by', 'CO2 Grams/Mile']
    chart_co2_year_fuel = px.line(co2_year_fuel, x='Year Manufactured', y='CO2 Grams/Mile',
    title='Co2 by Year Manufactured', color='Fueled by')
    st.write(chart_co2_year_fuel)
    
    drive = no_elec.groupby('Drive')['MPG'].mean().reset_index().sort_values(by='MPG')
    drive_chart = px.bar(drive, x='Drive', y='MPG',
    title="MPG by drive type")
    st.write(drive_chart)
    
    st.subheader('Annual costs')
    
    costs_fuel = df.groupby('Fueled by')['Annual Fuel Cost'].mean().reset_index().sort_values(by='Annual Fuel Cost')
    costs_fuel.columns=['Fuel Type', 'Annual Cost']
    chart_costs_fuel = px.bar(costs_fuel, x='Fuel Type', y='Annual Cost',
    title='Annual cost by fuel type')
    st.write(chart_costs_fuel)
    
    costs_engine = df.groupby('Engine Displacement')['Annual Fuel Cost'].mean().reset_index().sort_values(by='Engine Displacement')
    chart_costs_engine = px.line(costs_engine, x='Engine Displacement', y='Annual Fuel Cost',
    title='Annual cost by engine displacement')
    st.write(chart_costs_engine)
  
    chart_costs_turbo = px.box(df, x='Turbocharged', y='Annual Fuel Cost',
    title='Annual cost for Turbocharged engines')
    st.write(chart_costs_turbo) 
    st.text("note that 'Turbocharged' refers to engines with either a Supercharger or Turbocharger")
    

if page  == 'Interactive explorer':
    
    st.title('Explore vehicle data')

    x_axis = st.sidebar.selectbox(
        'X axis',
         X_cols,
         index=12)
    
    y_axis = st.sidebar.selectbox(
        'Y axis',
         y_cols,
         index=2)
    
    chart_type = st.sidebar.selectbox(
        'Chart Type',
         charts,
         index=0)
    
    if chart_type == 'Line':
        grouping = df.groupby(x_axis)[y_axis].mean().reset_index()
        st.header(f"Line chart: {x_axis} & {y_axis}")
        chart = px.line(grouping, x=x_axis, y=y_axis)
        st.write(chart)
    elif chart_type == 'Bar':
        grouping = df.groupby(x_axis)[y_axis].mean().reset_index()
        st.header(f"Bar chart: {x_axis} & {y_axis}")
        chart = px.bar(grouping, x=x_axis, y=y_axis)
        st.write(chart)
    elif chart_type == 'Box':
        st.header(f"Box plot: {x_axis} & {y_axis}")
        chart = px.box(df, x=x_axis, y=y_axis)
        st.write(chart)
    elif chart_type == 'Scatter':
        st.header(f"Scatter plot: {x_axis} & {y_axis}")
        chart = px.scatter(df, x=x_axis, y=y_axis)
        st.write(chart)
        
    st.header('Raw dataset')
    st.write(df)

                   
if page  == 'ML: Vehicle selector':
    
    st.title("Machine Learning - Vehicle Selector")
    st.text('Regression model designed to predict Annual Fuel costs, MPG and CO2 of vehicles.')
    st.text('Choose your desired vehicle from the left sidebar.')
    
    # create input variable for user to select
    predict = st.sidebar.selectbox('Predict..', y_cols)

    #@st.cache
    # set up pipeline with Ordinal encoding (has worked best for this dataset)
    # modified params to resolve overfitting within 1%
    mod1 = xgb.XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=100, colsample_bylevel=0.25)
    pipe = make_pipeline(OrdinalEncoder(), mod1)

    # set X & y using only basic vehicle details
    X = df[['Fueled by', 'Make', 'Model', 'Year', 'Class', 'Drive', 'Transmission',
          'Engine Cylinders', 'Engine Displacement', 'Turbocharger', 'Supercharger', 'Turbocharged']]
    
    y = df[predict]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
    
    make       = st.sidebar.selectbox('Make of vehicle', sorted(X_test['Make'].unique()))
    year       = st.sidebar.selectbox('Year manurfactured', sorted(X_test.loc[X_test['Make'] == make]['Year'].unique(),reverse=True))
    model      = st.sidebar.selectbox('Model of vehicle', sorted(X_test.loc[(X_test['Make'] == make) &\
                                                                     (X_test['Year'] == year)]['Model'].unique()))
        
    st.subheader(f"Currently predicting: {predict} for {make} {model} {year}")
    
    # get photo of vehicle
    URL = "https://rapidapi.p.rapidapi.com/api/Search/ImageSearchAPI"
    HEADERS = {
    'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
    'x-rapidapi-key': "0a51b88a70msh8e87cbfc6f681c1p12f1e7jsnc8ce67a7bef6"
    }

    q = make + ' ' + model + ' ' + year + ' side'
    page_number = 1
    page_size = 1
    auto_correct = True
    safe_search = True

    querystring = {"q": q,
               "pageNumber": page_number,
               "pageSize": page_size,
               "autoCorrect": auto_correct,
               "safeSearch": safe_search}

    response = requests.get(URL, headers=HEADERS, params=querystring).json()

    photo_url = ()
    for image in response["value"]:
        image_url = image["url"]
        photo_url = image_url
    
    # display photo
    st.markdown(f"![Alt Text]({photo_url})")
    
    # fit model
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    
    pred_results = pd.DataFrame()
    pred_results['true'] = y_train
    pred_results['predicted'] = pipe.predict(X_train)
    pred_chart = px.scatter(pred_results, x='true', y='predicted', trendline='ols',
                              title='True v predicted')
    
    # narrow down to make model and year results
    X_test['Pred'] = pipe.predict(X_test)
    results = X_test.loc[(X_test['Make'] == make) & (X_test['Year'] == year) & (X_test['Model'] == model)]
    
    st.subheader(predict + ' = ' + str(np.around(results.groupby(['Make', 'Year', 'Model'])['Pred'].mean().values.max())))
    
    results_table = results.groupby(['Make', 'Year', 'Model'])['Pred'].mean().reset_index()
    results_table['Accuracy'] = score
    results_table.index=['Result']
    results_table.rename({'Pred': predict}, axis=1, inplace=True)
    
    st.table(results_table)
    st.write(pred_chart)
    
   
    # feature importance 
    pipe.fit(X_train, y_train)
    feature_names = pipe.named_steps["ordinalencoder"].get_feature_names()
    feat = pd.DataFrame(
        {'Feature': X_train.columns,
         'Impact': pipe.steps[1][1].feature_importances_}).sort_values(by='Impact',
                                                                      ascending=False)
    feat = feat.head(5)
    feat['Impact'] = pd.Series(["{0:.2f}%".format(val * 100) for val in feat['Impact']], index=feat.index)
    
    # get top two features
    feat1 = feat.iloc[0:1,:1].values
    feat1 = [item for sublist in feat1 for item in sublist]
    feat1 = ''.join(feat1)
    
    feat2 = feat.iloc[1:2,:1].values
    feat2 = [item for sublist in feat2 for item in sublist]
    feat2 = ''.join(feat2)
    
    st.subheader(f"Top 5 features used in ML model for predicting {predict}")
    st.table(feat)
    
    st.subheader("Partial dependence for the top 2 features")
    st.text(f"{feat1} & {feat2}")
    
    # partial dependence
    pdp_1 = pdp.pdp_isolate(
    model=pipe[1], dataset=pipe[0].transform(X_train), model_features=X_train.columns.tolist(), 
    feature=feat1, cust_grid_points = pipe[0].transform(X_train)[feat1].tolist())
    fig, axes = pdp.pdp_plot(pdp_1, feat1, plot_lines=True, frac_to_plot=100)
    st.write(fig)
    
    pdp_2 = pdp.pdp_isolate(
    model=pipe[1], dataset=pipe[0].transform(X_train), model_features=X_train.columns.tolist(), 
    feature=feat2, cust_grid_points = pipe[0].transform(X_train)[feat2].tolist())
    fig2, axes = pdp.pdp_plot(pdp_2, feat2, plot_lines=True, frac_to_plot=100)
    st.write(fig2)
    
    st.subheader(f"Corelation betweeen {feat1} & {feat2} on {predict}")
    
    gbm_inter = pdp.pdp_interact(
    model=pipe[1], dataset=pipe[0].transform(X_train), model_features=pipe[0].get_feature_names(), 
    features=[feat1, feat2])
    fig3, axes = pdp.pdp_interact_plot(
    gbm_inter, [feat1, feat2], x_quantile=True, plot_type='grid', plot_pdp=True)
    
    st.write(fig3)
    
if page == 'ML: Engine specs':
    
    st.title("Machine Learning - Engine Specs")
    st.text('Regression model designed to predict Annual Fuel costs, MPG and CO2 of engine types.')
    st.text('Choose your desired engine specifications from the left sidebar.')
    
    predict = st.sidebar.selectbox('Predict..', y_cols)

    df = df.loc[~pd.isnull(df.Drive)]
    #@st.cache
    # modified params from previous ML model to improve overfitting using multiple engine examples
    mod1 = xgb.XGBRegressor(max_depth=2,learning_rate=0.05,n_estimators=750
                        , colsample_bylevel=0.3, colsample_bytree=0.5)
    pipe = make_pipeline(OrdinalEncoder(), mod1)
    
    drivetrain = sorted(df['Drive'].unique())
    any_drive  = ['Any']
    drivetrain = any_drive + drivetrain 
    drive = st.sidebar.selectbox('Drivetrain', drivetrain)

    if drive != 'Any':
        
        engine_c = st.sidebar.selectbox('Number of cylinders', sorted(df.loc[df['Drive'] == drive]['Engine Cylinders'].unique()))
        engine_d = st.sidebar.selectbox('CC displacement', sorted(df.loc[(df['Drive'] == drive) & (df['Engine Cylinders'] == engine_c)]['Engine Displacement'].unique()))
        df = df.loc[(df['Engine Cylinders'] == engine_c) & (df['Engine Displacement'] == engine_d) & (df['Drive'] == drive)]
    

    else:
        engine_c = st.sidebar.selectbox('Number of cylinders', sorted(df['Engine Cylinders'].unique()),index=6)
        engine_d = st.sidebar.selectbox('CC displacement', sorted(df.loc[df['Engine Cylinders'] == engine_c]['Engine Displacement'].unique()),index=2)
        df = df.loc[(df['Engine Cylinders'] == engine_c) & (df['Engine Displacement'] == engine_d)]


    st.subheader(f'Currently predicting: {predict} for an engine with\
                 {engine_c} cyclinders, {engine_d} CC displacement and {drive} drivetrain')
    
    X = df[['Fueled by', 'Make', 'Model', 'Year', 'Class', 'Drive', 'Transmission',
            'Engine Cylinders', 'Engine Displacement', 'Turbocharger', 'Supercharger', 'Turbocharged']]
    
    y = df[predict]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
    
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    
    X_test['Pred'] = pipe.predict(X_test)
    
    results_table = pd.DataFrame({
    'Drivetrain': drive,
    'Cylinders': engine_c,
    'CC displacement': engine_d,
    'Pred': X_test['Pred'].mean(),
    'Accuracy': score
    }, index=['Results'])
    results_table.rename({'Pred': predict}, axis=1, inplace=True)
    
    pred_results = pd.DataFrame()
    pred_results['true'] = y_train
    pred_results['predicted'] = pipe.predict(X_train)
    pred_chart = px.scatter(pred_results, x='true', y='predicted', trendline='ols',
                              title='True v predicted')

    # results
    st.subheader(predict + ' = ' + str(np.around(X_test['Pred'].mean())))
    st.table(results_table)
    st.write(pred_chart)
    
    
    # feature importance 
    feature_names = pipe.named_steps["ordinalencoder"].get_feature_names()
    feat = pd.DataFrame(
        {'Feature': X_train.columns,
         'Impact': pipe.steps[1][1].feature_importances_}).sort_values(by='Impact',
                                                                      ascending=False)
    feat = feat.head(5)
    feat['Impact'] = pd.Series(["{0:.2f}%".format(val * 100) for val in feat['Impact']], index=feat.index)
    
    # get top two features
    feat1 = feat.iloc[0:1,:1].values
    feat1 = [item for sublist in feat1 for item in sublist]
    feat1 = ''.join(feat1)
    
    feat2 = feat.iloc[1:2,:1].values
    feat2 = [item for sublist in feat2 for item in sublist]
    feat2 = ''.join(feat2)
    
    st.subheader(f"Top 5 features used in ML model for predicting {predict}")
    st.table(feat)
    
    st.subheader("Partial dependence for the top 2 features")
    st.text(f"{feat1} & {feat2}")
    
    # partial dependence
    pdp_1 = pdp.pdp_isolate(
    model=pipe[1], dataset=pipe[0].transform(X_train), model_features=X_train.columns.tolist(), 
    feature=feat1, cust_grid_points = pipe[0].transform(X_train)[feat1].tolist())
    fig, axes = pdp.pdp_plot(pdp_1, feat1, plot_lines=True, frac_to_plot=100)
    st.write(fig)
    
    pdp_2 = pdp.pdp_isolate(
    model=pipe[1], dataset=pipe[0].transform(X_train), model_features=X_train.columns.tolist(), 
    feature=feat2, cust_grid_points = pipe[0].transform(X_train)[feat2].tolist())
    fig2, axes = pdp.pdp_plot(pdp_2, feat2, plot_lines=True, frac_to_plot=100)
    st.write(fig2)
    
    st.subheader(f"Corelation betweeen {feat1} & {feat2} on {predict}")
    
    gbm_inter = pdp.pdp_interact(
    model=pipe[1], dataset=pipe[0].transform(X_train), model_features=pipe[0].get_feature_names(), 
    features=[feat1, feat2])
    fig3, axes = pdp.pdp_interact_plot(
    gbm_inter, [feat1, feat2], x_quantile=True, plot_type='grid', plot_pdp=True)
    
    st.write(fig3)
    
    
    
    
    
    
    
    
    
    