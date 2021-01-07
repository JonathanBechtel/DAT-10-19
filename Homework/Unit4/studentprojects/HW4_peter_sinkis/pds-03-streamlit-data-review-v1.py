#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: PRSmb
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime

# from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder
import xgboost as xgb

import plotly.express as px
from pdpbox import pdp, info_plots

# Setting - Draft sets to import to support scoring

all_draftaholics_sets = [
    'AER','MM3','AKH','HOU','XLN',
    'IMA','UST','RIX','A25','DOM',
    'BBD','M19','GRN','UMA','RNA',
    'WAR','MH1','M20','ELD','MB1',
    'THB','IKO','CUB','M21','2XM',
    'AKR','ZNR','KLR','CMR']

dict_setname_code = {
    'Mystery Booster':'MB1',
    'MTG Arena Draft Cube':'CUB',
    'Commander Legends':'CMR',
    'Double Masters':'2XM',
    'Amonkhet Remastered':'AKR',
    'Core Set 2019':'M19',
    'Kaladesh Remastered':'KLR',
    'Zendikar Rising':'ZNR',
    'Core Set 2020':'M20',
    'Ikoria: Lair of Behemoths':'IKO',
    'Ixalan':'XLN',
    'Core Set 2021':'M21',
    'Ravnica Allegiance':'RNA',
    'Ultimate Masters':'UMA',
    'Guilds of Ravnica':'GRN',
    'Modern Horizons':'MH1',
    'Masters 25':'A25',
    'Amonkhet':'AKH',
    'Battlebond':'BBD',
    'Throne of Eldraine':'ELD',
    'Theros Beyond Death':'THB',
    'Dominaria':'DOM',
    'Iconic Masters':'IMA',
    'War of the Spark':'WAR',
    'Modern Masters 2017 Edition':'MM3',
    'Unstable':'UST',
    'Rivals of Ixalan':'RIX',
    'Hour of Devastation':'HOU',
    'Aether Revolt':'AER',
    }

dict_code_setname={
    'MB1': 'Mystery Booster',
    'CUB': 'MTG Arena Draft Cube',
    'CMR': 'Commander Legends',
    '2XM': 'Double Masters',
    'AKR': 'Amonkhet Remastered',
    'M19': 'Core Set 2019',
    'KLR': 'Kaladesh Remastered',
    'ZNR': 'Zendikar Rising',
    'M20': 'Core Set 2020',
    'IKO': 'Ikoria Lair of Behemoths',
    'XLN': 'Ixalan',
    'M21': 'Core Set 2021',
    'RNA': 'Ravnica Allegiance',
    'UMA': 'Ultimate Masters',
    'GRN': 'Guilds of Ravnica',
    'MH1': 'Modern Horizons',
    'A25': 'Masters 25',
    'AKH': 'Amonkhet',
    'BBD': 'Battlebond',
    'ELD': 'Throne of Eldraine',
    'THB': 'Theros Beyond Death',
    'DOM': 'Dominaria',
    'IMA': 'Iconic Masters',
    'WAR': 'War of the Spark',
    'MM3': 'Modern Masters 2017 Edition',
    'UST': 'Unstable',
    'RIX': 'Rivals of Ixalan',
    'HOU': 'Hour of Devastation',
    'AER': 'Aether Revolt',
    }

# Settting - Plotly template

plotly_style_temp = "ggplot2"


# Helper functions to display data

def combine_column_values_as_str(df,col_list,seperator="|"):
    # For each column in a list combine the values for a given row and return as a list of strings
    series_to_combine=[]
    num_series = len(col_list)

    for col in col_list:
        series_to_combine.append(df[col].tolist())
    
    num_rows = len(series_to_combine[0])
    
    working_list = []
    for i in range(0,num_rows):  
        my_str = ''
        for j in range(0,num_series):
                my_str = my_str + str(series_to_combine[j][i]) + seperator
        working_list.append(my_str[:-1])

    return working_list    

@st.cache
def cards_with_gt1_elo(df_card_scores):
    df = df_card_scores.name.value_counts().reset_index().copy()
    df.rename({'index':'name','name':'count'},axis=1,inplace=True)
    df_target_cards = df[df['count'].apply(lambda x: True if x > 1 else False)]
    list_target_cards = combine_column_values_as_str(df_target_cards,['name','count'])
    return list_target_cards    

@st.cache
def column_string_match(df,match_string):
    match_list = [i for i in df.columns.tolist() if match_string.upper() in i.upper()]
    return match_list

# Helper functions to set up data

def sum_columns_starting_with(df, col_name_str):
    col_name_match_len = len(col_name_str)
    col_list = [i for i in df.columns.tolist() if str(i)[0:col_name_match_len] == col_name_str]
    sum_values = []
    for i in col_list:
        if sum_values==[]:
            sum_values = df[i].tolist()
        else:
            sum_values = [a+b for a,b in zip(sum_values, df[i])]
    return sum_values

def max_columns_starting_with(df, col_name_str):
    col_name_match_len = len(col_name_str)
    col_list = [i for i in df.columns.tolist() if str(i)[0:col_name_match_len] == col_name_str]
    max_values = []
    for i in col_list:
        if max_values==[]:
            max_values = df[i].tolist()
        else:
            max_values = [max(a,b) for a,b in zip(max_values, df[i])]
    return max_values

@st.cache(allow_output_mutation=True)
def import_card_data_features(card_data_path="./resources/cards.csv"):
    # Import card data, sets up features, and returns a data frame with that information
    # Data sourced from: https://mtgjson.com/api/v5/AllPrintingsCSVFiles.zip
    card_dtypes = {
        'colors':np.object,
        'faceConvertedManaCost':np.object,
        'flavorText': np.object,
        'frameEffects': np.object,
        'leadershipSkills': np.object,
        'name': np.object,
        'text': np.object,
    }
    
    df_base = pd.read_csv(card_data_path,dtype=card_dtypes,low_memory=False)
    
    # Remove key rows which have data we won't use
    df_base = df_base[(df_base.isOnlineOnly == 0)]
    df_base = df_base[(df_base.isOversized == 0)]
    df_base = df_base[(df_base.isPromo == 0)]
    df_base = df_base[~(df_base.layout == 'vanguard')]
    
    
    # Keep fields likely to support data feature build
    df = df_base[[
        'index',
        'id',
        'colorIdentity',
        'colorIndicator',
        'colors',
        'convertedManaCost',
        'faceConvertedManaCost',
        'faceName',
        'flavorText',
        'hand',
        'hasAlternativeDeckLimit',
        'isOnlineOnly',
        'isOversized',
        'isPromo',
        'isReprint',
        'isReserved',
        'isStarter',
        'isTextless',
        'keywords',
        'layout',
        'leadershipSkills',
        'life',
        'loyalty',
        'manaCost',
        'multiverseId',
        'name',
        'number',
        'otherFaceIds',
        'power',
        'printings',
        'rarity',
        'setCode',
        'side',
        'subtypes',
        'supertypes',
        'text',
        'toughness',
        'type',
        'types',
        'uuid',
        'variations',
        'watermark'
    ]].copy()
    
    ############################################################
    ############################################################
    
    # Create unique row per card name / allowing for multiple faces (i.e. names may be duplicated)
    # 'side' needs to be filled in or groupby portion of statement doesn't work properly
    df['side'].fillna('normal',inplace=True)
    df['name_row'] = df.sort_values(by='id',ascending=True).groupby(['name','side']).cumcount() + 1
    df = df[(df['name_row'] == 1)]
    
    # Flag double layout cards
    df['double_layout'] = 1
    df['double_layout'].where(df['layout'].isin(['transform','split','adventure','modal_dfc','flip','aftermath','meld']),0,inplace=True)
    
    
    ############################################################
    ############################################################
    
    
    # Add in mana cost counts
    df['manaCost_NA'] = df.manaCost.isna()*1 # Column to flag NA values for manaCost
    df['manaCost'].fillna('{none}',inplace=True) # Use '{none}' in lower case, since all other manaCost letters in upper case.
    df['manaCost_Generic_count'] = np.where(df.manaCost.str.contains('\{[\d]+?\}'),df.manaCost.str.extract('\{([\d]+?)\}',expand=False),0)
    df['manaCost_W_count'] = df.manaCost.str.count('{W}')
    df['manaCost_U_count'] = df.manaCost.str.count('{U}')
    df['manaCost_B_count'] = df.manaCost.str.count('{B}')
    df['manaCost_R_count'] = df.manaCost.str.count('{R}')
    df['manaCost_G_count'] = df.manaCost.str.count('{G}')
    df['manaCost_C_count'] = df.manaCost.str.count('{C}')
    df['manaCost_WP_count'] = df.manaCost.str.count('{W/P}')
    df['manaCost_UP_count'] = df.manaCost.str.count('{B/P}')
    df['manaCost_BP_count'] = df.manaCost.str.count('{U/P}')
    df['manaCost_RP_count'] = df.manaCost.str.count('{R/P}')
    df['manaCost_GP_count'] = df.manaCost.str.count('{G/P}')
    df['manaCost_H_WU_count'] = df.manaCost.str.count('{W/U}')
    df['manaCost_H_UB_count'] = df.manaCost.str.count('{U/B}')
    df['manaCost_H_BR_count'] = df.manaCost.str.count('{B/R}')
    df['manaCost_H_RG_count'] = df.manaCost.str.count('{R/G}')
    df['manaCost_H_GW_count'] = df.manaCost.str.count('{G/W}')
    df['manaCost_H_WB_count'] = df.manaCost.str.count('{W/B}')
    df['manaCost_H_UR_count'] = df.manaCost.str.count('{U/R}')
    df['manaCost_H_BG_count'] = df.manaCost.str.count('{B/G}')
    df['manaCost_H_RW_count'] = df.manaCost.str.count('{R/W}')
    df['manaCost_H_GU_count'] = df.manaCost.str.count('{G/U}')
    df['manaCost_H_2W_count'] = df.manaCost.str.count('{2/W}')
    df['manaCost_H_2U_count'] = df.manaCost.str.count('{2/U}')
    df['manaCost_H_2B_count'] = df.manaCost.str.count('{2/B}')
    df['manaCost_H_2R_count'] = df.manaCost.str.count('{2/R}')
    df['manaCost_H_2G_count'] = df.manaCost.str.count('{2/G}')
    df['manaCost_X_count'] = df.manaCost.str.count('{X}')
    df['manaCost_Y_count'] = df.manaCost.str.count('{Y}')
    df['manaCost_Z_count'] = df.manaCost.str.count('{Z}')
    df['manaCost_Snow_count'] = df.manaCost.str.count('{S}')
    df['manaCost_HW_count'] = df.manaCost.str.count('{HW}')
    
    ############################################################
    ############################################################
    
    # OneHot Encode all keywords in the data, and add some other info related to keywords
    df['keywords_NA'] = df.keywords.isna()*1
    df.keywords.fillna('{none}',inplace=True)
    df['keywords_count'] = [len(i) for i in df.keywords.str.split(',').tolist()] * np.where(df.keywords_NA,0,1)
    
    
    all_keywords = df.keywords.str.split(",").tolist()
    unique_keywords = []
    
    for i in all_keywords:
        for j in i:
            if j != '{none}':
                j.capitalize()
                unique_keywords.append(j)
    unique_keywords = set(unique_keywords)
    unique_keywords = list(unique_keywords)
    unique_keywords.sort()
    
    for keyword in unique_keywords:
        col_name = 'keyword_' + keyword.replace(' ','_')
        df[col_name] = df.keywords.str.contains(keyword) * 1
    
    ############################################################
    ############################################################
    
    
    # Other effects ############################################################
    
    df['text_NA'] = df.keywords.isna()*1
    df.text.fillna('{none}',inplace=True)
    
    # Drawing cards (generally a benefit)
    df_draw_cards = df.text.str.extract('[Dd]raw(?!\s[Ss]tep)\s(.*?)card?')
    df_draw_cards.rename({0:'extract_text'},axis=1,inplace=True)
    df_draw_cards.fillna(0,inplace=True)
    
    cond = [
        df_draw_cards['extract_text'].str[0] == 'a',
        df_draw_cards['extract_text'].str[0:3] == 'two',
        df_draw_cards['extract_text'].str[0:5] == 'three',
        df_draw_cards['extract_text'].str[0:4] == 'four',
        df_draw_cards['extract_text'].str[0:4] == 'five',
        df_draw_cards['extract_text'].str[0:3] == 'six',
        df_draw_cards['extract_text'].str[0:5] == 'seven',
        df_draw_cards['extract_text'].str[0:5] == 'eight',
        df_draw_cards['extract_text'].str[0:4] == 'nine',
        df_draw_cards['extract_text'].str[0:5] == 'half X',
        df_draw_cards['extract_text'].str[0:0] == 'X'
    ]
    
    output = [1,2,3,4,5,6,7,8,9,15,20]
    
    df['effect_draw_cards'] = np.select(cond,output,default=0)
    
    # Beneficial discards (i.e. apply to opponent / target player - since generally you choose your opponent, unless you're drawing cards)
    df_extract = (df.text.str.contains('[Tt]arget\s(opponent|player).*[Dd]iscards\s.*?card?',case=False))
    df['effect_discard_target_player'] = df_extract*1
    
    # Discard own cards (cost)
    df_extract = (df.text.str.contains('Discard\s.*?card?',case=False))
    df['effect_discard_own_cards'] = df_extract*1
    
    # Loot ability dummy variable
    df_extract = (df.text.str.contains('[Dd]raw a card, then discard a card',case=False))
    df['effect_loot'] = df_extract*1
    
    # Destroy effects / exile effects
    # Note - are just treating destroy and exile as identical effects for now, for the purpose of getting a model working
    #        ideally would split these up, and allow for some more nuance
    # Note - need to go back and check interaction of nonland and permanent to make sure it is handle properly
    df_extract_nonland = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*target.*?nonland(?=\.|\s)?')*1
    df_extract_permanent = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*target.*(?<!nonland\s)permanent(?=\.|\s)?')*1
    
    df_extract = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*target.*artifact?(\.|\s)')*1
    df['effect_destroy_artifact'] = df_extract + df_extract_nonland + df_extract_permanent
    
    df_extract = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*target.*creature?(\.|\s)')*1
    df['effect_destroy_creature'] = df_extract + df_extract_nonland + df_extract_permanent
    
    df_extract = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*target.*?(?<!is)(?<!non)land(?!walk)(?=\.|\s)?')*1
    df['effect_destroy_land'] = df_extract + df_extract_permanent
    
    df_extract = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*target.*?enchantment(?=\.|\s)?')*1
    df['effect_destroy_enchantment'] = df_extract + df_extract_nonland + df_extract_permanent
    
    df_extract = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*target.*?planeswalker(?=\.|\s)?')*1
    df['effect_destroy_planeswalker'] = df_extract + df_extract_nonland + df_extract_permanent
    
    # Destroying 'all' creatures
    df_extract = df.text.str.contains('([Dd]estroy|[Ee]xile)\s.*all.*creatures(?=\.|\s)?')*1
    df['effect_destroy_all_creatures'] = df_extract
    
    # Deals damage effects
    # Focus on damage to others, and excludes comabat damage to... triggers
    df_extract = df.text.str.contains('(deals)+\s[\dX]*.*(?!combat\s)(damage)\sto(?!\syou)')
    df['effect_deals_damage'] = df_extract*1
    
    # Counter spell effects
    df_extract = df.text.str.contains('[Cc]ounter.*spell')
    df['effect_counter_target_spell'] = df_extract*1
    
    # Enters the battlefield effect
    df_extract = df.text.str.contains('[Ee]nter(s)?\sthe\sbattlefield')
    df['effect_enter_the_battlefield'] = df_extract*1
    
    df_extract = df.text.str.contains('[Ee]nter(s)?\sthe\sbattlefield.*[Ss]acrifice\sit')
    df['effect_enter_the_battlefield_sacrific_it'] = df_extract*1
    
    # Activate ability as an effect
    df_extract = df.text.str.count('.*:.*')
    df['effect_has_activated_ability'] = df_extract*1
    
    ############################################################
    ############################################################
    
    # Set up base lines for efficiency metrics
    
    # Power + Toughness 
    # P+T Clean up power
    df.power.fillna('{none}',inplace=True)
    df['power_clean'] = 0
    df['power_clean'] = np.where(df.power.str.contains('\D(?<![{noe}])'),1,0)
    df['power_clean'] = [max(i/2,1) for i in df['convertedManaCost'].tolist()] * np.float64(df['power_clean'])
    df['power_clean'] = np.where(df['power_clean']==0,df['power'],df['power_clean'])
    df['power_clean'] = np.where(df['power_clean']=='{none}',0,df['power_clean'])
    
    # P+T Clean up toughness
    
    df.toughness.fillna('{none}',inplace=True)
    df['toughness_clean'] = 0
    df['toughness_clean'] = np.where(df.toughness.str.contains('\D(?<![{noe}])'),1,0)
    df['toughness_clean'] = [max(i/2,1) for i in df['convertedManaCost'].tolist()] * np.float64(df['toughness_clean'])
    df['toughness_clean'] = np.where(df['toughness_clean']==0,df['toughness'],df['toughness_clean'])
    df['toughness_clean'] = np.where(df['toughness_clean']=='{none}',0,df['toughness_clean'])
    
    # P+T Calculation
    df['power_plus_toughness'] = np.float64(df['power_clean']) + np.float64(df['toughness_clean'])
    df['power_plus_toughness']  = np.float64(df['power_plus_toughness'])
    
    # Count keywords and effects
    
    df['keyword_count'] = sum_columns_starting_with(df,'keyword_')
    df['effect_count'] = sum_columns_starting_with(df,'effect_')
    
    # Calculate effieciency ratings
    df['efficiency_power'] = np.where(df['convertedManaCost'].gt(0),np.float64(df['power_clean'])/df['convertedManaCost'],0)
    df['efficiency_toughness'] = np.where(df['convertedManaCost'].gt(0),np.float64(df['toughness_clean'])/df['convertedManaCost'],0)
    df['efficiency_p_plus_t'] = np.where(df['convertedManaCost'].gt(0),np.float64(df['power_plus_toughness'])/df['convertedManaCost'],0)
    df['efficiency_keywords'] = np.where(df['convertedManaCost'].gt(0),np.float64(df['keyword_count'])/df['convertedManaCost'],0)
    df['efficiency_effects'] = np.where(df['convertedManaCost'].gt(0),np.float64(df['effect_count'])/df['convertedManaCost'],0)
   
   
    df['efficiency_power'] = np.where((df.convertedManaCost==0) & ~(df['type'].str.contains("Land")),12,df['efficiency_power'])
    df['efficiency_toughness'] = np.where((df.convertedManaCost==0) & ~(df['type'].str.contains("Land")),12,df['efficiency_toughness'])
    df['efficiency_p_plus_t'] = np.where((df.convertedManaCost==0) & ~(df['type'].str.contains("Land")),12,df['efficiency_p_plus_t'])
    df['efficiency_keywords'] = np.where((df.convertedManaCost==0) & ~(df['type'].str.contains("Land")),12,df['efficiency_keywords'])
    df['efficiency_effects'] = np.where((df.convertedManaCost==0) & ~(df['type'].str.contains("Land")),12,df['efficiency_effects'])
    
    df['efficiency_max'] = max_columns_starting_with(df,'efficiency_')
    
    # Set index to speed up name merge later on
    df['name_1'] = df['name']
    df.set_index('name_1',inplace=True)
    
    print("Loaded card features.")
    
    return df

@st.cache
def get_draft_scores(set_list):
    # This function gets the draft scores from the draftaholicsanonymous site
    # based on an overarching list of sets (typicall using the standard 3 lettter code used to
    # define Magic:the Gathering sets.)
    # It only needs to be run when you want to update the scores you have on file.
    for i in set_list:
        req_path = "https://apps.draftaholicsanonymous.com/p1p1/" + i + "/results?ajax"
        results = requests.get(req_path)
        
        if results.status_code != 200:
            print(f'Was unable to retrieve data for {i}')
        else:
            results_json = json.loads(results.text)
            output_path = './draft-scores/scores_' + i + '.txt'
            with open(output_path,'w') as outfile:
                json.dump(results_json['data'],outfile)


@st.cache()
def import_card_draft_scores(set_list,draft_score_file_path='./draft-scores/'):
    # Note - only need to run get draft scores if we want to refresh data, otherwise
    # assume to work from the underlying files (so can just load)

    def load_draft_scores_to_pandas(set_list,draft_score_file_path='./draft-scores/'):
        # This function loads the files generated by get_draft_scores into a pandas data frame.
        first_load = True
        for i in set_list:
            if first_load == True:
                first_load = False
                load_path = draft_score_file_path + '/scores_' + i + '.txt' 
                df = pd.read_json(load_path)
                print(f"loaded {i} : {df['id'].count()}")
            else:
                load_path = draft_score_file_path + '/scores_' + i + '.txt' 
                df_next = pd.read_json(load_path)
                df = df.append(df_next,ignore_index=True)
                print(f"loaded {i} : {df_next['id'].count()} {df['id'].count()}")
        return df


    # get_draft_scores(all_draftaholics_sets)

    # This function adds score transforms into the pandas dataframe.
    
    df_scores = load_draft_scores_to_pandas(set_list,draft_score_file_path)
    
    # Some different elo score conversions
    
    df_scores['elo_log'] = np.log(df_scores['elo']) # Log conversion of elo
    df_scores['elo_range_all'] = df_scores['elo'].max() - df_scores['elo'].min()
    
    # Score relative to overall list of cards
    # Note - Adding 1 to the top, and 2 to the denominator to present any score being precisely 0 or 1
    df_scores['elo_relative_all'] = (df_scores['elo'] - df_scores['elo'].min() + 1) / (df_scores['elo_range_all'] + 2) 
    
    # Score relative to all cards in a given set
    
    elo_range_set = (df_scores.groupby('set_name')['elo'].max() - df_scores.groupby('set_name')['elo'].min()).reset_index()
    elo_range_set.rename({'elo':'elo_range_set'},axis=1,inplace=True)
    df_scores = df_scores.merge(elo_range_set,how='left',on='set_name')
    
    # Set relative scores for all cards in a given set
    elo_range_min = df_scores.groupby('set_name')['elo'].min().reset_index() 
    elo_range_min.rename({'elo':'elo_set_min'},axis=1,inplace=True)
    df_scores = df_scores.merge(elo_range_min,how='left',on='set_name')
    df_scores['elo_relative_set'] = (df_scores['elo'] - df_scores['elo_set_min'] + 1) / (df_scores['elo_range_set'] + 2) 
        
    # Drop unwanted columns for analysis
    columns_to_drop = [
        # Items from website that aren't necessary
        'image_small',
        'image',
        'image_large',
        'back_image_small',
        'back_image',
        'exclude_from_p1p1',
        # Items to exclude as placeholders while building data
        'elo_range_all',
        'elo_range_set',
        'elo_set_min',
        ]
    
    df_scores.drop(columns_to_drop,axis=1,inplace=True)
    
    # Create the unique scores as well
    df_scores.back_name.fillna('{none}',inplace=True) # Do this to support combining draft scores later
    df_scores_unique = df_scores.groupby(['name','back_name'])[['elo','elo_log','elo_relative_all','elo_relative_set']].mean().reset_index()
    df_scores_unique.rename({'name':'front_name'},axis=1,inplace=True)
    df_scores_unique['name'] = np.where(df_scores_unique.back_name=='{none}',df_scores_unique.front_name,df_scores_unique.front_name + ' // ' + df_scores_unique.back_name)
    df_scores_unique['name_1'] = df_scores_unique['name']
    df_scores_unique.set_index('name_1',inplace=True)
    
    print ("Loaded draft scores.")
    
    return df_scores, df_scores_unique

@st.cache
def add_scores_to_features(df_card_features, df_scores_unique):
    df = df_card_features.merge(df_scores_unique,on='name_1',how='left').copy()
    df.drop('name_y',axis=1,inplace=True)
    df.rename({'name_x':'name'},axis=1,inplace=True)
    print("Merged scores and features")
    return df



############

# Functions to support getting data set up for model build

def model_generate_training_data_sets(df_card_scores,validation_perc=0,testing_set_list=['ZNR'],random_state=None):
    # If validation_perc set to 0 returns 2 data sets (training and test)
    # If validation_perc set between 0 and 1 returns 3 sets (training, validaiton, test)
    
    # Drop all references to elo_relative_all - they need to be recalculated for model training
    df = df_card_scores.copy()
    df.drop('elo_relative_all',axis=1,inplace=True)
    
    # Separate out testing scores
    test_flag = df.set_name.apply(lambda x: True if dict_setname_code[x] in testing_set_list else False )
    test_score_raw = df[test_flag==True].copy()
    test_score_raw['elo_range_all'] = test_score_raw['elo'].max() - test_score_raw['elo'].min()
    test_score_raw['elo_relative_all'] = (test_score_raw['elo'] - test_score_raw['elo'].min() + 1) / (test_score_raw['elo_range_all'] + 2) 
    test_score_raw.drop('elo_range_all',axis=1,inplace=True)
    testing_scores = test_score_raw.copy()
    
    # Exclude test data
    df = df[~(test_flag)].copy()
    
    # Split scores
    if validation_perc == 0:
        train_score_raw = df
    else:
        val_score_raw = df.sample(frac=validation_perc,random_state=random_state)
        train_score_raw = df[df.id.apply(lambda x: False if x in val_score_raw['id'].tolist() else True)]
        
    # Update ELO_Relative_All
    train_score_raw = train_score_raw.copy()
    train_score_raw['elo_range_all'] = train_score_raw['elo'].max() - train_score_raw['elo'].min()
    train_score_raw['elo_relative_all'] = (train_score_raw['elo'] - train_score_raw['elo'].min() + 1) / (train_score_raw['elo_range_all'] + 2) 
    train_score_raw.drop('elo_range_all',axis=1,inplace=True)
    training_scores = train_score_raw.copy()
    
    if validation_perc > 0:
        val_score_raw = val_score_raw.copy()
        val_score_raw['elo_range_all'] = val_score_raw['elo'].max() - val_score_raw['elo'].min()
        val_score_raw['elo_relative_all'] = (val_score_raw['elo'] - val_score_raw['elo'].min() + 1) / (val_score_raw['elo_range_all'] + 2) 
        val_score_raw.drop('elo_range_all',axis=1,inplace=True)
        validation_scores = val_score_raw.copy()
    
    if validation_perc == 0:
        return training_scores, testing_scores
    else:
        return training_scores, validation_scores, testing_scores

# Function to generate unique scores within training / test / validation data
def create_unique_scores(df_input):
    # Create the unique scores as well
    df_input.back_name.fillna('{none}',inplace=True) # Do this to support combining draft scores later
    df_scores_unique = df_input.groupby(['name','back_name'])[['elo','elo_log','elo_relative_all','elo_relative_set']].mean().reset_index().copy()
    df_scores_unique.rename({'name':'front_name'},axis=1,inplace=True)
    df_scores_unique['name'] = np.where(df_scores_unique.back_name=='{none}',df_scores_unique.front_name,df_scores_unique.front_name + ' // ' + df_scores_unique.back_name)
    df_scores_unique['name_1'] = df_scores_unique['name']
    df_scores_unique.set_index('name_1',inplace=True)
    df_output = df_scores_unique.copy()
    return df_output    

# Function to link training / test / validation data to scores
def link_scores_to_features(features, scores):
    df = features.merge(scores,on='name_1',how='inner').copy()
    df.drop('name_y',axis=1,inplace=True)
    df.rename({'name_x':'name'},axis=1,inplace=True)
    return df

def model_cleanup_columns(scores):
    # Drops all columns that will definitely not be used in model build
    df = scores.copy()
    cols_to_drop = [
        'index','id','flavorText','hand','hasAlternativeDeckLimit','isOnlineOnly','isOversized',
        'isPromo','isReprint','isReserved','isStarter','isTextless','keywords','layout',
        'leadershipSkills','life','manaCost','multiverseId','number','otherFaceIds','printings',
        'setCode','side','uuid','variations','watermark','name_row',
        ]
    
    for col in cols_to_drop:
        df.drop(col,axis=1,inplace=True)
        
    return df

def model_gen_x_y(scores,y_target,cols_to_drop=[],cols_to_keep=[]):
    df = scores.copy()
    y = df[y_target]
    elo_cols = column_string_match(df,'elo')
    
    # Always drop alternate elo definitions
    for col in elo_cols:
        df.drop(col,axis=1,inplace=True)
    
    # Specific columns to be excluded
    if cols_to_drop:
        for col in cols_to_drop:
             df.drop(col,axis=1,inplace=True)
    
    # Specific columns to keep
    if cols_to_keep:
        df = df[cols_to_keep].copy()
    
    x = df.copy()
    
    return x,y


def model_fit(
        train_x,train_y,
        val_x,val_y,
        num_rounds,tree_depth, learning_rate, colsample_bytree, alpha):

    
    # Constructing this in a pipeline for future use    
    pipe = make_pipeline(OrdinalEncoder(),xgb.XGBRegressor())
    
    pipe[1].set_params(
        n_estimators=num_rounds, 
        max_depth=tree_depth, 
        learning_rate=learning_rate,
        colsample_bytree=colsample_bytree,
        alpha=alpha,
        # enable_categorical=True # don't need this since now utilising ordinal encoder
        )
    
    pipe.fit(train_x,train_y)
    training_score = pipe.score(train_x, train_y)
    val_score = pipe.score(val_x, val_y)
    
    mod_results = pd.DataFrame({
        'Train Size': train_x.shape[0],
        'Validation Size': val_x.shape[0],
        'Boosting Rounds': num_rounds,
        'Tree Depth': tree_depth,
        'Learning Rate': learning_rate,
        'Columns Sampled': colsample_bytree,
        'Alpha (L1 Regularization)': alpha,
        'Training Score': training_score,
        'Validation Score': val_score,
        }, index=['Values'])

    # This outputs the real vs predicted validation values, and returns them as a chart        
    pred_results = pd.DataFrame()
    pred_results['true'] = val_y
    pred_results['predicted'] = pipe.predict(val_x)
    plotly_chart = px.scatter(pred_results, x='true', y='predicted', trendline='ols', template=plotly_style_temp)
    
    return mod_results, plotly_chart

############

# Function to support parameter sweep
def model_fit_param_sweep (
                            train, val,
                            num_rounds,tree_depth, colsample_bytree):                
    # Provide lists of params to test
    
    pipe = make_pipeline(OrdinalEncoder(),xgb.XGBRegressor())
    
    # handle different elo types
    elo_types = column_string_match(df_card_scores,'elo')
    
    round_count = 0
    param_sweep_outputs =[['iteration',
                           'elo_type',
                           'boosting_rounds',
                           'tree_depth',
                           'learning_rate',
                           'column_sample_proportion',
                           'alpha',
                           'training_score',
                           'validation_score'
                           ]]
    
    # set up specific training and validation data
    for elo in elo_types:

        train_x,train_y = model_gen_x_y(train,elo)    
        val_x,val_y = model_gen_x_y(val,elo)
 

        for nr in num_rounds:
            for td in tree_depth:
                for cs in colsample_bytree:
                    round_count+=1
                    
                    pipe[1].set_params(
                        n_estimators=nr, 
                        max_depth=td, 
                        learning_rate=0.1,
                        colsample_bytree=cs,
                        alpha=1,
                        )
                    
                    pipe.fit(train_x,train_y)
                    training_score = pipe.score(train_x, train_y)
                    val_score = pipe.score(val_x, val_y)    
                    param_sweep_outputs.append(
                                                [round_count,
                                                    elo,
                                                    nr,
                                                    td,
                                                    0.1,
                                                    cs,
                                                    1,
                                                    training_score,
                                                    val_score                                                            
                                                ]
                                                )
                    print(f"Completed iteration round {round_count:,}")
    print("Completed all iterations")
    
    # Set up file name - added time paremater to ensure that a file is always written.
    now = datetime.now()  
    current_time = now.strftime("%y%m%H%M%S")
    csv_file = 'param_sweep_outputs' + current_time + '.csv'
              
    df = pd.DataFrame(param_sweep_outputs[1:],columns=param_sweep_outputs[0])
    df.to_csv('./' + csv_file,index=False)
    print(f"{csv_file} written")
    
    return df

@st.cache
def load_sweep_results(path):
    
    df = pd.read_csv(path)
    
    return df
    

############
def build_final_model(train_x, train_y,
                      test_x, test_y,
                      elo_type,
                      num_rounds,
                      tree_depth,
                      learning_rate,
                      colsample_bytree,
                      alpha,
                      ):   
    
    # Initiaise pipeline
    pipe = make_pipeline(OrdinalEncoder(),xgb.XGBRegressor())

    pipe[1].set_params(
        n_estimators=num_rounds, 
        max_depth=tree_depth, 
        learning_rate=learning_rate,
        colsample_bytree=colsample_bytree,
        alpha=alpha,
        )
                    
    pipe.fit(train_x,train_y)
    training_score = pipe.score(train_x, train_y)    
    test_score = pipe.score(test_x, test_y)
    
    # Note - returning model pipeline for use in partial dependency analysis
    
    return training_score, test_score, pipe


def get_ordinal_enc_vals(ordenc, feature):
    # Returns a dictionary containing the mappings
    mapping_dicts = [i for i in ordenc.mapping if i['col']==feature][0]['mapping']
    
    
    # Get the ordinal values and what they represent
    key_list = mapping_dicts.index.tolist()
    value_list = mapping_dicts.tolist()
    
    ord_mapping_series={} # Set up an empty dictionary
    
    for i,j in zip(key_list,value_list):            
        ord_mapping_series[j]=i
    
    return ord_mapping_series

############





# Load up all data for initial work

df_card_features = import_card_data_features()
df_card_scores, df_scores_unique = import_card_draft_scores(all_draftaholics_sets)
df_raw = add_scores_to_features(df_card_features,df_scores_unique)






st.title("Magic: The Gathering - Draft Score Predictions")
page = st.sidebar.radio("Sections",
                        ['Background',
                         'Data Sets Loaded', 
                         'Data Feature Explorer',
                         'Model Explorer',
                         'Parameter Sweep',
                         'Final Model Fit',
                         'Model Applications',
                         ])

if page == "Background":

    '''
    ## Project Overview
    Drafting is a fun format of Magic: the Gathering where players need to pick (draft) cards,
    from booster packs that are passed from one player to another. In these situations players
    have developed a variety of ways to shortcut and get a sense of the relative game strength
    of these cards. One of these wasy is known as "Pack 1 Pick 1". 
    
    **Pack 1 Pick 1:** This means the first card you
    would pick from the first booster you open. Players use this type of judgement
    since it means you don't need to make any allowance for any consideration of cards picked so far.
    
    A website known as [draftaholicsanonymous](https://www.draftaholicsanonymous.com/) have a webapp
    where players of the game can judge between two cards as to which one they would take pack 1 pick 1. The webapp
    then combines the picks from many different players to build an ongoing [elo score](https://en.wikipedia.org/wiki/Elo_rating_system)
    to rate the cards.
    
    The draftaholics team were kind enough to let me use their data for this project.
    
    ## What we're going to do
    This little webapp will give us a chance to explore this data, as well as look
    at whether the Pack 1 Pick 1 elo scores are amenable to an XGBoost machine learning approach.
    
    ## Section Overview
    
    ### Background
    This page. Provides an overview of the project.
    
    ### Data Sets Loaded
    Provides an overview of the two main data sets used in the project. Including aspects such as
    how they were developed, a discussion of some of the data features developed, and a view
    of the data sets themselves.
    
    ### Data Feature Explorer
    This gives the ability to compare explore various aspects of the data.
    
    ### Model Explorer
    This section supports the initial model build and review.
    
    ### Parameter Sweep
    This section looks at what happens when the model parameters are varied amongst
    relevant values. Note that these results are precalculated as a file and loaded into the model.
    
    ### Final Model Fit
    Produces the final model, and looks at model drivers and outliers.
    
    ### Model Applications
    Looks at potential practical uses of the outcomes of this model.
    
    '''

elif page == 'Data Sets Loaded':
    '''
    ## Data Set Overview
    There are two data sets, that are combined into a third to support this project.
    
    I will refer to these data sets as:
    * Card Data Features
    * Card Draft Scores
    * Combined Modelling Data
    
    '''
    
    st.subheader("Card Data Features")
    '''
    The first 50 rows of our card features.
    '''
    st.write(df_card_features.head(50))
    
    '''
    #### Comments
    * There are a very wide variety of features used. A full listing is shown below.
    * They fall into two categories:
        * The information that was included in the data set by default (things like colorIdentity / convertedManaCost)
        * Data features that were constructed
    
    #### Constructed data features
    All the fields after *double_layout* are constructed.
    
    ##### manacost_
    These were constructed based on the manaCost column. This is the cost used to play the spell.
    Generally more expensive cards will have bigger impact on the game. The restrictions on playing
    cards with particular mana are implied by the letters in bracers({}). 
    
    As such the manacost_ fields represent different counts of these letters, so they can be used by
    the model to split up the data.
    
    ##### keyword_
    Keywords are a rules shorthand used to demonstrate different effects in the game. As these are consolidated
    in the keywords column.
    
    These columns split up the keywords, and apply a dummy variable / onehot encoding style to them, so
    they are available for the model to use.
    
    
    ##### effect_
    These represent various effects cards can have on the game based on the content of the
    *text* field.
    
    These are only an initial select, however I've tried to identify common effects
    that I would expect to impact card power level. They are coded by using regular
    expressions to identify specific phrases or patterns.
    
    
    ##### efficiency_
    In general the less mana you pay for a given effect, the better it is for you in the game,
    since you then have more things you can do. This is a key component in judging cards with 
    similar effects to I wanted to give the model something to split on.
    
    These are set up as {some measure of gamplay effect}/{convertedManaCost}; this also makes some allowances
    for cases where the convertedManaCost is zero as these can also be powerful effects.


    #### Card Data Features columns
    This is a list of all the columns available to support modelling.
    '''  
    
    st.write(df_card_features.columns.tolist())
    
    st.subheader('Card Scores')
    st.write("The first 50 rows of our card scores.")
    st.write(df_card_scores.head(50))
    
    '''
    #### Card Scores Overview
    The card scores data is substantially simpler than our card data features.
    
    It covers 29 Magic: the Gathering sets, and in raw form is 9,153 cards. This
    does include a number of duplicates, which are created whenever a card has been 
    released (i.e. reprinted) in more than one set.
    
    The key manipulations made to this data are:
        
    * Removing null fields, and creating a name field that is consistent with the Card Data Features data set.
    
    * Adding in some different definitions of the elo score to give our model some 
    more normalised views.
    
    **elo_log:** Taking the natural log using np.log to get a smaller numeric range of values
    
    **elo_relative_all:** The relative position from lowest to highest comparing across every set
    
    **elo_relative_set:** The relative position of an elo value within a specific set
    
    #### Card Scores colums
    '''
        
    st.write(df_card_scores.columns.tolist())

    st.subheader('Combined Modelling Data')
    st.write("The first 10 rows of our combined data.")
    st.write(df_card_scores.head(10))
    
    '''
    #### 
    This combines the scores data in with our total set of all data. Further manipulations will be necessary
    to ensure that the data is suitable to support modelling.
    
    Essentially it just adds the elo columns to the card data features. 
    It is important to note a few things:
    
    * A card may have two faces (front and back), in these cases the same score is assigned to both sides of the card.
    This could create some odd effects in the model, given that these double sided cards will implicitly be over represented
    in our training and (potentially) test data sets. They have been flagged as such though.
    
    * Given the number of sets and the existence of repeats in the scoring data, a single name can have
    several different elo scores associated with it, as these are specific to a given set. In these instances
    as a starting point the elo for any card that exists across multiple sets has had been averaged to give
    a unique score for each name.
    
    * It will be important to exclude from the x-variables elo scores that are not being used as targets in a 
    particular round, as these would understandable provide the best estimate of an elo score.
    
    * There may be columns that it will be necessary to exclude (e.g. flavorText) as these only have aesthetic
    impacts on the game rather than any interaction with gameplay itself.
    
    '''
    
elif page == 'Data Feature Explorer':
    
    # Create collapsible sections for different types of data exploration
    with st.beta_expander(label="Explore value counts of data features", expanded=False):
        de1_data_set = st.radio("Select a data set:", ["Card Data Features","Card Draft Scores","Combined Modelling Data"],index=1,key='de1')
        
        # Set the options for the select box
        sb_options = ['Select a data set']
        if de1_data_set == "Card Data Features":
            sb1_options = df_card_features.columns.tolist()
        elif de1_data_set== "Card Draft Scores":
            sb1_options = df_card_scores.columns.tolist()
        elif de1_data_set == "Combined Modelling Data":
            sb1_options = df_raw.columns.tolist()
        
        de1_data_feature = st.selectbox("Select a data feature",sb1_options)
        
        if de1_data_set == "Card Data Features":
            de1_value_check = df_card_features[de1_data_feature].nunique()
        elif de1_data_set== "Card Draft Scores":
            de1_value_check = df_card_scores[de1_data_feature].nunique()
        elif de1_data_set == "Combined Modelling Data":
            de1_value_check = df_raw[de1_data_feature].nunique()
        
        if de1_value_check > 50:
            st.write("There are more than 50 values for this feature, please try a different data exploration option.")
        else:
            col1, col2 = st.beta_columns([1,2])
            if de1_data_set == "Card Data Features":
                with col1:  
                    st.write(df_card_features[de1_data_feature].value_counts().reset_index().sort_values(de1_data_feature,ascending=False))
                with col2:
                    chart_data = df_card_features[de1_data_feature].value_counts().reset_index()
                    st.bar_chart(chart_data,use_container_width=True)
            elif de1_data_set == "Card Draft Scores":
                with col1:
                    st.write(df_card_scores[de1_data_feature].value_counts().reset_index().sort_values(de1_data_feature,ascending=False))
                with col2:
                    chart_data = df_card_scores[de1_data_feature].value_counts().reset_index()
                    st.bar_chart(chart_data,use_container_width=True)
            elif de1_data_set == "Combined Modelling Data":
                with col1: 
                    st.write(df_raw[de1_data_feature].value_counts().reset_index().sort_values(de1_data_feature,ascending=False))
                with col2:
                    chart_data = df_raw[de1_data_feature].value_counts().reset_index()
                    st.bar_chart(chart_data,use_container_width=True)            

    with st.beta_expander(label="Explore features of a specific card", expanded=False):
        de2_data_set = st.radio("Select a data set:", ["Card Data Features","Card Draft Scores","Combined Modelling Data"],index=1,key='de2')
        
        sb2_options = ['Select a dataset to get a list of cards.']
        
        # Clean updata display so set is also displayed
        if de2_data_set == "Card Data Features":
            sb2_options = combine_column_values_as_str(df_card_features,['name','setCode'])
        elif de2_data_set== "Card Draft Scores":
            sb2_options = combine_column_values_as_str(df_card_scores,['name','set_name'])
        elif de2_data_set == "Combined Modelling Data":
            sb2_options = combine_column_values_as_str(df_raw,['name','setCode'])
        
        de2_card_name = st.selectbox("Select a card name",sb2_options)        
        
        # Clean up output so can be used to source data
        de2_set_id = de2_card_name[de2_card_name.find("|")+1:]
        de2_card_name = de2_card_name[:de2_card_name.find("|")]

        
        if de2_data_set == "Card Data Features":
            de2_temp_df = df_card_features[(df_card_features.name==de2_card_name) & (df_card_features.setCode==de2_set_id)].T.copy() 
            st.table(de2_temp_df)
        elif de2_data_set== "Card Draft Scores":
            de2_temp_df = df_card_scores[(df_card_scores.name==de2_card_name) & (df_card_scores.set_name==de2_set_id)].T.copy() 
            st.table(de2_temp_df)
        elif de2_data_set == "Combined Modelling Data":
            de2_temp_df = df_raw[(df_raw.name==de2_card_name) & (df_raw.setCode==de2_set_id)].T.copy() 
            st.table(de2_temp_df)
    
    with st.beta_expander(label="Explore cards with multiple elo scores",expanded=False):
       sb3_options = cards_with_gt1_elo(df_card_scores)
       sb3_options_len = len(sb3_options)
       st.write(f"There are {sb3_options_len} cards with multiple elo scores.")
       
       sb3_elo_types = column_string_match(df_card_scores,'elo')
       sb3_elo_selected = st.radio("Select an elo type",sb3_elo_types)
       
       de3_card_name = st.selectbox("Select a card name (the number that follows after the pipe (|) are the number of scores)",sb3_options)
       de3_card_name = de3_card_name[:de3_card_name.find("|")]
       
       de3_temp_df = df_card_scores[(df_card_scores.name==de3_card_name)][['name','set_id','set_name',sb3_elo_selected]].copy() 
       st.write(de3_temp_df)
       
       de3_mean = df_card_scores[(df_card_scores.name==de3_card_name)][sb3_elo_selected].mean()
       
       de3_mean_text = "The mean {a} for {b} was {c:.3f}.".format(a=sb3_elo_selected,b=de3_card_name,c=de3_mean)
       st.write(de3_mean_text)
       
       de3_chart_x = df_card_scores[(df_card_scores.name==de3_card_name)]['set_name'].tolist() 
       de3_chart_y = df_card_scores[(df_card_scores.name==de3_card_name)][sb3_elo_selected].tolist() 
       de3_chart_y_text = ["{:.4f}".format(float(i)) for i in de3_chart_y]
       de3_fig1 = px.bar(
            df_card_scores[(df_card_scores.name==de3_card_name)],
            x='set_name',y=sb3_elo_selected,
            text=de3_chart_y_text,template=plotly_style_temp
            )
       st.plotly_chart(de3_fig1)
        
       de3_fig2 = px.box(
            df_card_scores[(df_card_scores.name==de3_card_name)],
            y=sb3_elo_selected,
            template=plotly_style_temp
            )
       st.plotly_chart(de3_fig2)
       
    # Back to top level / outside expanders    
    st.write("This section will continue to evolve as new data features are considered and addressed.")
    
elif page == 'Model Explorer':

    with st.beta_expander(label="Model building approach", expanded=False):
        '''
        ## Considerations for model build
        Things that need to be considered for the model build include:
        
       ***        
        ### Training Data
        There are a number of columns that make sense to exclude from the training data,
        and as model training progresses there may be a desire to select only a subset of columns.
        
        This will mean excluding an initial set of columns that have no game influence - i.e. that
        are driven by other factors such as whether the card can be reprinted, flavor text added
        to help provide story to players, set data that has become an arbitrary selection due to the
        data construction.
        
        The initial set of columns/x-variables are described in their own section below.
        
        
        ***
        ### Validation Data
        
        The validation portion of the data set should be able to be set between 10-20%. Scores designated
        for use in validation should be excluded from the training score calculation.
        
        K-fold validation will need to be considered carefully, potentially conducted using only a single
        instance of each card (rather than averaging out multiple scores).

        
        ***
        ### Test Data
        We need to set aside a test data set. Given the nature of Magic: the Gathering is to be
        broken up into various expansion sets, selecting specific sets will be the most appropriate
        approach. Although the control settings below will allow it, it does not make sense to allow
        Amonkhet Remastered (AKR), or Kaledesh Remastered (KLR) given that these are online versions
        of sets previously released, so would not represent a strong set of test data (while still being
        useful for model training purposes).

        In general the work below will be conducted using the most recent standard format legal set,
        **Zendikar Raising (ZNR)** as the test data set.    
        
        
        ***
        ### Elo Scores
        It will be necessary to exclude or recalculate the 'across all sets' elo scores, as by their nature
        these could include the target data set in some instances, and where this occurs would not be suitable for 
        modelling.
        
        As such certain aspects of the score assembly for the data will need to be recreated, and re-joined to 
        create a suitable training / validation / testing set up.
        
        Given that the value of cards in a set can shift dramatically depending on the other cards present,
        I expect any R-Squared values to be relatively low.
        
        ***        
        ## Approach
        
        1. Seperate training, validation, and test data sets.
        2. Run a model build with initial parameter settings.
        3. Review validation outcomes (using only a single validation set)
        5. Train and test a final model
        
        
        '''

        
    with st.beta_expander(label="Initial expectations", expanded=False):
        '''
        
        My initial expectations of this model is that it will have a relatively low r-squared, given the high
        degree of variance in the underlying data sets. That said, the nature of elo to keep things to a 
        constrained and balanced range will be helpful.
        
        '''
    
    # Outside expandable block
    
    st.subheader("Set up data for model build:")
    
    me_data_elo_types = column_string_match(df_card_scores,'elo')
    me_data_elo_selected = st.radio("Select an elo type",me_data_elo_types)
    
    
    md_data_random_state = st.number_input("Random State",min_value=1,max_value=5000,value=845)
    st.write("This changes the seed used to split up our training and validation sets.")

    md_data_val_perc = st.number_input("Validation Percentage",min_value=float(0.01),max_value=0.5,value=float(0.1),step=0.05)
    st.write("This changes the seed used to split up our training and validation sets.")
    
    md_test_set_options = [i + "|" + dict_code_setname[i]  for i in all_draftaholics_sets]
    md_data_test_sets = st.multiselect("Test data selected", options=md_test_set_options,default="ZNR|Zendikar Rising")
    st.write("As a staring point set this to ZNR|Zendikar Rising")
    md_data_test_sets = [i[i.find("|")] for i in md_data_test_sets]
    
    
    # Set up model data
    train,val,test = model_generate_training_data_sets(df_card_scores,validation_perc=md_data_val_perc,testing_set_list=md_data_test_sets,random_state=md_data_random_state)
    
    train,val,test = create_unique_scores(train),create_unique_scores(val),create_unique_scores(test)
    
    train,val,test= link_scores_to_features(df_card_features,train),link_scores_to_features(df_card_features,val),link_scores_to_features(df_card_features,test)
    
    train,val,test= model_cleanup_columns(train),model_cleanup_columns(val),model_cleanup_columns(test)
    
    train_x,train_y = model_gen_x_y(train,me_data_elo_selected)
    
    val_x,val_y = model_gen_x_y(val,me_data_elo_selected)
    
    test_x,test_y = model_gen_x_y(test,me_data_elo_selected)
        
    st.subheader("Model Build Parameters")
    
    md_num_rounds = st.number_input("Number of boosting rounds",min_value=100,max_value=2000,value=100,step=50)
    md_tree_depth = st.number_input("Tree Depth",min_value=2,max_value=10,value=3,step=1)
    md_learning_rate = st.number_input("Learning Rate",min_value=0.01,max_value=1.0,value=0.1,step=0.05)
    md_colsample_perc = st.number_input("Column Sample Percentage",min_value=0.3,max_value=1.0,value=0.5,step=0.05)
    md_alpha = st.number_input("Alpha (controls for sparse data)",min_value=0,max_value=500000,value=1,step=10000)
    
    mod_results, mod_chart = model_fit(
                                train_x,train_y,
                                val_x,val_y,
                                num_rounds=md_num_rounds,
                                tree_depth=md_tree_depth, 
                                learning_rate=md_learning_rate, 
                                colsample_bytree=md_colsample_perc, 
                                alpha=md_alpha)
    
    
    st.subheader("Model Settings and Scores")
    st.table(mod_results.T)
    
    st.subheader("Real vs. Predicated Values")
    st.text('(Validation data)')
    st.plotly_chart(mod_chart)
    
    st.subheader("Comments based on model explorer")
    
    '''
    ### Modelling elo directly
    * As would be expected increasing boosting rounds improves validation scores.
    
    * Increasing tree depth provides little improvement beyond around 3
    
    * Increasing the number of columns sampled beyond 0.5 makes little difference to the score
    
    * Increasing alpha makes minimal change
    
    '''
    
    st.subheader("Next steps")
    st.markdown("The next section will run through a parameter sweep.")
    
    # Parameter sweep function - to be run once, then has been commented out and results loaded from a file.
    # model_fit_param_sweep (
    #                         train,
    #                         val,
    #                         num_rounds=[100,200,500,800],
    #                         tree_depth=[3,5,8], 
    #                         colsample_bytree=[0.3,0.5,0.7]
    #                         )
    
elif page == 'Parameter Sweep':
    
    '''
    ## Build out parameter sweep
    
    The variables our parameter sweep will test are: 
    * Each type of elo (to see if some transform of the value can result in lower error of 
                        prediction in our validation data set)
    * A different number of boosting rounds 100,200,500,800
    * Tree depth will be set to 3,5,8
    * The learning rate will be set to 0.1, as it did not make an appreciable difference
    * A different column sample percentage 0.3,0.5,0.7
    * Alpha will not be tested as it made minimal difference to the output. It will be set to 1
    '''
    
    with st.beta_expander(label="Coding notes", expanded=False):
        '''
        * The function to run the parameter sweep is run on a refresh of the model explorer to simplify data feed.
        
        * The function produces a text file. It is this output that is loaded below.

        * the path in the code assumes that the file has a specific name and is in the same working directory as the code. If it is not it will
        need to be modified in the function. It also assumes Mac / Linux filing system.    

        * The function settings for the paremeter sweep
        
        * Random state used to generate validation data was 845
        
        '''
        
        sweep_func_code = '''
            model_fit_param_sweep (
                                train_x,train_y,
                                val_x,val_y,
                                num_rounds=[100,200,500,800],
                                tree_depth=[3,5,8], 
                                colsample_bytree=[0.3,0.5,0.7]
            )       
        '''
        
        st.code(sweep_func_code)
    
    st.subheader("Model parameter sweep results")
    
    '''
    
    The parameter sweep results are shown below. These are loaded from a file rather than run each time.
    **Iteration 130 produced the highest validation score.**
    
    '''
    
    model_sweep_results = load_sweep_results(path='./param_sweep_outputs2101173902.csv') 
    # Note that the path assumes the file is present in the same working directory as the code and has the name above
    # Change it as needed.
    
    model_sweep_results.sort_values(by='validation_score',ascending=False)
    
    ps1_chart = px.scatter(
                    model_sweep_results,
                    x='validation_score',
                    y='validation_score',
                    hover_name="iteration",
                    title="Validation Score (lowest to highest)",
                    template=plotly_style_temp
                    )
    ps1_chart.update_layout(hovermode="x")
    st.plotly_chart(ps1_chart)
    
    # Let people lookup the outcome for a specific iteration.
    ps_sb1 = st.selectbox("Select a specific iteration:",model_sweep_results['iteration'],index=129)
    
    if ps_sb1 == 130: st.markdown("The highest validation score is at iteration 130.")
    st.table(model_sweep_results[(model_sweep_results['iteration']==ps_sb1)].T)
    
    # Set up a chart of validation outcomes
    ps_sb2_options=[   
                    'elo_type',
                    'boosting_rounds',
                    'tree_depth',
                    'learning_rate',
                    'column_sample_proportion',
                    'alpha',
                ]
    
    ps_sb2 = st.radio('Box Plot X-Axis Setting', ps_sb2_options)
    
    if ps_sb2=='elo_type':
        '''
        #### Comments on elo_type
        * It appears elo_relative_set gives the strongest match in predictions
        * elo_log has the widest range
        '''
    elif ps_sb2=='boosting_rounds':
        '''
        #### Comments on boosting_rounds
        * While moving up from 100 boosting rounds shows some imporvement, there is little variance in shifting from 200 to 800.
        '''
    elif ps_sb2=='tree_depth':
        '''
        #### Comments on tree_depth
        * Slightly deeper trees at 5 levels show some improvement vs 3
        * 8 levels simply show a lot more variance for little performance gain
        '''        
    elif ps_sb2=='learning_rate':
        '''
        #### Comments on learning rate
        * Only one value was selected for learning rate, so this shows the spread of all validation scores.
        '''
    elif ps_sb2=='column_sample_proportion':
        '''
        #### Comments on column_sample_proportion
        * Uplifting column sampling beyond 30% showed little gain vs 30% (noting that 30% showed substantial additional variance)
        '''
    elif ps_sb2=='alpha':
        '''
        #### Comments on alpha
        * Only one value was selected for alpha, so this shows the spread of all validation scores.
        '''
        
    ps2_chart = px.box(model_sweep_results, x=ps_sb2, y='validation_score')
    st.plotly_chart(ps2_chart)
        
elif page == 'Final Model Fit':
    
    st.subheader("Fit model using all data, look at test scores")
    '''
    Final model parameters, based on the strongest parameter sweep outcome:
    * elo_type = 'elo_relative_set'
    * boosting_rounds = 500
    * tree_depth = 5
    * learning_rate = 0.1 
    * column_sample_proportion = 0.3
    * alpha = 1
    
    '''
    
    train,test = model_generate_training_data_sets(df_card_scores,validation_perc=0,testing_set_list=['ZNR'])
    
    train,test = create_unique_scores(train),create_unique_scores(test)
    
    train,test= link_scores_to_features(df_card_features,train),link_scores_to_features(df_card_features,test)
    
    train,test= model_cleanup_columns(train),model_cleanup_columns(test)
    
    train_x,train_y = model_gen_x_y(train,'elo_relative_set')
    
    test_x,test_y = model_gen_x_y(test,'elo_relative_set')
    
    train_score, test_score, pipe = build_final_model(
                      train_x, train_y,
                      test_x, test_y,
                      #elo_type='elo',
                      elo_type='elo_relative_set',
                      num_rounds=500,
                      tree_depth=5,
                      learning_rate=0.1,
                      colsample_bytree=0.3,
                      alpha=1,
                      )
    
    train_score_str = "The final training data score is: {:.4f}".format(train_score)
    test_score_str = "The final test data score is: {:.4f}".format(test_score)
    
    st.markdown(train_score_str)
    st.markdown(test_score_str)
    '''
    #### Comments on initial test score (It was 0.8!)
    
    **Lesson: Always look at feature importances as part of model validation review / parameter sweeps**
    
    * This test score is substantially higher than the validation score's seen earlier.
    * Checking back over the model_generate_training_data_sets there is no immediately obvious data contamination.
    * This is something worth going over an checking over in detail prior to any use of 
    this model in a practical sense. **THIS IS DEFINITELY SUSPICIOUS!**. 
    * Looking at model features made the driver immediately apparent - it was driven by rank and set_name (which should be excluded)
    * Attempted to rebuild and re-test excluding these two features
    
    #### Comments on re-tested results (it was -0.07)
    * This is clearly not suitable, removing the features did not fix things.
    
    #### Final Score: 0.34
    * This final sensible score was as a result of finding errors in the train/test data generation process
    
    * *Short version - I was missing implementing a function!*
    
    * This succeeded in resolving the key issues.
    
        
    '''
    
    st.subheader("Look at prediction outliers")
    test['predicted_elo_relative_set'] = pipe.predict(test_x)
    fm_chart_1 = px.scatter(
                            test,
                            x='elo_relative_set',
                            y='predicted_elo_relative_set',
                            # color='rarity',
                            trendline='ols',                            
                            template=plotly_style_temp,
                            hover_name='name',
                            hover_data=['rarity']
                            )    
    st.plotly_chart(fm_chart_1)
    
    '''
    Reviewing the prediction outliers a couple of things come to light.
    
    * The model seems to be most out of line at the extremes (high end and low end)
    * Dimensions to consider outliers - look at rarity in the first instance    
    '''
    
    st.subheader("Partial dependency analysis of the final model")

    feature_final_names = test_x.columns.to_list()

    features_final = pd.DataFrame({
        'Columns': feature_final_names,
        'Importance': pipe[1].feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.write(features_final)
    features_over_point5perc = len([i for i in pipe[1].feature_importances_ if i>0.005])
    features_over_point5perc_str = "There are {:,} features that have over 0.5% impact on the model.".format(features_over_point5perc)
    
    st.markdown(features_over_point5perc_str)
    
    '''
    * Many older features, that are no longer seen on cards and so are missing from the scores are showing zero influence.
    * Rarity and supertypes as the strongest drivers make sense
        * More rare cards tend to have more powerful effects
        * Different supertypes have different influences on the outcomes
    * Apart from that there seem to be small impacts expected in draft formats:
        * Generally Flying is good
        * Having defender is typically poor (so this is worth investigating, would expect it to be a negative impact)
        * X Spells tend to be valued in draft so this being ranked highly makes sense
        * enter the battlefield tends to add value to creatures vs other abilities (i.e. it will be doing a bit more)
    * It is interesting that the efficiency metric does not come into play. I would expect this to have some strong effects.
    '''
    
    
    st.subheader("Partial Dependency Plots - Rarity")
    
    # what was the additive impact of EACH unique value of rarity?  What was its marginal impact?
    pdp_rarity = pdp.pdp_isolate(
        model=pipe[1], 
        dataset=pipe[0].transform(train_x), 
        model_features=train_x.columns.tolist(), 
        feature='rarity'
    )
    fig_rarity, axes_rarity = pdp.pdp_plot(pdp_rarity, 'rarity', plot_lines=True, frac_to_plot=100)
   
    rarity_mapping = get_ordinal_enc_vals(pipe[0],'rarity')

    st.write(fig_rarity)
    fm1_col1_rarity, fm2_col2_rarity = st.beta_columns(2)
    
    with fm1_col1_rarity:
        st.write(rarity_mapping)
    
    with fm2_col2_rarity:
        st.markdown('*Rarity Count*')
        st.write(train_x['rarity'].value_counts())    
        
    '''
    *It would be nice to adjust the axes, but I ran out of time*
    
    Overall we can see that as we shift from common to rarer cards relative scores increase significantly.
    
    '''
    
    st.subheader("Partial Dependency Plots - Supertypes")

    # what was the additive impact of EACH unique value of supertype?  What was its marginal impact?
    pdp_supertypes = pdp.pdp_isolate(
        model=pipe[1], 
        dataset=pipe[0].transform(train_x), 
        model_features=train_x.columns.tolist(), 
        feature='supertypes'
    )
    fig_supertypes, axes_supertypes = pdp.pdp_plot(pdp_supertypes, 'supertypes', plot_lines=True, frac_to_plot=100)
   
    supertypes_mapping = get_ordinal_enc_vals(pipe[0],'supertypes')
    
    st.write(fig_supertypes)
    
    fm1_col1_supertypes, fm2_col2_supertypes = st.beta_columns(2)
    
    with fm1_col1_supertypes:
        st.write(supertypes_mapping)
    
    with fm2_col2_supertypes:
        st.markdown('*Supertypes Count*')
        st.write(train_x['supertypes'].value_counts())
    
    
    '''
    Given the vast number of supertypes that would sit as NaN, it is surprising
    that this has such a strong effect.
    
    Something worth investigating is going on here. Potentially worth excluding
    and seeing what comes to the surface
    
    '''

    st.subheader("Partial Dependency Plots - subtypes")
    '''
    I jump straight to subtypes here given that the intervening items
    are one hot encoded.
    '''

    pdp_subtypes = pdp.pdp_isolate(
        model=pipe[1], 
        dataset=pipe[0].transform(train_x), 
        model_features=train_x.columns.tolist(), 
        feature='subtypes'
    )
    fig_subtypes, axes_subtypes = pdp.pdp_plot(pdp_subtypes, 'subtypes', plot_lines=True, frac_to_plot=100)
   
    subtypes_mapping = get_ordinal_enc_vals(pipe[0],'subtypes')
    
    st.write(fig_subtypes)
    
    fm1_col1_subtypes, fm2_col2_subtypes = st.beta_columns(2)
    
    with fm1_col1_subtypes:
        st.write(subtypes_mapping)
    
    with fm2_col2_subtypes:
        st.markdown('*subtypes Count*')
        st.write(train_x['subtypes'].value_counts())
    
    
    '''
    There are many, many subtypes represented here
    though aside from the impact of having a subtype (1) vs not having one (the other layers)
    it doesn't seem to make much difference.
    
    A further piece of possible investigation would be to see if this is merely
    sifting some power from creature type.
    
    Another adjustment could be to utilise target encoding for supertypes / types / subtypes
    rather than ordinal encoding.
    
    '''

    st.subheader("Partial Dependency Plots - colors")
    '''
    Lastly I review the impact of colors. In theory these should be evenly
    balanced however it is curious to see if some are coming through more 
    strongly than others. Also if White will have a more limited impact.
    
    '''

    pdp_colors = pdp.pdp_isolate(
        model=pipe[1], 
        dataset=pipe[0].transform(train_x), 
        model_features=train_x.columns.tolist(), 
        feature='colors'
    )
    fig_colors, axes_colors = pdp.pdp_plot(pdp_colors, 'colors', plot_lines=True, frac_to_plot=100)
   
    colors_mapping = get_ordinal_enc_vals(pipe[0],'colors')
    
    st.write(fig_colors)
    
    fm1_col1_colors, fm2_col2_colors = st.beta_columns(2)
    
    with fm1_col1_colors:
        st.write(colors_mapping)
    
    with fm2_col2_colors:
        st.markdown('*colors Count*')
        st.write(train_x['colors'].value_counts())
    
    
    '''
    This chart appears to suggest that red is the weakest color in the draft sets.
    
    GU and other combined colours makes sense, since cards with multiple colors are
    generally more powerful than others.
    
    '''

elif page == 'Model Applications':    
    '''
    Potential applications for this model include:
        
    ## Cube Drafting
    A drafting cube is a set of (generally 360 or 720) cards that has been selected by
    the cube builder to draft with their friends. With respect to cube drafting this 
    model could be used to:
    * Provide scores for cards that have not been captured by draftaholics
    * Provide a top X pick order for the cube (i.e. cards to prioritise)
    * Allow the power level of a cube to be tested vs other cubes
    * Provide (based on partial dependency analysis) highlighting of themes present in the cube
    
    
    ## New Sets
    When a previously unseen set is released, this model could help provide an initial
    view of relative card sets, which might otherwise be difficult to identify.
    
    '''
    
    
    
    
    
    