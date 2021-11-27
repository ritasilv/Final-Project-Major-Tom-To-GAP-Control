



import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


#List of features to use in func

role_list = ['Data Analyst', 'Data Scientist', 'Data Engineer','Other']
gender_list = ['Man', 'Woman', 'Other']
age_list = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69']
country_list = ['Russia', 'UK', 'Spain', 'Germany', 'France', 'Italy', 'Poland', 'Netherlands', 'Ukraine', 'Portugal', 'Greece', 'Ireland', 'Sweden', 'Switzerland', 'Belgium', 'Romania', 'Czech Republic', 'Denmark', 'Austria', 'Belarus']
experience_ml_list = ['Under 1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-20 years', '20 or more years']


def user_input_features():
    
    gender = st.sidebar.selectbox('Select your gender', gender_list,0)
    age = st.sidebar.selectbox('Select your age', age_list,0)
    country = st.sidebar.selectbox('Select your country', country_list,0)
    role = st.sidebar.selectbox('Select your country', role_list,0)
    experience = st.sidebar.selectbox('Select the years of experience in Machine Learning', experience_ml_list,0)
    #industry_list = st.sidebar.selectbox('Select the industry of your comapny', industry_list, 0)


    data = {
            'gender': gender,
            'age': age,
            'country': country,
            'experience': experience,
            'role': role}
    features = pd.DataFrame(data, index=[0])
    return features





def load_data():
    """
    Function to load the data
    
    """

    df1 = pd.read_csv('df_model_strm.csv', index_col=0)

    X = df1.drop('salary_avg', axis=1)
    y = df1['salary_avg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train,y_train



def encoding(df):

    enc = OrdinalEncoder()
    df = enc.fit_transform(df)

    return df


'''
# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)
'''

'''
def encoding (df):
    
    ordinal_cols_mapping = [{
        "col": "age",
        "mapping": {
            '18-21': 0,
            '22-24': 1,
            '25-29': 2,
            '30-34': 3,
            '35-39': 4,
            '40-44': 5,
            '45-49': 6,
            '50-54': 7,
            '55-59': 8,
            '60-69': 9    
        }}, {
        "col": "gender",
        "mapping": {
            'Man': 0,
            'Woman': 1,
            'Other': 2
        }}, {        
        "col": "education",
        "mapping": {
            'High school': 0,
            'Some college': 1,
            'Bachelor’s degree': 2,
            'Master’s degree': 3,
            'Doctoral degree': 4,
            'Professional doctorate': 5
        }}, {
        "col": "experience",
        "mapping": {
            'Under 1 year':0, 
            '1-2 years':1,
            '2-3 years':2,
            '3-4 years':3,
            '4-5 years':4,
            '5-10 years':5,
            '10-20 years':6,
            '20 or more years':7
        }}, {
        "col": "country",
        "mapping": {
            'Russia': 0,
            'UK': 1,
            'Spain': 2,
            'Germany': 3,
            'France': 4,
            'Italy': 5,
            'Poland': 6,
            'Netherlands': 7,
            'Ukraine': 8,
            'Portugal': 9,
            'Greece': 10,
            'Ireland': 11,
            'Sweden': 12,
            'Switzerland': 13,
            'Belgium': 14,
            'Romania': 15,
            'Czech Republic': 16,
            'Denmark': 17,
            'Austria': 18,
            'Belarus': 19
        }}, {
        "col": "role",
        "mapping": {
            'Data Scientist':0, 
            'Data Engineer':1,
            'Data Analyst':2,
            'Other': 3
        }}   
    ]

    encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mapping, return_df = True)
    
    df2 = encoder.fit_transform(df)


    return df2
'''

def model(df,x,y):
    """
    Function fit a Random Forest Model
    Args: 
        df: dataframe with the users parameters
        x: the predictors
        y: the response variable (kind of species)
    Returns:
        pred: which species did you see in the field
        pred_proba: the probability of such a prediction
    """
    #df_encoded = encoding(x)
    df = encoding(df)
    rf = RandomForestRegressor(random_state=0, max_features='auto', min_samples_leaf=2, min_samples_split=10, max_depth=20, n_estimators=400)
    rf.fit(x, y)
    
    prediction = int(rf.predict(df))

    return prediction
  