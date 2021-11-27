import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.metrics import mean_squared_error

#1. Function to Q7 "hat programming languages do you use on a regular basis?", as I would like to have them counted in one single row

def finalCount(row):
    count = 0
    if row['Q7_Part_1'] == 'Python':
        count = count + 1 
    if row['Q7_Part_2'] == 'R':
        count = count + 1 
    if row['Q7_Part_3'] == 'SQL':
        count = count + 1
    if row['Q7_Part_4'] == 'C':
        count = count + 1
    if row['Q7_Part_5'] == 'C++':
        count = count + 1 
    if row['Q7_Part_6'] == 'Java':
        count = count + 1
    if row['Q7_Part_7'] == 'Javascript':
        count = count + 1
    if row['Q7_Part_8'] == 'Julia':
        count = count + 1
    if row['Q7_Part_9'] == 'Swift':
        count = count + 1
    if row['Q7_Part_10'] == 'Bash':
        count = count + 1
    if row['Q7_Part_11'] == 'MATLAB':
        count = count + 1
    if row['Q7_Part_12'] == 'None':
        count = count + 1
    if row['Q7_OTHER'] == 'Other':
        count = count + 1
    return count


 #2. Function to get the unique value of columns

def unique_val(df):
    for col_names in list(df):
        print("\n" + col_names)
        print(df[col_names].unique(), '\n')


#3 Function to value count for each column

def value_count(df):
    for column in df.select_dtypes(np.object):
        print("\n" + column)
        print(df[column].value_counts())
'''


'''
#4 Function to show distribution plots and boxplots (for outliers) of numerical columns

def dist_boxplot_num(data):
    for col in data.select_dtypes(np.number):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.distplot(data[col], ax=axes[0])
        sns.boxplot(data[col], ax=axes[1])
        plt.show()



#5 BoxCox transformation 

def boxcox_transform(data, skip_columns=[]):
    numeric_cols = data.select_dtypes(np.number).columns
    _ci = {column: None for column in numeric_cols}
    for column in numeric_cols:
        if column not in skip_columns:
# since i know any columns should take negative numbers, to avoid -inf in df
            data[column] = np.where(data[column]<=0, np.NAN, data[column]) 
            data[column] = data[column].fillna(data[column].mean())
            transformed_data, ci = stats.boxcox(data[column])
            data[column] = transformed_data
            _ci[column] = [ci] 
        return data, _ci



#6 Function to show distribution of data categorical columns

def dist_cat(df):
    for col in df.select_dtypes(np.object):
        fig, axes = plt.subplots(1, figsize=(7, 4))
        sns.countplot(x=df[col], data=df)
        plt.show()

#7. Function to run the ChiSquare test to all variables

def col_cat_val(data, columns=[]):
    for i in columns:
        for j in columns:
            if i != j:
                data_crosstab = pd.crosstab(data[i], data[j], margins = False)
                print (i, j)
                print (chi2_contingency(data_crosstab, correction=False), '\n')

#8 Function to clean the data by removing special characteres and redundant words

def cleaner(x):
    
    check = ['+','<','>','$',',', ' years', ' employees', ' or more employees']
    
    for i in check:
        if i in str(x):
            x = str(x).replace(i, '')
    return str(x)


#9 Functions to group the data into categories

def country_group(row):
 
    list_1 = ['UK', 'Russia', 'Germany', 'Spain', 'France', 'Italy', 'Poland', 'Ukraine', 'Netherlands', 'Portugal', 'Greece', 'Ireland', 'Sweden', 'Switzerland', 'Belgium', 'Czech Republic', 'Romania', 'Austria', 'Belarus', 'Denmark'] 
    list_2 = ['USA']
    list_3 = ['India']
    
    if row in list_1: 
        return "Europe"
    elif row in list_2:
        return "USA"
    elif row in list_3:
        return "India"
    else:
        return "Other"

def size_group(row):
 
    list_1 = ['0-49 employees'] 
    list_2 = ['50-249 employees']
    #list_3 = ['250-999 employees','1000-9,999 employees','10,000 or more employees']
    
    if row in list_1: 
        return "Small"
    elif row in list_2:
        return "Medium"
    else:
        return "Large"


def size_group1(row):
 
    list_1 = ['0'] 
    list_2 = ['1-2', '3-4','5-9']
    list_3 = ['15-19','10-14']
    list_4 = ['20+']
    
    if row in list_1: 
        return "No team"
    elif row in list_2:
        return "Small"
    elif row in list_3:
        return "Medium"
    else:
        return "Large"


def salary_group(row):
 
    list_1 = ['0-999', '0', '1000-1999', '5000-7499', '2000-2999', '7500-9999', '4000-4999', '3000-3999']
    list_2 = ['10000-14999', '20000-24999', '15000-19999', '25000-29999']
    list_3 = ['30000-39999', '40000-49999']
    list_4 = ['50000-59999', '60000-69999', '70000-79999']
    list_5 = ['80000-89999', '90000-99999', '100000-124999']
    
    if row in list_1: 
        return "less than 10k"
    elif row in list_2:
        return "between 10k and 30k"
    elif row in list_3:
        return "between 30k and 50k"
    elif row in list_4:
        return "between 50k and 80k"
    elif row in list_5:
        return "between 80k and 125k"
    else:
        return "more than 125k"


def role_group(row):

    
    list_1 = ['Data Analyst', 'Product Manager' ,'Program/Project Manager', 'Business Analyst' ]
    list_2 = ['Data Engineer', 'DBA/Database Engineer','Machine Learning Engineer' ,'Software Engineer', 'Developer Relations/Advocacy']
    list_3 = ['Data Scientist', 'Research Scientist', 'Statistician']
    
    if row in list_1: 
        return "Data Analyst"
    elif row in list_2:
        return "Data Engineer"
    elif row in list_3:
        return "Data Scientist"
    else:
        return "Other"


#10 Function to check null values and return % of null

def check_null(df):
    nulls = pd.DataFrame(df.isna().sum()*100/len(df), columns=['percentage'])
    nulls.sort_values('percentage', ascending = False)
    return nulls


#11 Function to apply Random Forest Regressor without hyperparameters


def RandomForestReg(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))


#12 Function to check the RSME of the model


def RSME(y_test, y_pred):


    MSE= mean_squared_error(y_test, y_pred, squared=False)
    RMSE = math.sqrt(MSE)
    print(RMSE)
