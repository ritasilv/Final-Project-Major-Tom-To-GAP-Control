import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')



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