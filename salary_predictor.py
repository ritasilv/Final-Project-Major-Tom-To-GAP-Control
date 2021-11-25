


import streamlit as st
import pandas as pd
#from PIL import Image

#from sklearn.ensemble import RandomForestClassifier
#from src.support import user_input_features, load_data, model

role_list = ['Data Analyst', 'Data Scientist', 'Data Engineer', 'Other']
gender_list = ['Man', 'Woman', 'Other']
age_list = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69']
country_list = ['Russia', 'UK', 'Spain', 'Germany', 'France', 'Italy', 'Poland', 'Netherlands', 'Ukraine', 'Portugal', 'Greece', 'Ireland', 'Sweden', 'Sweden', 'Switzerland', 'Belgium', 'Romania', 'Czech Republic', 'Denmark', 'Austria', 'Belarus']
education_list = ['High School', 'Some college', 'Bachelor’s degree', 'Master’s degree', 'Doctoral degree', 'Professional doctorate']
experience_ml_list = ['Under 1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-20 years', '20 or more years']
#industry_list = ['Computers/Technology', 'Academics/Education', 'Accounting/Finance', 'Other', 'Medical/Pharmaceutical', 'Manufacturing/Fabrication','Government/Public Service', 'Online Service/Internet-based Services', 'Energy/Mining', 'Retail/Sales', 'Insurance/Risk Assessment', 'Broadcasting/Communications', 'Shipping/Transportation', 'Marketing/CRM', 'Online Business/Internet-based Sales', 'Military/Security/Defense', 'Non-profit/Service', 'Hospitality/Entertainment/Sports']


st.write("""
# Salary Prediction App
Major Tom To GAP Control: closing the gender pay gap
""")

#image= Image.open("bowie.jpg")
#st.image(image, use_column_width=True)


#we create a sidebar on the left of the page
st.sidebar.header('User Input Parameters')


def user_input_features():

    role = st.sidebar.selectbox('Select your role', role_list, 0)
    gender = st.sidebar.selectbox('Select your gender', gender_list, 0)
    age = st.sidebar.selectbox('Select your age', age_list, 0)
    country = st.sidebar.selectbox('Select your country', country_list, 0)
    education= st.sidebar.selectbox('Select your highest level of education', education_list, 0)
    experience = st.sidebar.selectbox('Select the years of experience in Machine Learning', experience_ml_list, 0)
    #industry_list = st.sidebar.selectbox('Select the industry of your comapny', industry_list, 0)


    data = {'role': role,
            'gender': gender,
            'age': age,
            'country': country,
            'education': education,
            'experience': experience}
            #'industry', industry
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Your data:')

st.write(df)



    






#df = user_input_features()

#st.subheader('User Input parameters')
#st.table(df)

#st.dataframe(load_data())

#salary = load_data()

#x = salary.drop(["target"], axis = 1)
#y = salary["target"]

#pred, prob_pred = model (df, x, y)


#st.subheader('Prediction')
#st.write(pred)
   

#st.subheader('Prediction Probability')
#st.table(prob_pred)