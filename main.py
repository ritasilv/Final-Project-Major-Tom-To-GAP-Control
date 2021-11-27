
import streamlit as st
import pandas as pd
#from PIL import Image

#from sklearn.ensemble import RandomForestClassifier
#from func import user_input_features
from func1 import user_input_features, load_data, model, encoding

st.write("""
# Salary Prediction App
Major Tom To GAP Control: closing the gender pay gap
""")

#image= Image.open("bowie.jpg")
#st.image(image, use_column_width=True)


#we create a sidebar on the left of the page
st.sidebar.header('User Input Parameters')


df = user_input_features()

st.subheader('Your data:') 

st.table(df)

st.dataframe(df)


X_train, y_train = load_data()

result = model(df, X_train, y_train)
st.write(result)

