import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# **Penguin App Prediction**

This app predicts the **Palmer penguin** species!""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file]['https://github.com/anishnair3091/penguins#:~:text=penguins_example.csv']""")

#Collect users input into Dataframe

uploaded_file= st.sidebar.file_uploader("Upload your input CSV file", type= ["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        
        island= st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm= st.sidebar.slider('Bill Length(mm)', 32.1, 59.6, 43.9)
        bill_depth_mm= st.sidebar.slider('Bill Depth(mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
        body_mass_g= st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'sex': sex,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm':flipper_length_mm,
                'body_mass_g': body_mass_g}       
        features= pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

penguins_raw = pd.read_csv(r'https://github.com/anishnair3091/penguins#:~:text=penguins_cleaned.csv', on_bad_lines='skip')
penguins= penguins_raw.drop('species', axis = 1)
df = pd.concat([input_df, penguins], axis =0)
    
encode= ['sex', 'island']

for col in encode:
    dummy= pd.get_dummies(df[col], prefix= col)
    df = pd.concat([df, dummy], axis= 1)
    del df[col]

df = df[:1]

st.subheader('User input features')

if uploaded_file is not None:
    st.write(df)

else:
    st.write('Awaiting the input file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

load_model = pickle.load(open('penguins_model.pkl', 'rb'))

prediction= load_model.predict(df)
prediction_proba= load_model.predict_proba(df)

st.subheader('Prediction')
penguin_species= np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguin_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
    
