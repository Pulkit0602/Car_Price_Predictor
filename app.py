import streamlit as st
import numpy as np
import pandas as pd 
import pickle

#READIND FILE AND MODEL 
pipe_final = pickle.load(open("LinearRegressionModel.pkl","rb"))
X = pickle.load(open("preprocessX.pkl","rb"))

#Setting up Data For Drop Down 

company_list = X["Company"].unique().tolist()
fuel_type_list = X["Fuel_type"].unique().tolist()
state_list = X["State"].unique().tolist()
year_list = sorted(X["Year"].unique().tolist())   

#SETTING UP INTERFACE
st.title("Car Price Predictor")
new_title = '<p style="  font-weight: bold; color:Red; font-size: 28px;">Looking To Sell</p>'
st.markdown(new_title, unsafe_allow_html=True)
new_title = '<p style=" font-weight: bold; color:#191970; font-size: 20px;">Know Your Car BEST VALUE</p>'
st.markdown(new_title, unsafe_allow_html=True)
st.text("")
st.text("")

#SETTING UP INPUTS IN INERFACE
col1,col2,col3= st.columns(3)
with col1:    
    company = st.selectbox("Company Team" , company_list)

with col2:
    model_list = X[X["Company"] == company].Name.tolist()
    model = st.selectbox("Model" , model_list)

with col3:
    fueltype = st.selectbox("Fuel Type" , fuel_type_list)

#Wicket, CurrentRun,Overs Completed
col1,col2 = st.columns(2)
with col1:    
     year = st.selectbox("Year" , year_list)

with col2:
    kms = st.number_input('Kms Driven')


state = st.selectbox("State" , state_list)

#Getting Results From Model
if st.button("Predict"):
    amount = pipe_final.predict(pd.DataFrame(columns=['Name','Company','Year','Kms_driven','Fuel_type',"State"],data=np.array([model,company,year,kms,fueltype,state]).reshape(1,6)))
    if amount<0:
       st.header("Sorry Can't Sell")

    else:
        st.header("Predicted Amount: " + str(np.ceil(amount[0])))