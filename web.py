import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from PIL import Image

img = Image.open("crop.png")
st.image(img, use_column_width=True)

df= pd.read_csv('Crop_recommendation.csv')

X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain,Ytrain)

RF_pkl_filename = 'RF.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()

RF_Model_pkl=pickle.load(open('RF.pkl','rb'))

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

def main():  
    st.markdown("<h1 style='text-align: center; color: #008000;'>SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    
    st.write("Please enter the environmental factors below:")

    nitrogen = st.number_input("Nitrogen", value=None)
    phosphorus = st.number_input("Phosphorus", value=None)
    potassium = st.number_input("Potassium", value=None)
    temperature = st.number_input("Temperature (°C)", value=None)
    humidity = st.number_input("Humidity (%)", value=None)
    ph = st.number_input("pH Level", value=None)
    rainfall = st.number_input("Rainfall (mm)", value=None)
   
    inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    if st.button("Predict", key="predict_button"):
        if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop is: {prediction[0]}")
if __name__ == '__main__':
    main()
