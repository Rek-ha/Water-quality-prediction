import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def water_quality(input_data):
    # Loading the saved model
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    
    # Convert input data to numeric values explicitly
    numeric_input_data = [float(value) if value.strip() != '' else 0.0 for value in input_data]
    
    # Change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(numeric_input_data)
    
    # Reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    scaler=StandardScaler()
    scaler.fit(input_data_reshaped)
    
    # Standardizing the input data
    input_data_std = scaler.transform(input_data_reshaped)
    
    # Giving the probability of the model
    prediction = loaded_model.predict(input_data_std)
    
    if prediction[0] == 1:
        return 'The water is portable'
    else:
        return 'The water is non portable'

def main():
    # Giving the title
    st.title('Water Quality Prediction')
    
    # Creating the input data options--getting the input data from the user
    ph = st.text_input('ph')
    Hardness = st.text_input('Hardness')
    Solids = st.text_input('Solids')
    Chloramines = st.text_input('Chloramines')
    Sulfate = st.text_input('Sulfate')
    Conductivity = st.text_input('Conductivity')
    Organic_carbon = st.text_input('Organic_carbon')
    Trihalomethanes = st.text_input('Trihalomethanes')
    Turbidity = st.text_input('Turbidity')

    # Code for prediction
    Predict = ''

    # Creating the button
    if st.button('Test result'):
        input_data = [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
        if all([value.strip() != '' for value in input_data]):
            Predict = water_quality(input_data)
        else:
            Predict = 'Please fill in all input fields'
    
    st.success(Predict)

if __name__ == '__main__':
    main()