import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

model = load_model("nutrition_model.h5")


st.title("Personalized Nutrition Recommendation System 🍎")
st.write("Enter your nutritional requirements to get food recommendations.")

protein = st.slider("Protein (g)", 0.0, 50.0, 25.0)
fat = st.slider("Total Fat (g)", 0.0, 50.0, 15.0)
carbs = st.slider("Carbohydrates (g)", 0.0, 200.0, 100.0)
fiber = st.slider("Fiber (g)", 0.0, 25.0, 5.0)
calcium = st.slider("Calcium (mg)", 0.0, 1000.0, 500.0)
iron = st.slider("Iron (mg)", 0.0, 25.0, 9.0)
vit_c = st.slider("Vitamin C (mg)", 0.0, 250.0, 30.0)
vit_a = st.slider("Vitamin A (IU)", 0.0, 5000.0, 1500.0)
sodium = st.slider("Sodium (mg)", 0.0, 2500.0, 750.0)
potassium = st.slider("Potassium (mg)", 0.0, 3000.0, 1500.0)
cholesterol = st.slider("Cholesterol (mg)", 0.0, 300.0, 100.0)


user_input = np.array([[protein, fat, carbs, fiber, calcium, iron, vit_c, vit_a, sodium, potassium, cholesterol]])
user_input_normalized = scaler.transform(user_input)

if st.button("Get Recommendations"):
    
    predicted_energy = model.predict(user_input_normalized)[0][0]
    st.write(f"Predicted Energy Requirement: {predicted_energy:.2f} Kcal")

    
    data_path = "ABBREV.csv"
    data = pd.read_csv(data_path)
    selected_columns = [
        'Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)',
        'Calcium_(mg)', 'Iron_(mg)', 'Vit_C_(mg)', 'Vit_A_IU',
        'Sodium_(mg)', 'Potassium_(mg)', 'Cholestrl_(mg)', 'Energ_Kcal', 'Shrt_Desc'
    ]
    data_cleaned = data[selected_columns].copy()

    
    numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns

    
    data_cleaned[numeric_columns] = data_cleaned[numeric_columns].apply(pd.to_numeric, errors='coerce')

    
    data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].mean())

   
    if 'Shrt_Desc' in data_cleaned.columns:
        data_cleaned['Shrt_Desc'] = data_cleaned['Shrt_Desc'].fillna("Unknown")

   
    data_cleaned['Energy_Diff'] = abs(data_cleaned['Energ_Kcal'] - predicted_energy)

   
    filtered_data = data_cleaned[
        (data_cleaned['Energ_Kcal'] > (predicted_energy - 100)) & 
        (data_cleaned['Energ_Kcal'] < (predicted_energy + 100))
    ]
    if filtered_data.empty:
        st.write("No foods found within the desired energy range. Showing closest matches.")
        filtered_data = data_cleaned

 
    recommendations = filtered_data.nsmallest(5, 'Energy_Diff')[['Shrt_Desc', 'Energ_Kcal']]
    st.write("### Top Food Recommendations:")
    st.table(recommendations)
