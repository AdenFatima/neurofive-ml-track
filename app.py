import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline
model = joblib.load('titanic_pipeline.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter the passenger's details below to predict if they would have survived the disaster.")

# Input fields arranged in columns for a clean UI
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30)
    fare = st.slider("Fare (in $)", 0.0, 500.0, 32.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], help="S = Southampton, C = Cherbourg, Q = Queenstown")

# Feature Engineering logic (replicating Task 7)
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

if st.button("Predict Survival"):
    # Create a dataframe from user inputs
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked],
        'FamilySize': [family_size],
        'IsAlone': [is_alone]
    })
    
    # Make prediction using the pipeline
    prediction = model.predict(input_data)
    
    st.divider()
    if prediction[0] == 1:
        st.success("✅ Prediction: This passenger would likely SURVIVE.")
    else:
        st.error("❌ Prediction: This passenger would likely NOT SURVIVE.")
