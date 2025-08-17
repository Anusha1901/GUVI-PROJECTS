import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("insurance_cost_model.pkl")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", 
                            ["🏠 Project Introduction", 
                             "📊 Visualizations", 
                             "🔮 Prediction", 
                             "🙋 About Me"])

# -------------------------------
# Project Introduction
# -------------------------------
if app_mode == "🏠 Project Introduction":
    st.title("Medical Insurance Cost Prediction 💡")
    st.write("""
    This project predicts **medical insurance costs** based on factors such as 
    age, gender, BMI, smoking status, children, and region.  
    It also provides insights into how lifestyle factors impact insurance charges.  
    """)

# -------------------------------
# Visualizations
# -------------------------------
elif app_mode == "📊 Visualizations":
    st.title("Exploratory Data Analysis 📊")
    st.write("Select a question to explore:")

    # Define your visualization questions and paths
    viz_options = {
        "1. What is the distribution of medical insurance charges?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/1.png",
        "2. What is the age distribution of the individuals?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/2.png",
        "3. How many people are smokers vs non-smokers?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/3.png",
        "5. Which regions have the most number of policyholders?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/5.png",
        "6. How do charges vary with age?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/6.png",
        "7. Is there a difference in average charges between smokers and non-smokers?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/7.png",
        "8. Does BMI impact insurance charges?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/8.png",
        "9. Do men or women pay more on average?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/9.png",
        "10. Is there a correlation between the number of children and the insurance charges?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/10.png",
        "11. How does smoking status combined with age affect medical charges?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/11.png",
        "12. What is the impact of gender and region on charges for smokers?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/12.png",
        "13. How do age, BMI, and smoking status together affect insurance cost?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/13.png",
        "14. Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/14.png",
        "15. Are there outliers in the charges column? Who are the individuals paying the highest costs?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/15.png",
        "16. Are there extreme BMI values that could skew predictions?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/16.png",
        "17. What is the correlation between numeric features like age, BMI, number of children, and charges?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/17.png",
        "18. Which features have the strongest correlation with the target variable (charges)?": "D:/ANUSHA/PROJECTS_WORK/GUVI_PROJECTS/PROJECT_3/IMAGES/18.png"
    }

    choice = st.selectbox("Choose a visualization", list(viz_options.keys()))

    img_path = viz_options[choice]
    img = Image.open(img_path)
    st.image(img, caption=choice, use_container_width=True)

# -------------------------------
# Prediction
# -------------------------------
elif app_mode == "🔮 Prediction":
    st.title("Predict Your Insurance Cost 💰")

    st.write("Enter your details below to get an estimate:")

    # Input form
    age = st.slider("Age", 18, 100, 30)
    gender = st.radio("Gender", ["male", "female"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Number of Children", 0, 10, 0)
    smoker = st.radio("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    # Calculate obesity from BMI
    obese = True if bmi >= 30 else False

    # Preprocess input
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [gender],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region],
        "obese": [obese]
    })

    if st.button("Predict Insurance Cost"):
        prediction = model.predict(input_df)[0]

        # Optional error margin (±10% as placeholder)
        error_margin = 0.1 * prediction
        lower = prediction - error_margin
        upper = prediction + error_margin

        st.success(f"Estimated Insurance Cost: **${prediction:,.2f}**")
        st.info(f"Confidence Range: ${lower:,.2f} - ${upper:,.2f}")

# -------------------------------
# About Me
# -------------------------------
elif app_mode == "🙋 About Me":
    st.title("About the Author 👩‍💻")
    st.write("""
    **Name:** Anusha Dixit  
    **Background:** Data Scientist with expertise in **Machine Learning, NLP, Deep Learning, and Analytics**.  
    Passionate about building **AI solutions** that solve real-world problems.  

    """)
