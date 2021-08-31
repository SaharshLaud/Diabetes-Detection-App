#Description: Diabetes detection web app

#imports

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Title and subtitle
st.write("""
# Diabetes Detection using ML
Detect if someone has diabetes using Machine Learning!
""")

# Display image
image = Image.open('C:/Users/ADMIN.DESKTOP-SALMH0B/PycharmProjects/Diabetes_Detection_App/diabetes_display.png')
st.image(image, use_column_width=True)

st.write("""
## Application Description
This is a Machine Learning Application to detect if the person has Diabetes or not. The app uses **Random Forest Model** to predict the outcome with the help of the predictor variables. 

The user can use the sliders to adjust the values of the various predictors and get predictions.  The app uses Machine Learning libraries Pandas and Scikit-Learn and it is deployed using Streamlit.
""")

# Get the data
df = pd.read_csv('C:/Users/ADMIN.DESKTOP-SALMH0B/PycharmProjects/Diabetes_Detection_App/diabetes.csv')

# Subheader
st.subheader('Data Information:')
# Show the data as a table
st.dataframe(df)
# Data Statistics
st.write(df.describe())
# Data visualization as a chart
chart = st.bar_chart(df,use_container_width = True)

# Independent variable: X, Dependent variable: Y
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Data Splitting: 75% training and 25% testing
X_train, X_test, Y_Train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)

# Get feature input from user
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 100)
    BMI = st.sidebar.slider('BMI', 0.0, 67.0, 32.0)
    DPF = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {'Pregnancies': pregnancies,
                 'Glucose': glucose,
                 'Blood Pressure': blood_pressure,
                 'Skin Thickness': skin_thickness,
                 'Insulin': insulin,
                 'BMI': BMI,
                 'Diabetes Pedigree Function': DPF,
                 'Age': age
                 }

    # Transfrom data into data frame

    features = pd.DataFrame(user_data, index=[0])
    return features

# Store user input into a variable
user_input = get_user_input()
# Set subheader and display user input
st.subheader('User Input:')
st.write(user_input)

# Create and train model
model = RandomForestClassifier(n_estimators = 500)
model.fit(X_train, Y_Train)

# Show the models metrics
st.subheader('Model Test Accuracy Score: ')
st.write( str(accuracy_score(Y_test, model.predict(X_test)) * 100)+'%' )

# Store models predictions in a variable
prediction = model.predict(user_input)
diab_pred = ''
# Set a subheader to display classification
st.subheader('Final Diabetes Prediction: ')
st.write("According to the Machine Learning App: ")
if prediction == 1:
    st.write("### The person **has Diabetes**")
else:
    st.write("### The person ** does not have Diabetes**")
