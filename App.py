import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression

st.write("""
# GRADEWISE
Simple prediction app for students' performance index
""")
st.sidebar.header('Select Parameter')

def input_features():
    # Slider inputs
    Hours_Studied = st.sidebar.slider('Hours Studied', 1.0, 4.77, 9.0)
    Previous_Scores = st.sidebar.slider('Previous Scores', 40.0, 68.87, 99.0)
    Sleep_Hours = st.sidebar.slider('Sleep Hours', 4.0, 7.19, 9.0)
    Sample_Question_Papers_Practiced = st.sidebar.slider('Sample Question Papers Practiced', 0.0, 4.21, 9.0)

    # Checkbox input
    check = st.sidebar.checkbox('Extracurricular Activities (Yes/No)')
    value = 1 if check else 0  # Assign 1 for Yes, 0 for No

    # Display the checkbox value in the sidebar
    st.sidebar.write(f'You chose: {"Yes" if value == 1 else "No"}')

    # Create data dictionary
    data = {
        'Hours Studied': Hours_Studied,
        'Previous Scores': Previous_Scores,
        'Sleep Hours': Sleep_Hours,
        'Sample Question Papers Practiced': Sample_Question_Papers_Practiced,
        'Extracurricular Activities': value  # encoded as 0 or 1
    }

    # Create a DataFrame with the data
    features = pd.DataFrame(data, index=[0])
    
    return features

# Calling the function to get the input features
df = input_features()

# Load the dataset and prepare the model
dataset = pd.read_csv('Student_Performance.csv')

# Ensure the 'Extracurricular Activities' is encoded properly in the dataset
dataset['Extracurricular Activities'] = dataset['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Target (Performance Index)

# Train the model
clf = LinearRegression()
clf.fit(X, y)

# Now ensure that the columns in df are in the same order as in the training data (X)
# Print columns of both training data and the input data (df)
st.write("Training data columns:", X.columns)
st.write("Input data columns:", df.columns)

# Reorder the input data columns to match the training data
df = df[X.columns]

# Get prediction
prediction = clf.predict(df)

# Prepare the results DataFrame
result_df = pd.DataFrame({
    'Hours Studied': df['Hours Studied'],
    'Previous Scores': df['Previous Scores'],
    'Extracurricular Activities': df['Extracurricular Activities'],
    'Sleep Hours': df['Sleep Hours'],
    'Sample Question Papers Practiced': df['Sample Question Papers Practiced'],
    'Predicted Performance Index': prediction[0]
})

# Display the results DataFrame
st.subheader('Prediction Results')
st.write(result_df)
