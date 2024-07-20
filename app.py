import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from sklearn.tree import plot_tree


# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/linnaaaa/klasifikasi-diabetes/main/Dataset/diabetes.csv"
    )
    return df


df = load_data()

# Display information about the data
st.title("Klasifikasi Diabetes menggunakan Decision Tree")
st.write("## Informasi tentang Dataset")
st.write(
    """
Dataset ini berisi informasi tentang pasien dan apakah mereka memiliki diabetes atau tidak. Berikut adalah penjelasan dari setiap kolom dalam dataset:
- **Pregnancies**: Jumlah kehamilan
- **Glucose**: Konsentrasi glukosa plasma
- **BloodPressure**: Tekanan darah diastolik (mm Hg)
- **SkinThickness**: Ketebalan lipatan kulit triceps (mm)
- **Insulin**: Kadar insulin serum dua jam (mu U/ml)
- **BMI**: Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2)
- **DiabetesPedigreeFunction**: Fungsi silsilah diabetes (skor berdasarkan riwayat diabetes keluarga)
- **Age**: Usia (tahun)
- **Outcome**: Hasil (1: Diabetes, 0: Non-Diabetes)
"""
)
st.write(df.head())

# Preprocessing
X = df.drop(["Outcome"], axis=1)
y = df.Outcome

# Split data 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Modelling
model = DecisionTreeClassifier()
model.fit(X_resampled, y_resampled)

# Fine Tuning
param_grid = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"],
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(criterion="entropy"),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
)
grid_search.fit(X_resampled, y_resampled)
best_model = grid_search.best_estimator_


# Prediction function
def make_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = best_model.predict(input_array)
    return prediction[0]


# Example inputs
example_inputs = [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    [8, 183, 64, 0, 0, 23.3, 0.672, 32],
    [1, 89, 66, 23, 94, 28.1, 0.167, 21],
    [0, 137, 40, 35, 168, 43.1, 2.288, 33],
]

st.write("## Contoh Input untuk Prediksi dan Hasilnya")
for i, example in enumerate(example_inputs, 1):
    prediction = make_prediction(example)
    result = "Diabetes" if prediction == 1 else "Non-Diabetes"
    st.write(f"Contoh {i}: {example} - Hasil: {result}")

st.write("## Buat Prediksi")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=0)
blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=150, value=0)
skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
diabetes_pedigree_function = st.number_input(
    "DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.0
)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

input_data = [
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
]
if st.button("Prediksi"):
    prediction = make_prediction(input_data)
    result = "Diabetes" if prediction == 1 else "Non-Diabetes"
    st.write(f"Hasil prediksi untuk data input adalah: {result}")
