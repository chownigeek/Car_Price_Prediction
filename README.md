# Car_Price_Prediction
This project is a Streamlit-based web application that predicts the selling price of a car based on various features like brand, model, year, engine size, mileage, fuel type, transmission, and condition. It uses a machine learning pipeline saved as a .pkl file.

## “Built ML model → Tuned → Saved → Deployed via Streamlit”

Built using:

- Python 🐍
- Scikit-learn 🤖
- XGBoost 🌲
- Streamlit 🎈

---

## 📌 Project Overview

This project demonstrates an end-to-end ML workflow:

✔ Data preprocessing  
✔ Feature engineering  
✔ Model training & evaluation  
✔ Hyperparameter tuning  
✔ Model persistence  
✔ Web app deployment  

The final model predicts car prices based on user inputs via a Streamlit interface.

---

## 🧠 Machine Learning Pipeline

### **1️⃣ Data Preprocessing**
- Handling missing values
- Encoding categorical variables using `pd.get_dummies()`
- Cleaning column names (spaces → underscores)
- Feature scaling (if applicable)

---

### **2️⃣ Models Evaluated**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Lasso Regression
- XGBoost Regressor
- LightGBM Regressor

---

### **3️⃣ Model Selection**
Models were compared using:

- R² Score
- RMSE
- MAE

The best-performing model was saved for deployment.

---

## 💾 Saved Artifacts

| File | Description |
|------|-------------|
| `best_car_price_model.pkl` | Trained ML model |
| `model_columns.pkl` | Feature column order used during training |

---

## 🎈 Streamlit Web App

The Streamlit app allows users to:

✔ Select car details  
✔ Enter numerical features  
✔ Generate price prediction instantly  

---
