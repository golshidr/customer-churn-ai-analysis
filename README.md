# Customer Churn Prediction with Explainable AI

This project analyzes customer churn in a telecom company using machine learning and explainable AI techniques.

The goal is to identify the key factors that lead customers to leave the service and provide business insights that can help companies improve customer retention.

---

## Dataset

The dataset used in this project is the **Telco Customer Churn dataset** from Kaggle.

It contains customer information such as:
- customer tenure
- monthly charges
- internet services
- contract type
- payment methods
- churn status

---

## Methodology

The analysis follows these steps:

1. Data cleaning and preprocessing  
2. Feature engineering and encoding categorical variables  
3. Training a **Random Forest classifier**  
4. Evaluating model performance  
5. Using **SHAP (SHapley Additive Explanations)** to interpret the model

---

## Results

The machine learning model achieves approximately **80% accuracy** in predicting customer churn.

Key factors influencing churn include:

- customer tenure
- monthly charges
- contract type
- payment method
- internet service type

---

## Business Insights

The analysis provides several actionable insights:

- Customers with **short tenure** are more likely to churn.
- Higher **monthly charges** increase churn probability.
- **Long-term contracts** significantly reduce churn risk.
- Customers using **fiber optic internet** show higher churn risk.
- Additional services such as **technical support and online security** reduce churn probability.

---

## Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- SHAP  
- Matplotlib  

---

## Project Structure
