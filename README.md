# 💰 EMIPredict AI — Intelligent Financial Risk Assessment Platform
🔍 Machine Learning • MLflow • Streamlit • Classification • Regression • FinTech

## 🚀 Tech Stack & Domain
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/Tool-MLflow-lightblue?logo=mlflow)
![Machine Learning](https://img.shields.io/badge/ML-Classification%20%26%20Regression-orange)
![SQL](https://img.shields.io/badge/Database-SQL-green?logo=postgresql)
![Domain](https://img.shields.io/badge/Domain-FinTech%20%26%20Banking-purple)

---

## 📘 Overview
**EMIPredict AI** is a financial risk assessment platform that predicts:
1️⃣ **Whether a customer is eligible for EMI (Loan Classification)**  
2️⃣ **Maximum safe EMI amount the customer can pay (Loan Regression)**  

The platform uses:
- Dual ML architecture (classification + regression)
- Real-time prediction via Streamlit Cloud
- MLflow for experiment tracking, version control & model registry
- 400,000+ financial records for highly accurate risk scoring

---

## 🎯 Problem Statement
Loan defaults often result from poor EMI budgeting and inaccurate risk approval.  
This system provides **data-driven EMI evaluation** to improve:

- Customer affordability prediction  
- Loan approval accuracy  
- Financial risk control across institutions  

---

## 💼 Business Use Cases

| Stakeholder | Value |
|------------|-------|
| 🏦 Banks | Reduced underwriting workload, improved approval decisions |
| 💻 FinTech Apps | Instant EMI pre-qualification and eligibility scoring |
| 🏛️ Credit Agencies | Data-driven risk prediction & portfolio protection |
| 👨‍💼 Loan Officers | Faster loan suggestions and transparent decision metrics |

---

## 🧠 Skills Demonstrated
- End-to-end ML lifecycle
- MLflow experiment tracking & deployment
- Streamlit Cloud CI/CD deployment
- Classification + Regression hybrid architecture
- Feature engineering for financial analytics
- CRUD operations for financial datasets

---

## 🧮 Model Evaluation — Classification

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|---------|
| 🌲 RandomForestClassifier | **0.9538** | 0.92 / 0.87 / 0.96 | 0.96 / 0.06 / 1.00 | 0.94 / 0.12 / 0.98 |
| 🌳 DecisionTreeClassifier | **0.9754** | 0.96 / 0.71 / 0.99 | 0.96 / 0.72 / 0.99 | 0.96 / 0.71 / 0.99 |
| ⚡ XGBClassifier | **🚀 0.9897 (Best)** | 0.98 / 0.90 / 1.00 | 0.99 / 0.85 / 1.00 | 0.98 / 0.87 / 1.00 |

### 🔥 Best Classification Model
✔ **XGBClassifier — 98.97% accuracy**  
✔ Best macro & weighted averages  
✔ Highest stability across all 3 target labels

---

## 📈 Model Evaluation — Regression

| Model | R² Score | MSE | MAE |
|-------|---------|-----|-----|
| ➖ Linear Regression | **1.0** | 6.36e-21 | 6.29e-11 |
| 🌲 RandomForestRegressor | **0.9999999986 (Best)** | 0.0430 | 0.0354 |
| 🌳 DecisionTreeRegressor | **0.9999999966** | 0.1080 | 0.0331 |

### 🔥 Best Regression Model
✔ **RandomForestRegressor — Most stable with near-perfect accuracy**  
✔ Lowest error on financial EMI prediction

---

## 🗺️ Project Workflow

### 1️⃣ Data Loading & Cleaning
- 400K+ financial records across 5 EMI scenarios
- Validated inconsistencies & missing values
- Stratified sampling for balanced ML training

### 2️⃣ Exploratory Data Analysis
- Demographic & behavioral spending insights
- Eligibility distribution analysis
- Correlation studies & anomaly detection

### 3️⃣ Feature Engineering
- Financial ratios (Debt-to-Income, Affordability Index)
- Categorical encodings + numerical scaling
- Risk scoring layer based on employment & credit patterns

### 4️⃣ Machine Learning
- Classification → Logistic, RF, XGBoost
- Regression → Linear, RF, XGBoost
- Hyperparameter tuning with cross-validation

### 5️⃣ MLflow Integration
- Metrics, params & artifacts logged for every run
- Model registry for versioned deployment
- Experiment comparisons for production model decision

### 6️⃣ Streamlit Web App
- Multi-page financial decision UI
- Real-time probability scoring + EMI calculator
- Visualization dashboards and MLflow insights

### 7️⃣ Cloud Deployment
- Streamlit Cloud hosting
- GitHub CI/CD auto-redeployment on update
- Accessible & responsive for desktop & mobile

---


<summary>📸 Click to view Streamlit UI screenshots</summary>

#### Home Page  
![Home Page](https://github.com/user-attachments/assets/9bc4f0df-8a89-4dde-9a5e-25894fd66880)


#### Classification Results Page  
![Result Page](https://github.com/user-attachments/assets/dbbc8d12-3a80-4192-99e7-eed395d1c9de)


####  Regression Results Page  
![Dashboard](https://github.com/user-attachments/assets/95637404-3667-4181-9006-94d7fb224db5)



---

## 📁 Project Structure
```bash

EMIPredict_AI/  
│  
├── EMI Clean & EDA /  
│   ├── emi_clean.ipynb  
│   └── emi_eda.ipynb  
│  
├── EMI Datasets/  
│   ├── EMI Prediction Clean.csv  
│   └── emi_prediction.csv  
│  
├── EMI Streamlit Codes/  
│   ├── emi_classification.py  
│   └── emi_regression.py  
│  
├── EMI Training Codes/  
│   ├── emi_classification.ipynb  
│   └── emi_regression.ipynb  
│  
├── app.py  
├── requirements.txt  
└── README.md  

---

## 🛠️ Run Locally

Install dependencies
```
pip install -r requirements.txt
```

Run Streamlit app
```
streamlit run app.py
```

---
