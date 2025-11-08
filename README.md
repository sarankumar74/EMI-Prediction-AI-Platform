# 💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/Tool-MLflow-lightblue?logo=mlflow)
![Machine Learning](https://img.shields.io/badge/ML-Classification%20%26%20Regression-orange)
![SQL](https://img.shields.io/badge/Database-SQL-green?logo=postgresql)
![Domain](https://img.shields.io/badge/Domain-FinTech%20%26%20Banking-purple)

---

## 📘 Overview
**EMIPredict AI** is an **intelligent financial risk assessment platform** that leverages **machine learning, data analytics, and MLflow tracking** to predict **EMI eligibility and maximum payable EMI amount**.  
The platform combines **classification and regression models**, advanced **feature engineering**, and **real-time web deployment** via **Streamlit Cloud** to empower financial institutions and individuals with smarter, data-driven loan decisions.

---

## 🎯 Problem Statement
Develop a **comprehensive financial risk assessment system** that predicts both **EMI eligibility** and **maximum safe EMI amount** using machine learning and experiment tracking via MLflow.

People often struggle with EMI payments due to poor financial planning and inadequate risk evaluation.  
This platform provides **data-driven insights** for better loan approvals, risk control, and affordability analysis.

The project delivers:
- 🤖 Dual ML problem-solving: **Classification (eligibility)** & **Regression (amount prediction)**
- 📊 Real-time financial analysis using **400,000 financial records**
- ⚙️ Advanced **feature engineering** across 22 demographic and financial variables
- 🧪 Full **MLflow experiment tracking** and model versioning
- ☁️ **Streamlit Cloud deployment** for real-time prediction and analytics
- 🧩 Full **CRUD operations** for data management

---

## 💼 Business Use Cases

### 🏦 Financial Institutions
- Automate loan approval and reduce manual underwriting by **80%**
- Implement **risk-based pricing** for different EMI scenarios  
- Enable **instant eligibility assessment** for in-branch or online customers  

### 💻 FinTech Companies
- Provide **real-time EMI eligibility checks** for digital lending apps  
- Integrate **pre-qualification models** in fintech platforms  
- Deliver **automated risk scoring** for instant loan applications  

### 🏛️ Banks & Credit Agencies
- Recommend **loan amounts based on capacity** and risk tolerance  
- Manage **portfolio risk** and predict defaults  
- Maintain **regulatory compliance** via traceable MLflow decision logs  

### 👨‍💼 Loan Officers & Underwriters
- Use AI-based recommendations for quick loan decisions  
- Access detailed financial profiles within seconds  
- Monitor model accuracy and decision transparency  

---

## 🧠 Skills Takeaway
- **Python** – Data preprocessing and machine learning model building  
- **MLflow** – Model tracking, experiment comparison, and version control  
- **Streamlit Cloud** – Interactive web deployment and real-time prediction  
- **Classification Models** – Logistic Regression, Random Forest, XGBoost  
- **Regression Models** – Linear, Random Forest, XGBoost  
- **Feature Engineering** – Derived ratios, encodings, scaling, and transformations  
- **Data Analysis** – Exploratory visualization, correlation studies, and validation  
- **FinTech Domain Knowledge** – Financial planning and credit risk assessment  

---

## 🗺️ Key Development Steps

### 🧾 Step 1: Data Loading & Preprocessing
- Loaded **400K financial records** across 5 EMI scenarios  
- Cleaned missing values, inconsistencies, and duplicates  
- Applied **validation checks and stratified splits** for model development  

### 📊 Step 2: Exploratory Data Analysis
- Analyzed **eligibility patterns and correlations**  
- Studied **demographic trends and spending behaviors**  
- Generated **statistical reports and business insights**  

### 🧮 Step 3: Feature Engineering
- Built financial ratios: **Debt-to-Income**, **Affordability Index**, etc.  
- Encoded categorical variables and scaled numerical data  
- Developed **risk scoring metrics** based on credit and employment stability  

### 🤖 Step 4: Machine Learning Model Development
#### Classification Models
- Logistic Regression (baseline interpretability)  
- Random Forest Classifier (feature importance)  
- XGBoost Classifier (boosted performance)  
> Evaluated using Accuracy, F1-score, and ROC-AUC  

#### Regression Models
- Linear Regression (baseline model)  
- Random Forest Regressor (ensemble method)  
- XGBoost Regressor (gradient boosting)  
> Evaluated using RMSE, MAE, R², and MAPE  

### 🧪 Step 5: MLflow Integration
- Tracked all model experiments with **MLflow Tracking Server**  
- Logged metrics, hyperparameters, and artifacts  
- Implemented **Model Registry** for best model deployment  

### 🖥️ Step 6: Streamlit Application Development
- Built **multi-page Streamlit web app**  
- Integrated real-time classification and regression predictions  
- Added **data visualization and MLflow dashboards**  
- Included **admin CRUD interface** for data management  

### ☁️ Step 7: Cloud Deployment
- Deployed application on **Streamlit Cloud**  
- Configured **CI/CD from GitHub** for continuous updates  
- Ensured cross-platform accessibility and responsive UI  

---

## 🧩 Project Structure
```bash
EMIPredict_AI/
│
├── data/
│   ├── EMI_dataset.csv           # Main dataset (400K records)
│   ├── processed/                # Cleaned and feature-engineered data
│   └── reports/                  # EDA and business insights
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│
├── models/
│   ├── classification/           # Trained classifiers
│   └── regression/               # Trained regressors
│
├── app/
│   ├── streamlit_app.py          # Streamlit web application
│
├── requirements.txt              # Python dependencies
├── MLproject                     # MLflow configuration file
└── README.md                     # Project documentation
