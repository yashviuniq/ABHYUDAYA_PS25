# JijaArogyaCare: AI Patient No-Show Predictor

**Project Name:** JijaArogyaCare 

**Team Name:** Chatrapati Shivaji Maharaj 

**Problem Statement ID:** PS25 

---

## 📋 Project Overview

JijaArogyaCare is a web-based AI-powered healthcare operations system designed to predict whether a patient is likely to miss (No-Show) or attend (Show) a scheduled hospital or clinic appointment. By leveraging historical scheduling data, the system aims to reduce clinic inefficiency, minimize wasted staff time, and prevent delays in patient care.

**Note:** This system is strictly for operational support and does not perform medical diagnosis.

---

## ✨ Key Features

**Predictive Risk Scoring:** Generates a Show/No-Show risk score and outcome for upcoming appointments.

**Explainable AI:** Provides clear contributing factors (e.g., lead time, travel distance) so staff can understand the "why" behind a prediction.

**Real-time Insights:** Enables proactive, targeted outreach and operational interventions.

**Operational Optimization:** Supports data-driven overbooking strategies and workforce planning.

**Resource Management:** Reduces idle time for high-cost clinical resources.

---

## 🛠️ Technical Specifications

### **Architecture Workflow**

The system follows a structured pipeline from data ingestion to automated intervention:

**Data Sources:** Hospital appointments and demographics.

**Preprocessing:** Data cleaning and feature engineering (e.g., calculating lead time).

**ML Engine:** Random Forest Classifier utilizing SHAP for explainability, hosted on SageMaker.

**Centralized API:** API Gateway and Lambda functions to route risk scores.
 
**Output:** Results displayed on a Streamlit Staff Dashboard and notifications sent via SMS/Email (SNS/SES).

### **Technologies Used**

**Frontend:** Streamlit (Python) for a single-page dashboard with visual risk gauges.

**Backend:** Flask/FastAPI (Python).

**AI Model:** Random Forest Classifier with SHAP values for interpretability.

**Database:** SQLite/CSV for session-based data storage.

**Cloud:** Streamlit Community Cloud/Render for deployment.

---

## 🚀 Competitive Advantage

Compared to legacy solutions (like Epic), JijaArogyaCare offers several innovations:

| Feature | Legacy Solutions | JijaArogyaCare | Benefit |
| --- | --- | --- | --- |
| **Model Type** | Basic Logistic Regression | Advanced ML (Random Forest) | Captures complex non-linear patterns.

| **Data Scope** | Clinical/Demographic only | External Data (Weather, SES) | Includes top predictors like transport status.

| **Explainability** | "Black Box" | Explainable AI (SHAP/LIME) | Builds clinician trust through transparency.

| **Intervention** | Generic SMS | Tiered Actions | Suggests Uber vouchers or Telehealth.

| **Fairness** | Often Biased | Equitable AI Framework | Actively corrects for bias in race/insurance.

---

## 📈 Impact & Benefits

**For Patients:** Improved timely access to healthcare.

**For Staff:** Balanced working hours, reduced workload stress, and prevention of burnout.

**For Administration:** Better staff planning, lower revenue loss, and reduced overtime costs.

**Psychological:** Improved mental well-being by creating predictable workloads.

---

## 📚 Research & References
* Predicting Patient No-Shows Using Machine Learning* (2025 Review).

* Real-Time Predictive Analytics and Dashboard Integration* (2025).

* Predicting Missed Appointments in Primary Care* (2025).

* Decision Analysis Framework for ML No-Show Prediction* (2024).

* WHO/ILO Report on Long Working Hours and Health* (2021).