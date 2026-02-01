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

# 🔮 Futuristic Vision: Intelligent Patient Flux Management

**JijaArogyaCare** is designed to evolve beyond individual appointment prediction into a **hospital-wide patient flux intelligence system**, enabling healthcare providers to anticipate and manage patient movement at scale.

---

## 🚑 Patient Flux Rate Prediction

Using aggregated appointment data, historical no-show trends, seasonal patterns, and external signals, the system can forecast:

- **Hourly/Daily Patient Inflow Rates**
- **Department-wise Congestion Probability**
- **Peak vs. Low-Demand Windows**
- **Emergency Spillover Risk**

### This enables administrators to dynamically adjust:
- Doctor rosters  
- OPD slot availability  
- Resource allocation (beds, diagnostics, staff)

📊 **Outcome:** Reduced overcrowding, smoother patient flow, and optimized hospital throughput.

---

## 📱 Public Patient App Alerts (Preventive Engagement)

A future public-facing mobile/web app can notify patients in advance using AI-driven insights:

### Smart Appointment Alerts
> *“High crowd expected tomorrow between 10–12 PM. Consider rescheduling.”*

### Alternative Slot Suggestions
- Recommends low-flux time slots based on real-time predictions.

### Teleconsultation Nudges
- Automatically suggests virtual visits for high-risk no-show patients.

📱 **Outcome:** Patients make informed decisions, reducing missed visits and unnecessary travel.

---

## 💬 WhatsApp-Based Intelligent Reminders

To maximize accessibility, the system integrates **WhatsApp-first communication**, especially effective in low-tech or rural settings.

### Tiered Reminder System
- **T-48 hrs:** Gentle reminder  
- **T-24 hrs:** Confirmation request  
- **T-6 hrs:** Travel + location reminder  

### Context-Aware Messaging
- Includes weather alerts, transport advisories, or delay warnings.

### Two-Way Interaction
- Patients can confirm, reschedule, or cancel directly via WhatsApp.

💬 **Outcome:** Higher response rates, reduced no-shows, and improved patient trust.

---

## 🧠 Analytical Reasoning & Decision Intelligence

Beyond predictions, JijaArogyaCare provides actionable analytical reasoning for decision-makers:

### Why Flux Is Increasing
- Festival seasons  
- Weather disruptions  
- Public transport strikes  

### What-if Simulations
- Impact of adding extra OPD slots  
- Effect of telehealth substitution  

### Bias & Fairness Monitoring
- Ensures vulnerable populations are not disproportionately flagged as no-shows.

🧠 **Outcome:** Transparent, ethical, and data-backed operational decisions.

---

## 🌱 Long-Term Vision

**JijaArogyaCare** aims to become a **self-learning healthcare operations co-pilot**, continuously adapting to patient behavior, societal patterns, and public health signals—bridging the gap between **AI prediction** and **real-world healthcare efficiency**.

---

## 📚 Research & References
* Predicting Patient No-Shows Using Machine Learning* (2025 Review).

* Real-Time Predictive Analytics and Dashboard Integration* (2025).

* Predicting Missed Appointments in Primary Care* (2025).

* Decision Analysis Framework for ML No-Show Prediction* (2024).

* WHO/ILO Report on Long Working Hours and Health* (2021).
