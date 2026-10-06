# 🎓 AI Student Performance Analytics

An AI-powered machine learning application for predicting student academic performance, identifying academic risk, and providing explainable insights.

## 📌 Project Overview

AI Student Performance Analytics is a machine learning based web application developed using Python and Streamlit.

The system analyzes important academic indicators and predicts a student's performance as:

- 🟢 High
- 🟡 Medium
- 🔴 Low

It also provides risk analysis, prediction probability, what-if simulation, explainable AI insights, and multi-student prediction.

---

## 🎯 Objectives

The main objectives of this project are:

- Predict student academic performance using machine learning.
- Identify students who may require academic support.
- Analyze important academic performance indicators.
- Provide prediction confidence and probability.
- Provide an early warning risk system.
- Allow users to simulate different academic scenarios.
- Provide understandable and explainable insights.
- Analyze multiple students using CSV data.

---

## 🤖 Machine Learning Models

The project compares three machine learning models:

1. Decision Tree
2. Random Forest
3. Logistic Regression

### 🏆 Best Performing Model

**Logistic Regression — 76% Accuracy**

---

## 📊 Input Features

The model uses the following five academic indicators:

| Feature | Description |
|---|---|
| Attendance | Student attendance percentage |
| Internal Marks | Internal assessment marks out of 30 |
| Assignment | Assignment performance percentage |
| Study Hours | Average study hours per day |
| Previous Score | Previous academic score percentage |

---

## ✨ Features

### 🤖 Performance Prediction

Predicts student performance using the trained machine learning model.

### 📊 Prediction Probability

Displays the probability of each performance category.

### 🚨 Risk & Early Warning

Identifies students who may be academically at risk using a rule-based risk scoring system.

### 📋 Performance Breakdown

Provides a visual breakdown of important academic indicators.

### 🔮 What-If Simulator

Allows users to change academic inputs and observe how predictions may change.

### 👥 Multi-Student Prediction

Allows multiple student records to be uploaded through a CSV file for batch prediction.

### 🧠 Explainable AI

Provides simple explanations of which academic indicators are healthy or need improvement.

### 📈 Dataset Analytics

Provides an overview of the student dataset and performance distribution.

### 🧠 Model Analytics

Provides model comparison, accuracy visualization, confusion matrix, and feature importance analysis.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Machine Learning

---

## 📁 Project Structure

```text
AI_Student_Performance_Analytics_GitHub/
│
├── app.py
├── requirements.txt
├── .gitignore
├── student_performance_model.pkl
├── confusion_matrix.png
├── feature_importance.png
│
└── data/
    └── student_performance_demo.csv