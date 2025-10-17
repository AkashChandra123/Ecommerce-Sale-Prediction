# 🛍️ E-commerce Return Prediction

Predicting product returns using synthetic e-commerce data with PySpark.

## 📌 Overview

This project builds a machine learning pipeline to predict whether a product will be returned after purchase. While the labels are simulated, the pipeline demonstrates real-world techniques for data cleaning, feature engineering, model training, and evaluation using PySpark.

Understanding return behavior is crucial for e-commerce platforms to reduce logistics costs, improve customer satisfaction, and optimize inventory planning.

## 🧰 Tech Stack

- **PySpark**: Distributed data processing and ML pipeline
- **Python**: Core scripting and orchestration
- **Jupyter Notebook**: Interactive development and visualization
- **Matplotlib & Seaborn**: Optional visualizations

## 🔄 Pipeline Steps

1. **Data Cleaning & Joining**
   - Drops nulls and duplicates
   - Joins customer, order, payment, product, and item data

2. **Feature Engineering**
   - Product volume (length × width × height)
   - Indexed categorical features: category, payment type, customer state
   - Assembled feature vector for modeling

3. **Model Training & Evaluation**
   - Logistic Regression and Random Forest classifiers
   - ROC AUC and accuracy metrics
   - Label simulation based on price, category, and payment type

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/ecommerce-return-prediction.git
   cd ecommerce-return-prediction
