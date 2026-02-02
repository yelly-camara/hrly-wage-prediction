# Hourly Wage Prediction & Fairness Audit

This project uses Statistics Canada Labour Force Survey (LFS) data to predict hourly wages. It features a CatBoost predictive model and an analysis of how fairly the model performs across different demographic groups.

## Live API Demo
The model is hosted as a FastAPI service on Render.

Swagger UI (Interactive Docs): https://hrly-wage-prediction.onrender.com/docs

Note: Because I am using a free hosting tier, the first request may take 30-60 seconds to "wake up" the server.

## Methodology

1. Data Processing

Target Group: Filtered the data to focus only on employees, as they are the only group with hourly wage information available.

Cleaning: Removed outliers and unusual values.

Categorical Data: Used CatBoost’s built-in tools to handle categorical variables like Occupation and Province without needing extra encoding steps.

2. Feature Selection & Model Results

Feature Optimization: I used SHAP (Shapley Additive Explanations) to identify which variables had the strongest impact on wages.

API Design: To keep the API user-friendly, I only included the 10 most important features. This prevents users from having to input an excessive amount of information to get a prediction.

Accuracy: The final model achieved a Root Mean Square Error (RMSE) of $5.40.

3. Fairness Audit

I performed a fairness check to see if the model's accuracy changes for different groups of people.

Tool: Used the Fairlearn library for the analysis.

Goal: Checked for "Predictive Parity"—meaning I looked at whether the model is equally accurate for different genders, provinces, and education levels.

Note: This is an analytical audit; the API provides the raw prediction based on the trained model.