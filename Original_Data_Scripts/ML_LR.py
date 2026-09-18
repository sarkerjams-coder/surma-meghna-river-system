# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:03:24 2026

@author:Azizul_islam
"""

# ===========================
# Multiple Linear Regression
# Prediction + Performance
# ===========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ======================================
# 1. Load Dataset
# ======================================



df = pd.read_excel("D:/AA Thesis task/Imputed data/LR File/SW 267 linear Regressin.xlsx")

X=df.iloc[:,1:-1]

y=df.iloc[:,-1]


# # ======================================
# # 2. Define Target and Predictors
# # ======================================

# target_column = "267_Target"

# X = df.drop(columns=[target_column])
# y = df[target_column]

# ======================================
# 3. Train-Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)

# ======================================
# 4. Train MLR Model
# ======================================

model = LinearRegression()

model.fit(X_train, y_train)

# ======================================
# 5. Prediction
# ======================================

y_pred = model.predict(X_test)

# ======================================
# 6. Performance Metrics
# ======================================

R2 = r2_score(y_test, y_pred)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

MAE = mean_absolute_error(y_test, y_pred)

# ======================================
# 7. Save Prediction Result
# ======================================

prediction_result = pd.DataFrame({

    "Actual": y_test.values,
    "Predicted": y_pred

})

prediction_result.to_excel(
    "Prediction_Result.xlsx",
    index=False
)

# ======================================
# 8. Save Performance Result
# ======================================

performance = pd.DataFrame({

    "Metric": ["R2 Score", "RMSE", "MAE"],
    "Value": [R2, RMSE, MAE]

})

performance.to_excel(
    "Performance_Result.xlsx",
    index=False
)

# ======================================
# 9. Regression Equation
# ======================================

coefficients = pd.DataFrame({

    "Variable": X.columns,
    "Coefficient": model.coef_

})

coefficients.loc[len(coefficients)] = [
    "Intercept",
    model.intercept_
]

coefficients.to_excel(
    "Regression_Coefficients.xlsx",
    index=False
)

# ======================================
# 10. Print Results
# ======================================

print("="*50)
print("Multiple Linear Regression Completed")
print("="*50)

print(f"R2 Score : {R2:.4f}")
print(f"RMSE     : {RMSE:.4f}")
print(f"MAE      : {MAE:.4f}")

print("\nFiles Saved Successfully")
print("1. Prediction_Result.xlsx")
print("2. Performance_Result.xlsx")
print("3. Regression_Coefficients.xlsx")