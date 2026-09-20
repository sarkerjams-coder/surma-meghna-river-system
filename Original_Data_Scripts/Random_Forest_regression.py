# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 00:57:30 2026

@author: Jams
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# ======================================
# 1. LOAD DATA
# ======================================
df = pd.read_excel(r"D:/River System/Imputed data/Random_Forest_regression/SW 277 WL.xlsx")

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Ensure numeric only
X = X.select_dtypes(include=[np.number])

# Drop NaN safely
data = pd.concat([X, y], axis=1).dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# ======================================
# 2. TRAIN-TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ======================================
# 3. RANDOM FOREST MODEL
# ======================================
model = RandomForestRegressor(

    n_estimators=500,          # Number of trees

    criterion="squared_error", # Regression criterion

    max_depth=None,            # Allow full tree growth

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_features="sqrt",

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    bootstrap=True,

    oob_score=True,

    n_jobs=-1,

    random_state=42,

    verbose=0
)


# ======================================
# 4. TRAIN MODEL
# ======================================
model.fit(X_train, y_train)


# ======================================
# 5. PREDICTION
# ======================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ======================================
# 6. EVALUATION
# ======================================
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)


# ======================================
# 7. SAVE PREDICTIONS
# ======================================
prediction_result = pd.concat([

    pd.DataFrame({
        "Dataset": "Train",
        "Actual": y_train.values,
        "Predicted": y_train_pred
    }),

    pd.DataFrame({
        "Dataset": "Test",
        "Actual": y_test.values,
        "Predicted": y_test_pred
    })

], ignore_index=True)

prediction_result.to_excel(
    r"277WL_Prediction_Result_RF.xlsx",
    index=False
)


# ======================================
# 8. SAVE PERFORMANCE
# ======================================
performance = pd.DataFrame({

    "Dataset": ["Train", "Test"],

    "R2 Score": [train_r2, test_r2],

    "RMSE": [train_rmse, test_rmse],

    "MAE": [train_mae, test_mae]

})

performance.to_excel(
    r"277WL_Performance_Result_RF.xlsx",
    index=False
)


# ======================================
# 9. FEATURE IMPORTANCE
# ======================================
feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

}).sort_values(by="Importance", ascending=False)

feature_importance.to_excel(
    r"277WL_Feature_Importance_RF.xlsx",
    index=False
)


# ======================================
# 10. PRINT RESULTS
# ======================================
print("\nPerformance Result:")
print(performance)

print("\nOOB Score:")
print(model.oob_score_)

print("\nFiles saved:")
print("277_WL_Prediction_Result_RF.xlsx")
print("277_WL_Performance_Result_RF.xlsx")
print("277_WL_Feature_Importance_RF.xlsx")