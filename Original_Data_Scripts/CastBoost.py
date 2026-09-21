# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 01:51:40 2026

@author: Jams
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor


# ======================================
# 1. LOAD DATA
# ======================================
df = pd.read_excel(r"New Volume D/Imputed Data/CatBoost/SW 277 CatBoost.xlsx")

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
# 3. CATBOOST MODEL
# ======================================
model = CatBoostRegressor(

    # Number of boosting iterations
    iterations=500,

    # Learning rate
    learning_rate=0.05,

    # Tree depth
    depth=6,

    # Loss Function
    loss_function='RMSE',

    # Evaluation Metric
    eval_metric='RMSE',

    # L2 Regularization
    l2_leaf_reg=3,

    # Randomness
    random_strength=1,

    # Bootstrap
    bootstrap_type='Bayesian',

    # Bagging Temperature
    bagging_temperature=1,

    # Feature sampling
    rsm=0.80,

    # Border Count
    border_count=254,

    # Leaf estimation
    leaf_estimation_method='Newton',

    # Overfitting Detector
    od_type='Iter',
    od_wait=50,

    # Random seed
    random_seed=42,

    # Thread
    thread_count=-1,

    # Silent Mode
    verbose=False
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
    r"277_LW_Prediction_Result_CatBoost.xlsx",
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
    r"277_LW_Performance_Result_CatBoost.xlsx",
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
    r"277_LW_Feature_Importance_CatBoost.xlsx",
    index=False
)


# ======================================
# 10. PRINT RESULTS
# ======================================
print("\nPerformance Result:")
print(performance)

print("\nFiles saved:")
print("277_LW_Prediction_Result_CatBoost.xlsx")
print("277_LW_Performance_Result_CatBoost.xlsx")
print("277_LW_Feature_Importance_CatBoost.xlsx")
