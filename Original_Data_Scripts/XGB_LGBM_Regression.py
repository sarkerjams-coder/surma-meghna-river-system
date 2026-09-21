import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ======================================
# 1. LOAD DATA
# ======================================
df = pd.read_excel(r"D:/RiverXGB_LGBM/SW 277 linear Regression.xlsx")

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Ensure numeric only (important for Excel datasets)
X = X.select_dtypes(include=[np.number])

# Drop NaN safely
data = pd.concat([X, y], axis=1).dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# ======================================
# 2. TRAIN-TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)


# ======================================
# 3. MODEL DEFINITIONS
# ======================================
models = {
    "XGB": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.80,
        colsample_bytree=0.80,
        gamma=0,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    ),
    "LGBM": LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}


# ======================================
# 4. TRAIN, PREDICT, AND EVALUATE
# ======================================
all_predictions = []
all_performance = []
all_feature_importance = []

for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_pred_df = pd.DataFrame({
        "Model": model_name,
        "Dataset": "Train",
        "Actual": y_train.values,
        "Predicted": y_train_pred
    })

    test_pred_df = pd.DataFrame({
        "Model": model_name,
        "Dataset": "Test",
        "Actual": y_test.values,
        "Predicted": y_test_pred
    })

    all_predictions.append(train_pred_df)
    all_predictions.append(test_pred_df)

    all_performance.append({
        "Model": model_name,
        "Dataset": "Train",
        "R2 Score": train_r2,
        "RMSE": train_rmse,
        "MAE": train_mae
    })

    all_performance.append({
        "Model": model_name,
        "Dataset": "Test",
        "R2 Score": test_r2,
        "RMSE": test_rmse,
        "MAE": test_mae
    })

    if hasattr(model, "feature_importances_"):
        feature_importance_df = pd.DataFrame({
            "Model": model_name,
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        all_feature_importance.append(feature_importance_df)


# ======================================
# 5. SAVE PREDICTIONS
# ======================================
prediction_result = pd.concat(all_predictions, ignore_index=True)
prediction_result.to_excel(
    r"277_LW_Prediction_Result_XGB_LGBM.xlsx",
    index=False
)


# ======================================
# 6. SAVE PERFORMANCE METRICS
# ======================================
performance = pd.DataFrame(all_performance)
performance.to_excel(
    r"277_LW_Performance_Result_XGB_LGBM.xlsx",
    index=False
)


# ======================================
# 7. SAVE FEATURE IMPORTANCE
# ======================================
feature_importance = pd.concat(all_feature_importance, ignore_index=True)
feature_importance.to_excel(
    r"277_LW_Feature_Importance_XGB_LGBM.xlsx",
    index=False
)


# ======================================
# 8. PRINT RESULTS IN SPYDER CONSOLE
# ======================================
print("\nPerformance Result:")
print(performance)

print("\nFiles saved:")
print("Prediction_Result_XGB_LGBM.xlsx")
print("Performance_Result_XGB_LGBM.xlsx")
print("Feature_Importance_XGB_LGBM.xlsx")
