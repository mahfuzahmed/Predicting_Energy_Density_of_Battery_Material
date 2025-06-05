import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded dataset
file_path = 'Data/Merged_Cleaned/Distinct_Name_All_Properties_Merged_V3.csv'
df = pd.read_csv(file_path)

# Display basic info and first few rows
df.info(), df.head()

# Drop unnecessary columns
df_cleaned = df.drop(columns=["Compound_name", "Extracted_name"])

le = LabelEncoder()
df_cleaned["Type"] = le.fit_transform(df_cleaned["Type"])

# Remove outliers in the target column using IQR
Q1 = df_cleaned["Energy_in_Watt_hour_per_kg"].quantile(0.25)
Q3 = df_cleaned["Energy_in_Watt_hour_per_kg"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_trimmed = df_cleaned[
    (df_cleaned["Energy_in_Watt_hour_per_kg"] >= lower_bound) &
    (df_cleaned["Energy_in_Watt_hour_per_kg"] <= upper_bound)
    ]

# Separate features and target
X = df_trimmed.drop(columns=["Energy_in_Watt_hour_per_kg"])
y = df_trimmed["Energy_in_Watt_hour_per_kg"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Scale features for models that require it (SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Helper function to evaluate and plot results
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"--- {model_name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")

    # For confusion matrix, we bin the target and predictions
    bins = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 10)
    y_true_binned = np.digitize(y_true, bins)
    y_pred_binned = np.digitize(y_pred, bins)

    cm = confusion_matrix(y_true_binned, y_pred_binned)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title(f"{model_name} - Confusion Matrix (Binned)")
    plt.xlabel("Predicted Bin")
    plt.ylabel("Actual Bin")
    plt.show()


def plot_pred_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Predictions vs. Actual")
    plt.tight_layout()
    plt.show()


# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
evaluate_model(y_test, lr_preds, "Linear Regression")
plot_pred_vs_actual(y_test, lr_preds, "Linear Regression")

# Linear Regression Feature Importance (Coefficients)
coefficients = pd.Series(lr_model.coef_, index=X.columns)
plt.figure(figsize=(10, 6))
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Importance - Linear Regression (Coefficients)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# 2. Random Forest Regressor
# n_estimators: number of trees in the forest
# max_depth: controls the depth of each tree to prevent overfitting
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
evaluate_model(y_test, rf_preds, "Random Forest Regressor")
plot_pred_vs_actual(y_test, rf_preds, "Random Forest Regressor")

# Random Forest Feature Importance
importances_rf = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importances_rf.sort_values().plot(kind='barh')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 3. Support Vector Regressor
# C: regularization parameter (trade-off between error and margin)
# epsilon: defines the margin of tolerance
svr_model = SVR(C=10, epsilon=0.1, kernel='rbf')
svr_model.fit(X_train, y_train)
svr_preds = svr_model.predict(X_test)
evaluate_model(y_test, svr_preds, "Support Vector Regressor")
plot_pred_vs_actual(y_test, svr_preds, "Support Vector Regressor")

# 4. XGBoost Regressor
# n_estimators: number of boosting rounds
# learning_rate: step size shrinkage used in update to prevent overfitting
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
evaluate_model(y_test, xgb_preds, "XGBoost Regressor")
plot_pred_vs_actual(y_test, xgb_preds, "XGBoost Regressor")

# XGBoost Feature Importance
importances_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importances_xgb.sort_values().plot(kind='barh')
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 5. Tuned Support Vector Regressor (with scaling)
svr_param_grid = {
    'C': [1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['rbf']
}
svr_grid = GridSearchCV(SVR(), svr_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid.fit(X_train_scaled, y_train)
svr_best = svr_grid.best_estimator_
svr_preds = svr_best.predict(X_test_scaled)
evaluate_model(y_test, svr_preds, "Tuned SVR (Scaled)")
plot_pred_vs_actual(y_test, svr_preds, "Tuned SVR (Scaled)")

# 6. Tuned XGBoost Regressor
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2]
}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_param_grid, cv=5, scoring='neg_mean_squared_error',
                        n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_preds = xgb_best.predict(X_test)
evaluate_model(y_test, xgb_preds, "Tuned XGBoost Regressor")
plot_pred_vs_actual(y_test, svr_preds, "Tuned XGBoost Regressor")
