import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('C:/Users/sivab/OneDrive/Documents/mod2_updated.csv')

X = data.drop(columns=["Irrigation Duration (mins)", "Water Flow Percent"])
print("Columns in X before preprocessing:", X.columns)
y_duration = data["Irrigation Duration (mins)"]
y_flow = data["Water Flow Percent"]

# Preprocessing
categorical_features = [ "Crop Type"]
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train_duration, y_test_duration = train_test_split(X_processed, y_duration, test_size=0.2, random_state=42)
_, _, y_train_flow, y_test_flow = train_test_split(X_processed, y_flow, test_size=0.2, random_state=42)

# Convert training data to DMatrix for XGBoost
dtrain_duration = xgb.DMatrix(X_train, label=y_train_duration)
dtrain_flow = xgb.DMatrix(X_train, label=y_train_flow)

# XGBoost models with GPU support
duration_model = xgb.train({'tree_method': 'hist', 'device': 'cuda'}, dtrain_duration)
flow_model = xgb.train({'tree_method': 'hist', 'device': 'cuda'}, dtrain_flow)

# Convert test data to DMatrix for prediction
dtest = xgb.DMatrix(X_test)

# Prediction
y_pred_duration = duration_model.predict(dtest)
mse_duration = mean_squared_error(y_test_duration, y_pred_duration)
r2_duration = r2_score(y_test_duration, y_pred_duration)

y_pred_flow = flow_model.predict(dtest)
mse_flow = mean_squared_error(y_test_flow, y_pred_flow)
r2_flow = r2_score(y_test_flow, y_pred_flow)
print(f"Irrigation Duration - Mean Squared Error: {mse_duration}, R² Score: {r2_duration}")
print(f"Water Flow Percentage - Mean Squared Error: {mse_flow}, R² Score: {r2_flow}")

# Save the models and preprocessor
joblib.dump(duration_model, 'models/model2d.pkl')
joblib.dump(flow_model, 'models/model2f.pkl')
import pickle
with open("C:/Users/sivab/AppData/Local/Programs/Python/Python311/Smart_Irrigation/models/preprocessor2.pkl", 'wb') as f:
    pickle.dump(preprocessor, f)
print("Models and preprocessor saved successfully.")
