import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset
data_path = "C:/Users/sivab/OneDrive/Documents/final_modified_energy_demand.csv"
df = pd.read_csv(data_path)

# Identify categorical and numerical columns
categorical_features = ["Season"]
numerical_features = ["Year","Economic_Growth", "GDP_Growth", "Population_Growth", "Energy_Prices"]

# Define features and target variable
X = df.drop(columns=["Energy_Demand"], errors='ignore')
y = df["Energy_Demand"]

# One-hot encode categorical features and scale numerical features
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scale', StandardScaler(), numerical_features)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the features
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

# Linear Regression
lr_model = LinearRegression()
lr_mae, lr_mse, lr_rmse, lr_r2 = evaluate_model(lr_model)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_mae, dt_mse, dt_rmse, dt_r2 = evaluate_model(dt_model)

# Support Vector Regressor
svr_model = SVR(kernel='rbf')
svr_mae, svr_mse, svr_rmse, svr_r2 = evaluate_model(svr_model)

# Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_mae, gbr_mse, gbr_rmse, gbr_r2 = evaluate_model(gbr_model)

print("Linear Regression - MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, R2 Score: {:.4f}".format(lr_mae, lr_mse, lr_rmse, lr_r2))
print("Decision Tree Regressor - MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, R2 Score: {:.4f}".format(dt_mae, dt_mse, dt_rmse, dt_r2))
print("Support Vector Regressor - MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, R2 Score: {:.4f}".format(svr_mae, svr_mse, svr_rmse, svr_r2))
print("Gradient Boosting Regressor - MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, R2 Score: {:.4f}".format(gbr_mae, gbr_mse, gbr_rmse, gbr_r2))
