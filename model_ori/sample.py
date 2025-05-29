import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
f = "C:/Users/sivab/Downloads/updated_irrigation_model (1).csv"
df = pd.read_csv(f)

# Features and Targets
X = df.drop(columns=["Irrigation Needed", "Irrigation Duration (mins)", "Water Flow Percent"])
y_class = df["Irrigation Needed"].values
y_duration = df["Irrigation Duration (mins)"].values.reshape(-1, 1)
y_flow = df["Water Flow Percent"].values.reshape(-1, 1)

# Identify categorical and numerical columns
categorical_columns = ["Crop Type"]
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

X_processed = preprocessor.fit_transform(X)
joblib.dump(preprocessor, "preprocessor.pkl")  # Save the preprocessor

# Normalize Target Variables
scaler_duration = MinMaxScaler()
scaler_flow = MinMaxScaler()

y_duration_scaled = scaler_duration.fit_transform(y_duration)
y_flow_scaled = scaler_flow.fit_transform(y_flow)

joblib.dump(scaler_duration, "scaler_duration.pkl")
joblib.dump(scaler_flow, "scaler_flow.pkl")

# Train-test split
X_train, X_test, y_train_class, y_test_class, y_train_duration, y_test_duration, y_train_flow, y_test_flow = train_test_split(
    X_processed, y_class, y_duration_scaled, y_flow_scaled, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_class_tensor = torch.tensor(y_train_class, dtype=torch.long)  # Classification
y_train_duration_tensor = torch.tensor(y_train_duration, dtype=torch.float32)  # Regression
y_train_flow_tensor = torch.tensor(y_train_flow, dtype=torch.float32)  # Regression

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_class_tensor = torch.tensor(y_test_class, dtype=torch.long)
y_test_duration_tensor = torch.tensor(y_test_duration, dtype=torch.float32)
y_test_flow_tensor = torch.tensor(y_test_flow, dtype=torch.float32)

# Define Unified Model with More Neurons
class UnifiedFFN(nn.Module):
    def __init__(self, input_dim):
        super(UnifiedFFN, self).__init__()
        self.shared_fc1 = nn.Linear(input_dim, 128)
        self.shared_bn1 = nn.BatchNorm1d(128)
        self.shared_fc2 = nn.Linear(128, 64)
        self.shared_bn2 = nn.BatchNorm1d(64)

        # Classification head (Irrigation Needed: 0 or 1)
        self.class_head = nn.Linear(64, 2)

        # Regression heads (Irrigation Duration & Water Flow %)
        self.duration_head = nn.Linear(64, 1)
        self.flow_head = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.shared_bn1(self.shared_fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.shared_bn2(self.shared_fc2(x)))
        x = self.dropout(x)

        class_out = self.class_head(x)  # Classification
        duration_out = self.duration_head(x)  # Regression
        flow_out = self.flow_head(x)  # Regression

        return class_out, duration_out, flow_out

# Initialize Model
input_dim = X_train.shape[1]
model = UnifiedFFN(input_dim)

# Losses
class_criterion = nn.CrossEntropyLoss()
regression_criterion = nn.MSELoss()

# Optimizers (Separate for Classification & Regression)
optimizer_class = optim.Adam(model.class_head.parameters(), lr=0.001)
optimizer_reg = optim.Adam(list(model.duration_head.parameters()) + list(model.flow_head.parameters()), lr=0.0005)

# Training Loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch_class = y_train_class_tensor[i:i+batch_size]
        y_batch_duration = y_train_duration_tensor[i:i+batch_size].unsqueeze(1)
        y_batch_flow = y_train_flow_tensor[i:i+batch_size].unsqueeze(1)

        optimizer_class.zero_grad()
        optimizer_reg.zero_grad()

        class_pred, duration_pred, flow_pred = model(X_batch)

        class_loss = class_criterion(class_pred, y_batch_class)
        duration_loss = regression_criterion(duration_pred, y_batch_duration)
        flow_loss = regression_criterion(flow_pred, y_batch_flow)

        total_loss = class_loss + duration_loss + flow_loss
        total_loss.backward()

        optimizer_class.step()
        optimizer_reg.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Evaluate Model
with torch.no_grad():
    class_pred, duration_pred, flow_pred = model(X_test_tensor)

    # Classification Accuracy
    _, class_predicted = torch.max(class_pred, 1)
    class_accuracy = (class_predicted == y_test_class_tensor).sum().item() / len(y_test_class_tensor)

    # Reverse Scaling for Regression Predictions
    duration_pred_unscaled = scaler_duration.inverse_transform(duration_pred.numpy())
    flow_pred_unscaled = scaler_flow.inverse_transform(flow_pred.numpy())

    duration_mse = regression_criterion(torch.tensor(duration_pred_unscaled, dtype=torch.float32).squeeze(), y_test_duration_tensor).item()
    flow_mse = regression_criterion(torch.tensor(flow_pred_unscaled, dtype=torch.float32).squeeze(), y_test_flow_tensor).item()

    print(f'Test Accuracy (Irrigation Needed): {class_accuracy:.4f}')
    print(f'Irrigation Duration MSE: {duration_mse:.4f}')
    print(f'Water Flow Percentage MSE: {flow_mse:.4f}')

# Save the Model
torch.save(model.state_dict(), "models/unified_model.pth")
print("Unified Model saved successfully.")
