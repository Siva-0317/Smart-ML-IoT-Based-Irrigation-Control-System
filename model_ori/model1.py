import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib  # For saving the preprocessor
import os

# Load dataset
f = "C:/Users/sivab/Downloads/updated_irrigation_model (1).csv"
df = pd.read_csv(f)

# Features and Target
X = df.drop(columns=["Irrigation Needed"])  # Keep X as a DataFrame initially
y = df["Irrigation Needed"].values

# Identify categorical and numerical columns (while X is a DataFrame)
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(exclude=['object']).columns


# Create Preprocessor (including OneHotEncoder for Crop Type)
preprocessor1 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # OneHotEncoder for Crop Type
    ])

# Fit and transform the data using the preprocessor
X_processed = preprocessor1.fit_transform(X)

# Save the preprocessor
joblib.dump(preprocessor1, "preprocessor1.pkl")
print("Preprocessor saved as preprocessor1.pkl")

# Train-test split (using the *processed* data)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors (using the *processed* data)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define Feedforward Neural Network
class FFN(nn.Module):
    def __init__(self, input_dim):  # Corrected: __init__
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 2)    # Output 2 classes (0 or 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model
input_dim = X_train.shape[1]
model = FFN(input_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Correct loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate model
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy:.4f}')

# Save the model and preprocessor
model_dir = 'models'  # Directory to save models
os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
torch.save(model.state_dict(), os.path.join(model_dir, 'model1.pth'))
print("Model saved successfully.")
import pickle
with open("C:/Users/sivab/AppData/Local/Programs/Python/Python311/Smart_Irrigation/models/preprocessor1.pkl", 'wb') as f:
    pickle.dump(preprocessor1,f)
print("Preprocessor saved successfully.")

print("Model and preprocessor saved successfully in the 'models' directory.")