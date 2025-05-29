import os
import torch
import pickle
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io

# Paths to models and preprocessors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL1_PATH = os.path.join(BASE_DIR, 'models', 'model1.pth')
MODEL2_DURATION_PATH = os.path.join(BASE_DIR, 'models', 'model2d.pkl')
MODEL2_FLOW_PATH = os.path.join(BASE_DIR, 'models', 'model2f.pkl')
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet50_plantvillage.pth')

PREPROCESSOR1_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor1.pkl')
PREPROCESSOR2_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor2.pkl')

# Define Model 1: Feed-Forward Neural Network
class FFN(nn.Module):
    def __init__(self, input_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 2)  # Output 2 classes (0 or 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load models

def load_model(model_path, model_type="torch", input_dim=7):
    if model_type == "torch":
        model = FFN(input_dim=input_dim)
        state_dict = torch.load(model_path, map_location=torch.device('cuda'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    elif model_type == "pickle":
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Load preprocessors
def load_preprocessor(preprocessor_path):
    with open(preprocessor_path, 'rb') as f:
        return pickle.load(f)

# Initialize models and preprocessors
model1 = load_model(MODEL1_PATH, model_type="torch")
model2_duration = load_model(MODEL2_DURATION_PATH, model_type="pickle")
model2_flow = load_model(MODEL2_FLOW_PATH, model_type="pickle")
preprocessor1 = load_preprocessor(PREPROCESSOR1_PATH)
preprocessor2 = load_preprocessor(PREPROCESSOR2_PATH)

# Load ResNet-50 for Disease Classification
disease_model = models.resnet50(weights=None)
disease_model.fc = nn.Linear(disease_model.fc.in_features, 39)  # Assuming 39 disease classes
disease_model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=torch.device('cuda')))
disease_model.eval()

def transform_image(image):
    image = Image.open(io.BytesIO(image.read())).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_disease(image_file):
    image_tensor = transform_image(image_file)
    with torch.no_grad():
        outputs = disease_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()