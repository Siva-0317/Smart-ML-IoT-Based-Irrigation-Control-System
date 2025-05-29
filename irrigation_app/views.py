import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from smart_irrigation.settings import db
from .models import model1, model2_duration, model2_flow, preprocessor1, preprocessor2 , disease_model, transform_image
from PIL import Image
import torch
from datetime import datetime
from bson.objectid import ObjectId
import json
import pandas as pd
import numpy as np
import xgboost as xgb

# OpenWeatherMap API settings
OPENWEATHERMAP_API_KEY = 'your-api-key' 
OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather_data(latitude, longitude):
    #Fetch weather data from OpenWeatherMap API.
    try:
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': OPENWEATHERMAP_API_KEY,
            'units': 'metric'  # Temperature in Celsius
        }
        response = requests.get(OPENWEATHERMAP_URL, params=params)
        response.raise_for_status()
        data = response.json()
        atmospheric_temp = data['main']['temp']  # Temperature
        rainfall = data.get('rain', {}).get('1h', 0)  # Rainfall in the last hour
        return atmospheric_temp, rainfall
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None, None  # Return None on failure


def store_sensor_data(sensor_data):
    #Store sensor data in MongoDB.
    sensor_data["timestamp"] = datetime.utcnow()
    result = db.sensor_data.insert_one(sensor_data)
    return result.inserted_id


def store_model_predictions(predictions, sensor_data_id):
    #Store ML predictions in MongoDB.
   
    predictions["sensor_data_id"] = ObjectId(sensor_data_id)
    predictions["timestamp"] = datetime.utcnow()
    db.model_predictions.insert_one(predictions) # Import pandas for DataFrame conversion

@csrf_exempt
def predict_and_store(request):
    if request.method == 'POST':
        try:
            # Parse JSON payload
            data = json.loads(request.body)
            print("Received Data:", data)  # Debugging line to print the received data
            
            # Extract and validate data
            soil_moisture = float(data.get('Soil Moisture',0.0))
            rainfall = float(data.get('Rainfall',0.0))
            humidity = float(data.get('Humidity',0.0))
            soil_temp = float(data.get('Temperature',0.0))
            crop_type = data.get('Crop Type', "Rabi")  # Use string directly if available, else default to "Rabi"
            latitude = float(data.get('Latitude', 0.0))
            longitude = float(data.get('Longitude', 0.0))

            # Fetch real-time weather data
            atmospheric_temp, rainfall = get_weather_data(latitude, longitude)
            if atmospheric_temp is None or rainfall is None:
                return JsonResponse({"error": "Failed to fetch weather data"}, status=500)

            # Store sensor data in MongoDB
            sensor_data = {
                "soil_moisture": soil_moisture,
                "rainfall": rainfall,
                "humidity": humidity,
                "soil_temp": soil_temp,
                "crop_type": crop_type,
                "latitude": latitude,
                "longitude": longitude,
            }
            sensor_data_id = store_sensor_data(sensor_data)
            #print("Stored Sensor Data ID:", sensor_data_id)  # Debugging line

            # Prepare data for Model 1
            try:
                # Combine all inputs into a DataFrame for transformation
                model1_input_data = {
                    "Soil Moisture": [soil_moisture],
                    "Rainfall": [rainfall],
                    "Humidity": [humidity],
                    "Temperature": [soil_temp],
                    "Crop Type": [crop_type]
                }
                df_model1_input = pd.DataFrame(model1_input_data)
                print("Before Transformation DataFrame:", df_model1_input)  # Debugging line
                
                model1_input = preprocessor1.transform(df_model1_input)
                print("After Transformation:", model1_input)  # Debugging line

                model1_input = torch.tensor(model1_input, dtype=torch.float32)
                print("Model 1 Input Tensor:", model1_input)  # Debugging line

                # Model 1 Prediction (Irrigation Need)
                # Ensure output is a single value
                with torch.no_grad():
                    output = model1(model1_input)  # Get raw logits
                    irrigation_need = torch.argmax(output, dim=1).item()  # Get predicted class (0 or 1)

                #print("Irrigation Need:", irrigation_need)  # Debugging line

            except Exception as e:
                print(f"Error in Model 1 prediction: {e}")  # Debugging line
                return JsonResponse({"error": f"Error in Model 1 prediction: {e}"}, status=400)

            duration = 0
            water_flow = 0
            if irrigation_need >= 0.5:  # If irrigation is needed
                # Prepare data for Model 2
                try:
                    # Combine all inputs into a DataFrame for transformation
                    model2_input_data = {
                        "Soil Moisture": [soil_moisture],
                        "Rainfall": [rainfall],
                        "Humidity": [humidity],
                        "Temperature": [atmospheric_temp],
                        "Crop Type": [crop_type]
                    }
                    
                    df_model2_input = pd.DataFrame(model2_input_data)
                    print("Before Transformation Model 2 DataFrame:", df_model2_input)  # Debugging line

                    model2_input = preprocessor2.transform(df_model2_input)
                    print("After Transformation Model 2:", model2_input)  # Debugging line

                    dmatrix_model2_input = xgb.DMatrix(model2_input)
                    print("Model 2 Input Tensor:", model2_input)  # Debugging line

                    # Predict duration and water flow
                    duration = model2_duration.predict(dmatrix_model2_input)[0]
                    water_flow = model2_flow.predict(dmatrix_model2_input)[0]
                    #print("Duration:", duration)  # Debugging line
                    #print("Water Flow:", water_flow)  # Debugging line

                except Exception as e:
                    print(f"Error in Model 2 prediction: {e}")  # Debugging line
                    return JsonResponse({"error": f"Error in Model 2 prediction: {e}"}, status=400)

            # Store predictions in MongoDB
            predictions = {
                "irrigation_needed": int(irrigation_need >= 0.5),
                "water_flow_percentage": float(water_flow),
                "irrigation_duration": float(duration),
            }
            store_model_predictions(predictions, sensor_data_id)
            #print("Predictions Stored:", predictions)  # Debugging line

            # Convert ObjectId to string for JSON serialization
            response_data = { "sensor_data_id": str(sensor_data_id), 
                             **predictions, 
                             "timestamp": datetime.now().isoformat() } 
            # Convert any additional ObjectId fields to strings 
            for key, value in response_data.items(): 
                if isinstance(value, ObjectId): 
                    response_data[key] = str(value)
            #print("Response Data:", response_data)  # Debugging line

            # Return response
            return JsonResponse(response_data)
        except Exception as e:
            print(f"Error: {e}")  # Debugging line
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import transforms
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import os
from datetime import datetime
import google.generativeai as genai

# Define paths
MODEL_PATH = "resnet50_plantvillage.pth"
DATA_DIR = "./PlantVillage_split"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
train_dir = os.path.join(DATA_DIR, "train")
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print(f"Loaded {num_classes} classes.")

# Load the pretrained model
disease_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = disease_model.fc.in_features
disease_model.fc = nn.Linear(num_ftrs, num_classes)
disease_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
disease_model = disease_model.to(device)
disease_model.eval()

# **Apply the same transformations as used in training**
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import os
import json
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming these are defined elsewhere or imported
# transform, device, disease_model, class_names (list of 39 disease names)

# Gemini API configuration
GEMINI_API_KEY = "your-api-key"  # Move to settings.py or env vars
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# BERT configuration
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(bert_device)
bert_model.eval()

# JSON file for storing predictions and recommendations
SAVE_DIR = 'D:/plant_disease_data/'  # Adjust this path as needed
JSON_FILE = os.path.join(SAVE_DIR, 'disease_recommendations.json')

# Ensure directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load existing data from JSON file
def load_stored_data():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
        return data
    return []

# Extract embeddings and recommendations for similarity comparison
stored_data = load_stored_data()
stored_disease_embeddings = [entry["disease_embedding"] for entry in stored_data]
stored_recommendations = [entry["recommendation"] for entry in stored_data]

# Function to get BERT embedding
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(bert_device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()  # Convert to list for JSON

# Function to query Gemini for recommendations
def query_gemini(disease):
    try:
        prompt = f"Provide recommendations and measures to cure the plant disease, give it short and sweet: {disease}"
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return "No recommendations available from Gemini."
    except Exception as e:
        return f"Error fetching recommendations from Gemini: {str(e)}"

# Function to update JSON file
def update_json_file(predicted_disease, probabilities, recommendation, disease_embedding, rec_embedding):
    data_entry = {
        "predicted_disease": predicted_disease,
        "probabilities": probabilities,  # List of 39 class probabilities
        "recommendation": recommendation,
        "disease_embedding": disease_embedding,
        "recommendation_embedding": rec_embedding,
        "timestamp": datetime.now().isoformat()
    }
    
    # Load existing data or initialize new list
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    # Append new entry and save
    data.append(data_entry)
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    # Update in-memory stored data
    stored_disease_embeddings.append(disease_embedding)
    stored_recommendations.append(recommendation)

@csrf_exempt
def receive_image_data(request):
    if request.method == 'POST':
        try:
            if 'image' not in request.FILES:
                return JsonResponse({"error": "No image file received"}, status=400)

            # Read and preprocess the image
            image_file = request.FILES['image']
            image = Image.open(image_file).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)  # Apply transformation

            with torch.no_grad():
                outputs = disease_model(image)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy().tolist()  # Get probabilities for all 39 classes
                predicted_idx = torch.argmax(outputs, dim=1).item()

            predicted_disease = class_names[predicted_idx]

            # Generate BERT embedding for the predicted disease
            disease_embedding = get_bert_embedding(predicted_disease)
            disease_embedding_np = np.array(disease_embedding).reshape(1, -1)

            # Check similarity with stored embeddings
            recommendation = None
            if stored_disease_embeddings:  # If there are stored embeddings
                stored_embeddings_np = np.array(stored_disease_embeddings)
                similarities = cosine_similarity(disease_embedding_np, stored_embeddings_np)[0]
                max_similarity = similarities.max()
                confidence_threshold = 0.95  # Adjust this threshold as needed

                if max_similarity > confidence_threshold:
                    best_match_idx = similarities.argmax()
                    recommendation = stored_recommendations[best_match_idx]
                    print(f"Reusing stored recommendation (similarity: {max_similarity:.4f})")
                else:
                    print(f"Similarity too low ({max_similarity:.4f}), querying Gemini...")
                    recommendation = query_gemini(predicted_disease)
            else:
                print("No stored data, querying Gemini...")
                recommendation = query_gemini(predicted_disease)

            # Generate BERT embedding for the recommendation
            rec_embedding = get_bert_embedding(recommendation)

            # Store in JSON file
            update_json_file(predicted_disease, probabilities, recommendation, disease_embedding, rec_embedding)

            return JsonResponse({
                "predicted_disease": predicted_disease,
                "recommendations": recommendation,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    
    return JsonResponse({"error": "Invalid request method"}, status=405)


