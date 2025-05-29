💧 Smart ML-Based Irrigation Control System
7th Place – CMR Hackfest 2.0 (2025)

AI + IoT powered system for sustainable and precise irrigation.

🔍 Overview
This project presents a smart irrigation solution that integrates IoT sensors, Machine Learning models, and real-time weather forecasting to intelligently control irrigation for agricultural fields. 
The system ensures optimal water usage, reducing waste and increasing crop productivity.

🧠 Core Objectives
Automate irrigation decisions using ML.

Optimize water flow and duration based on crop and soil conditions.

Integrate weather data for improved prediction accuracy.

Communicate between hardware and backend via LoRa & Django.

🔬 ML Models
📌 Model 1: Irrigation Need Classifier
Type: Feed-Forward Neural Network (PyTorch)

Inputs: Soil moisture, temperature, humidity, rainfall, crop type

Output: Binary prediction (Irrigation Needed: Yes/No)

📌 Model 2: Water Flow & Duration Predictor
Type: XGBoost Regression

Inputs: Same as Model 1

Outputs:

Water Flow Percentage

Irrigation Duration (in minutes)

🌐 Technologies Used
ML Frameworks: PyTorch, XGBoost

IoT Hardware: Arduino Uno, Arduino Uno R4 WiFi

Backend: Django + MongoDB

Communication: LoRa (data relay), WiFi (data to Django)

Weather API: OpenWeatherMap for real-time forecast integration

💡 Features
🌿 Real-time prediction of irrigation need

🚿 Dynamic water flow and timing optimization

☁️ Weather-based adjustment using forecast API

📡 End-to-end data relay from field sensors to Django backend

📊 Logging of predictions and sensor data into MongoDB

🏅 Recognition
This project was showcased at CMR Hackfest 2.0 (2025) and secured a spot in the Top 10, placing 7th overall out of 1500+ teams.

👨‍💻 Team
Developed by a passionate team of undergrads:

-Sivakumar Balaji

-Risha Jayaraj

-Aswath S

-Shreyya P

-Harshitha KG

