from django.urls import path
from .views import predict_and_store, receive_image_data  # Import both views

urlpatterns = [
    path('irrigation_app/receive_sensor_data/',predict_and_store, name='receive_sensor_data'),  # Sensor Data
    path('irrigation_app/receive_image_data/', receive_image_data, name='receive_image_data'),  # Image Data
]
