import os
from pymongo import MongoClient
from urllib.parse import quote_plus

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Secret key and debug mode
SECRET_KEY = 'your_secret_key'
DEBUG = True
#ALLOWED_HOSTS = ['127.0.0.1','192.168.1.9','192.168.1.7','192.168.53.161','182.76.27.87','192.168.85.161','192.168.136.207'] #'192.168.1.6', 192.168.26.161
ALLOWED_HOSTS = ['*']
# Installed apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'irrigation_app',  # Your app
]

# Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Root URL configuration
ROOT_URLCONF = 'smart_irrigation.urls'

# Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI application
WSGI_APPLICATION = 'smart_irrigation.wsgi.application'

# MongoDB Configuration
# Encode password for safe URI usage
password = "astranova"  # Replace with your actual MongoDB password
encoded_password = quote_plus(password)

# MongoDB connection URI
MONGO_URI = f"mongodb+srv://astra:{encoded_password}@smartirrigationcluster.yavh8.mongodb.net/?retryWrites=true&w=majority&appName=SmartIrrigationCluster"

# Initialize MongoDB client
mongo_client = MongoClient(MONGO_URI)

# Specify the database to use
db = mongo_client["smart_irrigation"]

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

# Default auto field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Custom configurations (Optional)
# Add any additional configurations like log levels, email backends, etc.

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'django_debug.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

#DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB limit
