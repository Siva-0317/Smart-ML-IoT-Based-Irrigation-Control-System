from pymongo import MongoClient
import pymongo.errors

uri = "your mongodb url"

client = MongoClient(uri, serverSelectionTimeoutMS=60000)  # 60-second timeout

try:
    print("Attempting to connect...")
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    print("Databases:", client.list_database_names())
except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Timeout Error: {e}")
except pymongo.errors.ConfigurationError as e:
    print(f"Configuration Error (check credentials or URI): {e}")
except Exception as e:
    print(f"Other Error: {e}")
finally:
    client.close()
