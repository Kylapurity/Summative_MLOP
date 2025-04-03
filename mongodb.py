import zipfile
import os
import bson
from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB Atlas with explicit TLS settings
connection_string = "mongodb+srv://kihiupurity29:nVk1e36Hu4HDWN27@cluster0.f6b4yet.mongodb.net/retraining_db?retryWrites=true&w=majority&tls=true"
try:
    client = MongoClient(connection_string, serverSelectionTimeoutMS=30000)  # Increase timeout to 30s
    db = client['retraining_db']
    collection = db['valid']
    collection = db['train']  # Storing all in 'valid' collection
    # Test connection
    client.admin.command('ping')
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Failed to connect to MongoDB Atlas: {e}")
    exit(1)

# Extract the zip file
zip_path = r'C:\Users\Kyla\Downloads\Archive.zip'
extract_dir = 'Data'

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Zip file extracted to:", extract_dir)
except FileNotFoundError:
    print(f"Error: Zip file not found at {zip_path}")
    exit(1)
except zipfile.BadZipFile:
    print("Error: Invalid zip file")
    exit(1)

# Valid image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Store images with retraining metadata
for split in os.listdir(extract_dir):
    split_path = os.path.join(extract_dir, split)
    if os.path.isdir(split_path):
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(image_extensions):
                        image_path = os.path.join(class_path, image_file)
                        try:
                            with open(image_path, 'rb') as f:
                                image_data = f.read()
                                collection.insert_one({
                                    "split": split,
                                    "class_name": class_name,
                                    "filename": image_file,
                                    "image": bson.Binary(image_data),
                                    "retraining_batch": "batch_2025_03_31",
                                    "uploaded_at": datetime.now().isoformat()
                                })
                            print(f"Stored {image_file} from {split}/{class_name} as BinData")
                        except Exception as e:
                            print(f"Error storing {image_file}: {e}")
                    else:
                        print(f"Skipped {image_file} (not an image)")

# Close the connection
client.close()
print("Images stored in MongoDB Atlas for retraining!")