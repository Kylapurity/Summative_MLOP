import os
import bson
from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB Atlas with explicit TLS settings
connection_string = "mongodb+srv://kihiupurity29:nVk1e36Hu4HDWN27@cluster0.f6b4yet.mongodb.net/retraining_db?retryWrites=true&w=majority&tls=true"
try:
    client = MongoClient(connection_string, serverSelectionTimeoutMS=30000)  # Increase timeout to 30s
    db = client['retraining_db']
    collection = db['train']  # Storing all in 'train' collection
    # Test connection
    client.admin.command('ping')
    print("Connected to MongoDB Atlas successfully!")
    # Test write permission by inserting a dummy document
    test_doc = {"test": "write_permission_check", "timestamp": datetime.now().isoformat()}
    collection.insert_one(test_doc)
    print("Successfully inserted a test document into 'train' collection!")
except Exception as e:
    print(f"Failed to connect to MongoDB Atlas or write to 'train' collection: {e}")
    exit(1)

# Directory containing the extracted data
extract_dir = r'C:\Users\Kyla\Downloads\retrain_test_data\train'

# Verify that the directory exists
if not os.path.exists(extract_dir):
    print(f"Error: Directory not found at {extract_dir}")
    exit(1)
print(f"Processing directory: {extract_dir}")

# Valid image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Counter for tracking inserted images
inserted_count = 0

# Print the directory structure for debugging
print("\nDirectory structure:")
for root, dirs, files in os.walk(extract_dir):
    level = root.replace(extract_dir, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}    {f}")

# Function to process a single image file
def process_image(image_path, split, class_name, image_file):
    global inserted_count
    if image_file.lower().endswith(image_extensions):
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                if not image_data:
                    print(f"Warning: {image_file} is empty, skipping.")
                    return
                doc = {
                    "split": split,
                    "class_name": class_name,
                    "filename": image_file,
                    "image": bson.Binary(image_data),
                    "retraining_batch": "batch_2025_03_31",
                    "uploaded_at": datetime.now().isoformat()
                }
                collection.insert_one(doc)
                inserted_count += 1
                print(f"Stored {image_file} from {split}/{class_name} as BinData (Total inserted: {inserted_count})")
        except Exception as e:
            print(f"Error storing {image_file}: {e}")
    else:
        print(f"Skipped {image_file} (not an image)")

# Traverse the directory structure
for root, dirs, files in os.walk(extract_dir):
    # Determine split and class_name based on the directory structure
    rel_path = os.path.relpath(root, extract_dir)
    path_parts = rel_path.split(os.sep)
    
    # Skip the root directory itself
    if rel_path == '.':
        split = "train"  # Since we're already in the 'train' directory
        class_name = "unknown"  # Default if no class directory
    elif len(path_parts) >= 1:
        # Structure: train/class_name/ (e.g., train/Apple___Cedar_apple_rust/)
        split = "train"
        class_name = path_parts[0]  # First level is the class name
    else:
        split = "train"
        class_name = "unknown"

    # Process files in this directory
    for image_file in files:
        image_path = os.path.join(root, image_file)
        process_image(image_path, split, class_name, image_file)

# Verify the number of documents in the collection
final_count = collection.count_documents({})
print(f"\nTotal documents in 'train' collection: {final_count}")

# Close the connection
client.close()
if inserted_count > 0:
    print("Images stored in MongoDB Atlas for retraining!")
else:
    print("No images were stored in MongoDB Atlas. Check for errors above.")