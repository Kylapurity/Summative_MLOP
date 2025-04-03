import os
import shutil
import zipfile
import io
import json
import base64
import uvicorn
from datetime import datetime
from typing import List
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from PIL import Image
from pydantic import BaseModel  # Added for request body validation

# ================== MongoDB Setup ================== #
MONGO_CONNECTION_STRING = "mongodb+srv://kihiupurity29:nVk1e36Hu4HDWN27@cluster0.f6b4yet.mongodb.net/retraining_db?retryWrites=true&w=majority"
mongo_client = MongoClient(MONGO_CONNECTION_STRING)
db = mongo_client['retraining_db']
images_collection = db['training_images']
history_collection = db['training_history']

# ================== INITIAL SETUP ================== #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

# ================== CONFIGURATION ================== #
UPLOAD_FOLDER = "uploaded_data"
VISUALIZATION_DIR = "visualizations"
MODEL_DIR = "models"
KERAS_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip'}
MAX_IMAGES_PER_BATCH = 20

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# ================== CLASS NAMES ================== #
CLASS_NAMES = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry_healthy', 'Cherry_Powdery_mildew', 'Cherry_healthy',
    'Corn_Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_Common_rust',
    'Corn_Northern_Leaf_Blight', 'Corn_healthy', 'Grape_Black_rot',
    'Grape_Esca_Black_Measles', 'Grape_Leaf_blight_Isariopsis_Leaf_Spot', 'Grape_healthy',
    'Orange_Haunglongbing_Citrus_greening', 'Peach_Bacterial_spot', 'Peach_healthy',
    'Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight',
    'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew',
    'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus', 'Tomato_healthy'
]

# ================== MODEL LOADING ================== #
try:
    if not os.path.exists(KERAS_PATH):
        raise FileNotFoundError(f"Model file not found at {KERAS_PATH}")
    model = tf.keras.models.load_model(KERAS_PATH)
except Exception as e:
    raise RuntimeError(f"Model initialization failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Classifier API",
    description="API for classifying plant diseases from leaf images",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Utility functions
def preprocess_image(img_bytes: bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_upload_file(file: UploadFile, destination: str) -> None:
    try:
        with open(destination, "wb") as buffer:
            while chunk := await file.read(8192):
                buffer.write(chunk)
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise

def resize_image(img_bytes: bytes, max_size=(128, 128)) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize(max_size, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    img.save(output, format=img.format if img.format in ['PNG', 'JPEG'] else 'JPEG', quality=85)
    return output.getvalue()

def save_visualizations(y_true, y_pred_classes, target_names, history=None):
    """Save enhanced visualizations including classification report, confusion matrix, and training plots."""
    # 1. Beautified Classification Report
    class_report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)
    
    # Prepare data for the table
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for cls in target_names:
        if cls in class_report:
            rows.append([
                cls,
                f"{class_report[cls]['precision']:.2f}",
                f"{class_report[cls]['recall']:.2f}",
                f"{class_report[cls]['f1-score']:.2f}",
                f"{class_report[cls]['support']}"
            ])
    # Add accuracy row
    total_support = sum(class_report[cls]['support'] for cls in target_names if cls in class_report)
    rows.append(["Accuracy", "", "", f"{class_report['accuracy']:.2f}", f"{total_support}"])

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, len(target_names) * 0.6 + 2))  # Adjust height based on number of classes
    ax.axis('off')
    
    # Create a table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#4CAF50'] * len(headers),  # Green header background
        colWidths=[0.4, 0.15, 0.15, 0.15, 0.15],  # Adjust column widths
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  # Scale table for better readability
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4CAF50')  # Green background for headers
        else:  # Data rows
            cell.set_text_props(color='black')
            cell.set_facecolor('#F5F5F5' if row % 2 == 0 else '#FFFFFF')  # Alternating row colors
        cell.set_edgecolor('#D3D3D3')  # Light gray borders
    
    # Add title
    plt.title("Classification Report", fontsize=18, weight='bold', pad=20, color='#333333')
    
    # Save the figure
    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "classification_report.png"),
        bbox_inches='tight',
        dpi=300,
        facecolor='white',
        edgecolor='none'
    )
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(max(10, len(target_names)), max(10, len(target_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "confusion_matrix.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Training and Validation Loss
    if history and 'loss' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, "loss_plot.png"), bbox_inches='tight', dpi=300)
        plt.close()

    # 4. Training and Validation Accuracy
    if history and 'accuracy' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, "accuracy_plot.png"), bbox_inches='tight', dpi=300)
        plt.close()
        
# Request model for /retrain_from_db
class RetrainFromDBRequest(BaseModel):
    retraining_batch: str
    learning_rate: float = 0.0001
    epochs: int = 5

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Plant Disease Classifier API",
        "status": "operational",
        "model_status": "loaded" if model else "not loaded",
        "class_count": len(CLASS_NAMES),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=dict, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    try:
        start_time = datetime.now()
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}{file.filename.replace(' ', '')}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        await save_upload_file(file, filepath)
        
        img_bytes = open(filepath, 'rb').read()
        img = preprocess_image(img_bytes)
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        
        return {
            "filename": safe_filename,
            "prediction": predicted_class,
            "confidence": float(confidence),
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain", tags=["Training"])
async def retrain(files: List[UploadFile] = File(...),
                  learning_rate: float = 0.0001,
                  epochs: int = 5):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_FOLDER, "new_data")
    temp_model_path = os.path.join(MODEL_DIR, "temp_model.keras")
    os.makedirs(new_data_dir, exist_ok=True)
    
    try:
        start_time = datetime.now()
        image_paths = []
        extracted_dirs = []
        retraining_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_count = 0
        
        # Check MongoDB space and clean up if necessary
        total_size = (db.command("collstats", "training_images")["size"] + 
                      db.command("collstats", "training_history")["size"]) / (1024 * 1024)
        if total_size > 400:
            oldest_batch = images_collection.find_one(sort=[("uploaded_at", 1)])
            if oldest_batch:
                oldest_batch_id = oldest_batch["retraining_batch"]
                images_collection.delete_many({"retraining_batch": oldest_batch_id})
                history_collection.delete_many({"retraining_batch": oldest_batch_id})
                print(f"Deleted batch {oldest_batch_id} to free space")
        
        for file in files:
            if not allowed_file(file.filename):
                continue
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            await save_upload_file(file, file_path)
            
            if file.filename.endswith(".zip"):
                extract_dir = os.path.join(UPLOAD_FOLDER, f"extract_{os.path.splitext(file.filename)[0]}")
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                extracted_dirs.append(extract_dir)
                os.remove(file_path)
                
                for subdir in ['train', 'val', 'test']:
                    subdir_path = os.path.join(extract_dir, subdir)
                    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                        for item in os.listdir(subdir_path):
                            item_path = os.path.join(subdir_path, item)
                            if os.path.isdir(item_path) and item not in ['__MACOSX']:
                                target_dir = os.path.join(new_data_dir, item)
                                os.makedirs(target_dir, exist_ok=True)
                                for img in os.listdir(item_path):
                                    if allowed_file(img) and image_count < MAX_IMAGES_PER_BATCH:
                                        img_path = os.path.join(item_path, img)
                                        with open(img_path, 'rb') as f:
                                            image_data = f.read()
                                            image_data = resize_image(image_data)
                                            base64_image = base64.b64encode(image_data).decode('utf-8')
                                            images_collection.insert_one({
                                                "split": subdir,
                                                "class_name": item,
                                                "filename": img,
                                                "image_base64": base64_image,
                                                "retraining_batch": retraining_batch,
                                                "uploaded_at": datetime.now().isoformat()
                                            })
                                        shutil.copy(img_path, os.path.join(target_dir, img))
                                        image_count += 1
            else:
                if image_count < MAX_IMAGES_PER_BATCH:
                    with open(file_path, 'rb') as f:
                        img_bytes = f.read()
                        img_array = preprocess_image(img_bytes)
                        prediction = model.predict(img_array)
                        label_index = np.argmax(prediction)
                        label = CLASS_NAMES[label_index]
                        img_bytes = resize_image(img_bytes)
                        base64_image = base64.b64encode(img_bytes).decode('utf-8')
                        images_collection.insert_one({
                            "split": "train",
                            "class_name": label,
                            "filename": file.filename,
                            "image_base64": base64_image,
                            "retraining_batch": retraining_batch,
                            "uploaded_at": datetime.now().isoformat()
                        })
                    label_dir = os.path.join(new_data_dir, label)
                    os.makedirs(label_dir, exist_ok=True)
                    shutil.copy(file_path, os.path.join(label_dir, file.filename))
                    image_paths.append(file_path)
                    image_count += 1
        
        if image_count == 0:
            raise HTTPException(status_code=400, detail="No valid images found for retraining")
        
        # Common retraining logic (factored out for reuse)
        class_counts = {}
        for class_dir in os.listdir(new_data_dir):
            class_path = os.path.join(new_data_dir, class_dir)
            if os.path.isdir(class_path):
                image_count_dir = len([f for f in os.listdir(class_path) if allowed_file(f)])
                if image_count_dir >= 2:
                    class_counts[class_dir] = image_count_dir
                else:
                    shutil.rmtree(class_path)
        
        if not class_counts:
            raise HTTPException(status_code=400, detail="No valid classes with sufficient data found")
        
        target_names = list(class_counts.keys())
        use_validation = all(count >= 4 for count in class_counts.values())
        
        if use_validation:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            validation_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
        else:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                shuffle=True
            )
            validation_generator = None
        
        model.save(temp_model_path)
        working_model = tf.keras.models.load_model(temp_model_path)
        
        num_classes = len(train_generator.class_indices)
        if working_model.output_shape[-1] != num_classes:
            base_model = tf.keras.Sequential(working_model.layers[:-1])
            base_model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name="output_dense"))
            working_model = base_model
        
        num_layers = len(working_model.layers)
        freeze_until = int(num_layers * 0.98)
        for i, layer in enumerate(working_model.layers):
            layer.trainable = i >= freeze_until
        
        working_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if use_validation else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if use_validation else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        if use_validation:
            history = working_model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator)),
                validation_steps=max(1, len(validation_generator))
            )
        else:
            history = working_model.fit(
                train_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator))
            )
        
        if use_validation:
            validation_generator.reset()
            y_pred = working_model.predict(validation_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = validation_generator.classes
            target_names = list(validation_generator.class_indices.keys())
        else:
            train_generator.reset()
            y_pred = working_model.predict(train_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = train_generator.classes
            target_names = list(train_generator.class_indices.keys())
        
        class_report = classification_report(
            y_true,
            y_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        
        save_visualizations(y_true, y_pred_classes, target_names, history)
        
        working_model.save(KERAS_PATH)
        model = tf.keras.models.load_model(KERAS_PATH)
        
        CLASS_NAMES = list(train_generator.class_indices.keys())
        with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
            json.dump(CLASS_NAMES, f)
        
        history_doc = {
            "retraining_batch": retraining_batch,
            "epochs_run": len(history.history['loss']),
            "history": {
                "loss": history.history['loss'],
                "accuracy": history.history['accuracy'],
                "val_loss": history.history.get('val_loss', []),
                "val_accuracy": history.history.get('val_accuracy', [])
            },
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
            "class_counts": class_counts,
            "image_count": image_count
        }
        history_collection.insert_one(history_doc)
        
        class_metrics = {}
        for class_name in target_names:
            if class_name in class_report:
                class_metrics[class_name] = {
                    "precision": float(class_report[class_name]['precision']),
                    "recall": float(class_report[class_name]['recall']),
                    "f1_score": float(class_report[class_name]['f1-score']),
                    "support": int(class_report[class_name]['support'])
                }
        
        training_accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None
        validation_accuracy = float(history.history['val_accuracy'][-1]) if use_validation and 'val_accuracy' in history.history else None
        
        base_url = "https://summative-mlop-f4v0.onrender.com"
        response_content = {
            "status": "success",
            "metrics": {
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "class_metrics": class_metrics
            },
            "visualization_files": {
                "classification_report": f"{base_url}/visualizations/classification_report.png",
                "confusion_matrix": f"{base_url}/visualizations/confusion_matrix.png",
                "loss_plot": f"{base_url}/visualizations/loss_plot.png",
                "accuracy_plot": f"{base_url}/visualizations/accuracy_plot.png"
            },
            "retraining_batch": retraining_batch,
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat(),
            "images_stored": image_count
        }
        return response_content
    
    except OperationFailure as e:
        if "you are over your space quota" in str(e):
            raise HTTPException(status_code=507, detail="MongoDB storage quota exceeded. Old data cleared, please retry.")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
    
    finally:
        for extract_dir in extracted_dirs:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

from typing import Optional
from pydantic import BaseModel

class RetrainFromDBRequest(BaseModel):
    retraining_batch: Optional[str] = None  # Optional field
    learning_rate: float = 0.0001
    epochs: int = 5

@app.post("/retrain_from_db", tags=["Training"])
async def retrain_from_db(request: RetrainFromDBRequest):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_FOLDER, "new_data")
    temp_model_path = os.path.join(MODEL_DIR, "temp_model.keras")
    os.makedirs(new_data_dir, exist_ok=True)
    
    try:
        start_time = datetime.now()
        retraining_batch = request.retraining_batch
        learning_rate = request.learning_rate
        epochs = request.epochs
        
        # Pull images from MongoDB
        images_collection = db['train']
        if retraining_batch:
            images = images_collection.find({"retraining_batch": retraining_batch})
        else:
            images = images_collection.find({})  # Fetch all images if no batch specified
        image_count = 0
        
        # Write images to filesystem
        for img_doc in images:
            image_data = img_doc["image"]
            
            # Handle different possible structures of image_data
            if isinstance(image_data, dict) and "$binary" in image_data:
                # Case 1: Structured as {"$binary": {"base64": "...", "subType": "00"}}
                base64_string = image_data["$binary"]["base64"]
                img_bytes = base64.b64decode(base64_string)
            elif isinstance(image_data, str):
                # Case 2: Direct base64 string
                img_bytes = base64.b64decode(image_data)
            elif isinstance(image_data, bytes):
                # Case 3: Raw bytes (already decoded)
                img_bytes = image_data
            else:
                raise ValueError(f"Unexpected image data format: {type(image_data)}")
            
            class_name = img_doc["class_name"].replace("", "")
            filename = img_doc["filename"]
            
            target_dir = os.path.join(new_data_dir, class_name)
            os.makedirs(target_dir, exist_ok=True)
            
            img_path = os.path.join(target_dir, filename)
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
            image_count += 1
        
        if image_count == 0:
            raise HTTPException(status_code=404, detail="No images found in the database")
        
        # Common retraining logic (unchanged from previous version)
        class_counts = {}
        for class_dir in os.listdir(new_data_dir):
            class_path = os.path.join(new_data_dir, class_dir)
            if os.path.isdir(class_path):
                image_count_dir = len([f for f in os.listdir(class_path) if allowed_file(f)])
                if image_count_dir >= 2:
                    class_counts[class_dir] = image_count_dir
                else:
                    shutil.rmtree(class_path)
        
        if not class_counts:
            raise HTTPException(status_code=400, detail="No valid classes with sufficient data found")
        
        target_names = list(class_counts.keys())
        use_validation = all(count >= 4 for count in class_counts.values())
        
        if use_validation:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            validation_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
        else:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                shuffle=True
            )
            validation_generator = None
        
        model.save(temp_model_path)
        working_model = tf.keras.models.load_model(temp_model_path)
        
        num_classes = len(train_generator.class_indices)
        if working_model.output_shape[-1] != num_classes:
            base_model = tf.keras.Sequential(working_model.layers[:-1])
            base_model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name="output_dense"))
            working_model = base_model
        
        num_layers = len(working_model.layers)
        freeze_until = int(num_layers * 0.98)
        for i, layer in enumerate(working_model.layers):
            layer.trainable = i >= freeze_until
        
        working_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if use_validation else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if use_validation else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        if use_validation:
            history = working_model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator)),
                validation_steps=max(1, len(validation_generator))
            )
        else:
            history = working_model.fit(
                train_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator))
            )
        
        if use_validation:
            validation_generator.reset()
            y_pred = working_model.predict(validation_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = validation_generator.classes
            target_names = list(validation_generator.class_indices.keys())
        else:
            train_generator.reset()
            y_pred = working_model.predict(train_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = train_generator.classes
            target_names = list(train_generator.class_indices.keys())
        
        class_report = classification_report(
            y_true,
            y_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        
        save_visualizations(y_true, y_pred_classes, target_names, history)
        
        working_model.save(KERAS_PATH)
        model = tf.keras.models.load_model(KERAS_PATH)
        
        CLASS_NAMES = list(train_generator.class_indices.keys())
        with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
            json.dump(CLASS_NAMES, f)
        
        history_doc = {
            "retraining_batch": retraining_batch or "all_data",
            "epochs_run": len(history.history['loss']),
            "history": {
                "loss": history.history['loss'],
                "accuracy": history.history['accuracy'],
                "val_loss": history.history.get('val_loss', []),
                "val_accuracy": history.history.get('val_accuracy', [])
            },
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
            "class_counts": class_counts,
            "image_count": image_count
        }
        history_collection.insert_one(history_doc)
        
        class_metrics = {}
        for class_name in target_names:
            if class_name in class_report:
                class_metrics[class_name] = {
                    "precision": float(class_report[class_name]['precision']),
                    "recall": float(class_report[class_name]['recall']),
                    "f1_score": float(class_report[class_name]['f1-score']),
                    "support": int(class_report[class_name]['support'])
                }
        
        training_accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None
        validation_accuracy = float(history.history['val_accuracy'][-1]) if use_validation and 'val_accuracy' in history.history else None
        
        base_url = "https://summative-mlop-f4v0.onrender.com"
        response_content = {
            "status": "success",
            "metrics": {
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "class_metrics": class_metrics
            },
            "visualization_files": {
                "classification_report": f"{base_url}/visualizations/classification_report.png",
                "confusion_matrix": f"{base_url}/visualizations/confusion_matrix.png",
                "loss_plot": f"{base_url}/visualizations/loss_plot.png",
                "accuracy_plot": f"{base_url}/visualizations/accuracy_plot.png"
            },
            "retraining_batch": retraining_batch or "all_data",
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat(),
            "images_used": image_count
        }
        return response_content
    
    except OperationFailure as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
    
    finally:
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
# Server Startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300,
        reload=False
    )