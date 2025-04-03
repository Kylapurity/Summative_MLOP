import os
import shutil
import zipfile
import io
import json
import base64
from datetime import datetime
from typing import List
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from PIL import Image
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# ================== MongoDB Setup ================== #
MONGO_CONNECTION_STRING = "mongodb+srv://kihiupurity29:nVk1e36Hu4HDWN27@cluster0.f6b4yet.mongodb.net/"
mongo_client = MongoClient(MONGO_CONNECTION_STRING)
db = mongo_client['retraining_db']
images_collection = db['training_images']
history_collection = db['training_history']
zip_collection = db['zip_files']

# ================== INITIAL SETUP ================== #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

# ================== CONFIGURATION ================== #
UPLOAD_FOLDER = "uploaded_data"
VISUALIZATION_DIR = "visualizations"
MODEL_DIR = "models"
KERAS_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip', 'JPG'}
MAX_IMAGES_PER_BATCH = 20
MIN_IMAGES_PER_CLASS = 1

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
    return ('.' in filename and 
            not filename.startswith('._') and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

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

def classify_images_in_directory(directory: str, model, class_names: List[str], split: str):
    predictions = {}
    split_dir = os.path.join(directory, split)
    if not os.path.exists(split_dir):
        logger.debug(f"No {split} directory found at {split_dir}")
        return predictions
    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if os.path.isdir(class_path) and class_name in class_names and class_name != '__MACOSX':
            predictions[class_name] = []
            for img_file in os.listdir(class_path):
                if allowed_file(img_file):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                            img_array = preprocess_image(img_bytes)
                            pred = model.predict(img_array)
                            pred_index = np.argmax(pred, axis=1)[0]
                            confidence = float(np.max(pred))
                            predicted_class = class_names[pred_index]
                            predictions[class_name].append({
                                "filename": img_file,
                                "true_class": class_name,
                                "predicted_class": predicted_class,
                                "confidence": confidence
                            })
                    except Exception as e:
                        logger.error(f"Failed to process {img_path}: {str(e)}")
    return predictions

def save_visualizations(y_true, y_pred_classes, target_names, history=None):
    class_report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True, zero_division=0)
    
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
    total_support = sum(class_report[cls]['support'] for cls in target_names if cls in class_report)
    rows.append(["Accuracy", "", "", f"{class_report['accuracy']:.2f}", f"{total_support}"])

    fig, ax = plt.subplots(figsize=(12, len(target_names) * 0.6 + 2))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#4CAF50'] * len(headers),
        colWidths=[0.4, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4CAF50')
        else:
            cell.set_text_props(color='black')
            cell.set_facecolor('#F5F5F5' if row % 2 == 0 else '#FFFFFF')
        cell.set_edgecolor('#D3D3D3')
    plt.title("Classification Report", fontsize=18, weight='bold', pad=20, color='#333333')
    plt.savefig(os.path.join(VISUALIZATION_DIR, "classification_report.png"), bbox_inches='tight', dpi=300)
    plt.close()

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

# Pydantic model for retrain endpoint
class RetrainRequest(BaseModel):
    retraining_batch: str
    learning_rate: float = 0.0001
    epochs: int = 10

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

@app.post("/upload", tags=["Upload"])
async def upload(files: List[UploadFile] = File(...)):
    try:
        start_time = datetime.now()
        retraining_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        total_size = (db.command("collstats", "training_images")["size"] + 
                      db.command("collstats", "training_history")["size"] + 
                      db.command("collstats", "zip_files")["size"]) / (1024 * 1024)
        if total_size > 400:
            oldest_batch = zip_collection.find_one(sort=[("uploaded_at", 1)])
            if oldest_batch:
                oldest_batch_id = oldest_batch["retraining_batch"]
                zip_collection.delete_many({"retraining_batch": oldest_batch_id})
                images_collection.delete_many({"retraining_batch": oldest_batch_id})
                history_collection.delete_many({"retraining_batch": oldest_batch_id})
                logger.info(f"Deleted batch {oldest_batch_id} to free space")
        
        for file in files:
            if not file.filename.endswith(".zip") or not allowed_file(file.filename):
                logger.warning(f"Skipping file {file.filename}: not a valid ZIP")
                continue
            
            zip_bytes = await file.read()
            zip_base64 = base64.b64encode(zip_bytes).decode('utf-8')
            
            zip_collection.insert_one({
                "filename": file.filename,
                "zip_base64": zip_base64,
                "retraining_batch": retraining_batch,
                "uploaded_at": datetime.now().isoformat()
            })
            logger.info(f"Stored ZIP file {file.filename} in MongoDB with batch {retraining_batch}")
        
        response_content = {
            "status": "success",
            "retraining_batch": retraining_batch,
            "message": "ZIP file(s) uploaded successfully. Use the retrain endpoint with the retraining_batch to process.",
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat()
        }
        return response_content
    
    except OperationFailure as e:
        if "you are over your space quota" in str(e):
            raise HTTPException(status_code=507, detail="MongoDB storage quota exceeded. Old data cleared, please retry.")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Upload failed with exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/retrain", tags=["Training"])
async def retrain(request: RetrainRequest):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_FOLDER, "new_data")
    temp_model_path = os.path.join(MODEL_DIR, "temp_model.keras")
    os.makedirs(new_data_dir, exist_ok=True)
    
    try:
        start_time = datetime.now()
        retraining_batch = request.retraining_batch
        learning_rate = request.learning_rate
        epochs = request.epochs
        
        logger.info(f"Starting retrain for batch: {retraining_batch}")
        
        # Retrieve ZIP file from MongoDB
        zip_doc = zip_collection.find_one({"retraining_batch": retraining_batch})
        if not zip_doc:
            raise HTTPException(status_code=404, detail=f"No ZIP file found for retraining batch {retraining_batch}")
        
        # Decode and extract ZIP file
        zip_bytes = base64.b64decode(zip_doc["zip_base64"])
        zip_path = os.path.join(UPLOAD_FOLDER, f"temp_{retraining_batch}.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        
        extract_dir = os.path.join(UPLOAD_FOLDER, f"extract_{retraining_batch}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_path)
        logger.info(f"Extracted ZIP file for batch {retraining_batch} to {extract_dir}")
        
        # Process extracted data
        class_counts = {}
        all_classes = set()
        for subdir in ['train', 'val']:
            subdir_path = os.path.join(extract_dir, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                logger.debug(f"Extracted {subdir} contents: {os.listdir(subdir_path)}")
                for class_name in os.listdir(subdir_path):
                    class_path = os.path.join(subdir_path, class_name)
                    # Normalize class name by replacing triple underscores with single underscore
                    normalized_class = class_name.replace("", "")
                    # Find matching class name in CLASS_NAMES (case-insensitive)
                    class_name_normalized = next(
                        (cn for cn in CLASS_NAMES if cn.lower() == normalized_class.lower()),
                        None
                    )
                    all_classes.add(class_name)
                    if (os.path.isdir(class_path) and 
                        class_name_normalized and 
                        class_name != '__MACOSX'):
                        target_dir = os.path.join(new_data_dir, subdir, class_name_normalized)
                        os.makedirs(target_dir, exist_ok=True)
                        logger.debug(f"Found class {class_name} (normalized to {class_name_normalized}) at {class_path}")
                        image_count = 0
                        for img in os.listdir(class_path):
                            if allowed_file(img):
                                img_path = os.path.join(class_path, img)
                                shutil.copy(img_path, os.path.join(target_dir, img))
                                image_count += 1
                        logger.debug(f"Class {class_name_normalized} in {subdir} has {image_count} images")
                        if image_count >= MIN_IMAGES_PER_CLASS:
                            class_counts[class_name_normalized] = class_counts.get(class_name_normalized, 0) + image_count
                        else:
                            shutil.rmtree(target_dir)
        
        if not class_counts:
            logger.error(f"No valid classes found. All classes in ZIP: {list(all_classes)}")
            raise HTTPException(status_code=400, detail=f"No valid classes with sufficient data found. All classes: {list(all_classes)}")
        logger.info(f"Processed {sum(class_counts.values())} images from ZIP file. Detected classes: {list(class_counts.keys())}")
        
        # Classify images with pre-trained model against all 38 classes
        logger.debug("Classifying train images with pre-trained model")
        train_predictions = classify_images_in_directory(new_data_dir, model, CLASS_NAMES, "train")
        logger.debug("Classifying val images with pre-trained model")
        val_predictions = classify_images_in_directory(new_data_dir, model, CLASS_NAMES, "val")
        pre_train_predictions = {"train": train_predictions, "val": val_predictions}
        logger.debug(f"Pre-train predictions: {pre_train_predictions}")
        
        # Store images in MongoDB with pre-train predictions
        for split in ["train", "val"]:
            split_predictions = pre_train_predictions[split]
            for class_name, preds in split_predictions.items():
                for pred in preds:
                    img_path = os.path.join(new_data_dir, split, class_name, pred["filename"])
                    logger.debug(f"Storing {img_path} in MongoDB")
                    with open(img_path, 'rb') as f:
                        img_bytes = resize_image(f.read())
                        base64_image = base64.b64encode(img_bytes).decode('utf-8')
                        images_collection.insert_one({
                            "split": split,
                            "class_name": class_name,
                            "filename": pred["filename"],
                            "image_base64": base64_image,
                            "retraining_batch": retraining_batch,
                            "uploaded_at": datetime.now().isoformat(),
                            "pre_train_prediction": pred["predicted_class"],
                            "pre_train_confidence": pred["confidence"]
                        })
                    logger.debug(f"Stored {pred['filename']} in MongoDB with split {split}")
        
        # Prepare data for retraining with only available classes
        target_names = list(class_counts.keys())
        train_dir = os.path.join(new_data_dir, 'train')
        val_dir = os.path.join(new_data_dir, 'val')
        use_validation = os.path.exists(val_dir) and bool(os.listdir(val_dir))
        
        if not os.path.exists(train_dir) or not os.listdir(train_dir):
            logger.warning("No training data found, using all data as training")
            train_dir = new_data_dir
            use_validation = False
        
        logger.info(f"Training with classes: {target_names}, use_validation: {use_validation}")
        
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            validation_split=0.2 if use_validation and not os.path.exists(val_dir) else None
        )
        
        train_generator = data_generator.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            classes=target_names,
            subset='training' if use_validation and not os.path.exists(val_dir) else None,
            shuffle=True
        )
        
        if len(train_generator) == 0:
            raise HTTPException(status_code=400, detail="Training generator is empty")
        
        validation_generator = None
        if use_validation:
            validation_generator = data_generator.flow_from_directory(
                val_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                classes=target_names,
                shuffle=False
            )
            if len(validation_generator) == 0:
                logger.warning("Validation generator is empty, proceeding without validation")
                use_validation = False
        
        # Fine-tune the model with a temporary output layer for the subset
        model.save(temp_model_path)
        working_model = tf.keras.models.load_model(temp_model_path)
        
        num_layers = len(working_model.layers)
        freeze_until = int(num_layers * 0.98)
        for i, layer in enumerate(working_model.layers):
            layer.trainable = i >= freeze_until
        
        # Replace the output layer with one matching the subset
        num_subset_classes = len(target_names)
        working_model.pop()  # Remove the 38-class output layer
        working_model.add(Dense(num_subset_classes, activation='softmax', name='temp_output_layer'))
        
        working_model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
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
        
        logger.info(f"Starting model training with {len(train_generator)} steps per epoch on {num_subset_classes} classes")
        if use_validation and validation_generator:
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
        
        # Evaluate on the subset classes
        if use_validation and validation_generator:
            validation_generator.reset()
            y_pred = working_model.predict(validation_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = validation_generator.classes
        else:
            train_generator.reset()
            y_pred = working_model.predict(train_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = train_generator.classes
        
        class_report = classification_report(
            y_true,
            y_pred_classes,
            target_names=target_names,
            labels=range(len(target_names)),
            output_dict=True,
            zero_division=0
        )
        
        save_visualizations(y_true, y_pred_classes, target_names, history)
        
        # Restore the 38-class output layer with updated weights
        working_model.pop()  # Remove the temporary subset output layer
        working_model.add(Dense(len(CLASS_NAMES), activation='softmax', name='restored_output_layer'))
        
        # Recompile with the original 38-class output
        working_model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        working_model.save(KERAS_PATH)
        model = tf.keras.models.load_model(KERAS_PATH)
        
        # Save training history
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
            "image_count": sum(class_counts.values())
        }
        history_collection.insert_one(history_doc)
        
        class_metrics = {name: class_report[name] for name in target_names if name in class_report}
        training_accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None
        validation_accuracy = float(history.history['val_accuracy'][-1]) if use_validation and 'val_accuracy' in history.history else None
        
        base_url = "https://summativemlop-production.up.railway.app"
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
            "pre_train_predictions": pre_train_predictions,
            "processing_time": str(datetime.now() - start_time),
            "timestamp": datetime.now().isoformat(),
            "images_used": sum(class_counts.values()),
            "classes_detected": target_names
        }
        return response_content
    
    except HTTPException as e:
        raise e
    except OperationFailure as e:
        if "you are over your space quota" in str(e):
            raise HTTPException(status_code=507, detail="MongoDB storage quota exceeded. Old data cleared, please retry.")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Retraining failed with exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
    
    finally:
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if 'extract_dir' in locals() and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

# Server Startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=300,
        reload=False
    )