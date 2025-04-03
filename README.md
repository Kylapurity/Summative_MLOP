# Smart Farm

## Project Overview
Smart Farm is a web-based application that helps farmers diagnose plant diseases accurately and provides treatment recommendations. By leveraging machine learning, our mission is to reduce crop losses, minimize pesticide use, and promote sustainable farming practices through accurate and timely plant disease diagnosis.
#### Web app Link
- *Public URl from Samrt : https://smartfarm-livid.vercel.app/
- #### Railway Link
- *Swager :https://summativemlop-production.up.railway.app/docs#/
- #### Dataset
- *Dataset :https://www.kaggle.com/code/imtkaggleteam/plant-diseases-detection-pytorch/input

## Features
- **Disease Prediction**: Users can upload images of plants, and the model predicts the disease and suggests treatments.
- **Data Management**: Model training data is stored in MongoDB.
- **Retraining Feature**: The system allows retraining on newly uploaded data.
- **Scalability**: Built using FastAPI for the backend and React with Tailwind CSS for the frontend.
- **Deployment**: The website is hosted on Vercel.
- **API Endpoints**:
  - `/predict` – Predict plant disease from an uploaded image.
  - `/upload` – Upload new training data.
  - `/build` – Build dataset for retraining.
  - `/retrain` – Retrain the model with new data.

## Directory Structure
```
Project_name/
│
├── README.md
│
├── notebook/
│   ├── project_name.ipynb  # Jupyter Notebook for analysis
│
├── src/
│   ├── preprocessing.py     # Data preprocessing
│   ├── model.py            # Model definition
│   ├── prediction.py       # Prediction logic
│
├── data/
│   ├── train/              # Training dataset
│   └── test/               # Testing dataset
│
└── models/
    ├── _model_name.pkl     # Trained model (Pickle format)
    ├── _model_name.tf      # TensorFlow model
```

## Technologies Used
### Frontend
- React.js
- Tailwind CSS
- Vercel (Deployment)

### Backend
- FastAPI (API Development)
- MongoDB (Database for storing images and metadata)
- Python (Machine Learning Model)
- TensorFlow / Scikit-learn (Model Training & Prediction)

## How to Run
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Node.js & npm
- MongoDB
- Virtual environment (optional but recommended)

### Backend Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-url.git
   cd project_name
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
   ```sh
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. API will be available at: `http://localhost:8000/docs`

### Frontend Setup
1. Navigate to the frontend directory:
   ```sh
   cd frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```
4. Open `http://localhost:3000` in your browser.

## Deployment
### Backend Deployment
- Deploy FastAPI on a cloud service (e.g., AWS, Google Cloud, or Heroku)
- Ensure MongoDB is hosted on a cloud provider (e.g., MongoDB Atlas)

### Frontend Deployment
- Deploy React on Vercel:
  ```sh
  vercel deploy
  ```

## Model Training & Retraining
### Initial Training
1. Place training data in `data/train/`
2. Run the model training script:
   ```sh
   python src/model.py
   ```
3. The trained model will be saved in the `models/` directory.

### Retraining
- Upload new data via the `/upload` API endpoint.
- Trigger retraining using the `/retrain` endpoint.

## Future Enhancements
- Expand from 5 to 38 plant disease classes.
- Improve UI for better user experience.
- Implement real-time prediction with WebSockets.



