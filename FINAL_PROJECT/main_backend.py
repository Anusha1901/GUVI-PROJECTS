from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare AI API",
    description="API for Healthcare AI/ML predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models at startup
classification_model = None
classification_scaler = None
classification_label_encoder = None
regression_pipeline = None
clustering_model = None
clustering_scaler = None
clustering_features = None
clustering_optimal_k = None
pneumonia_model = None
sentiment_model = None
sentiment_vectorizer = None

@app.on_event("startup")
async def load_models():
    global classification_model, classification_scaler, classification_label_encoder
    global regression_pipeline, clustering_model, clustering_scaler
    global clustering_features, clustering_optimal_k
    global pneumonia_model, sentiment_model, sentiment_vectorizer
    
    try:
        # Classification models
        classification_model = joblib.load('MODELS/randomforest_classification_model.pkl')
        classification_scaler = joblib.load('MODELS/scaler_classification.pkl')
        classification_label_encoder = joblib.load('MODELS/label_encoder_classification.pkl')
        
        # Regression model
        regression_pipeline = joblib.load('MODELS/best_model_pipeline_regression.pkl')
        
        # Clustering models
        clustering_model = joblib.load('MODELS/kmeans_model_clustering.pkl')
        clustering_scaler = joblib.load('MODELS/scaler_clustering.pkl')
        clustering_features = joblib.load('MODELS/features_to_cluster.pkl')
        clustering_optimal_k = joblib.load('MODELS/optimal_k_clustering.pkl')
        
        # Deep learning model
        pneumonia_model = keras.models.load_model('MODELS/best_model.h5')
        
        # Sentiment models
        sentiment_model = joblib.load('MODELS/sentiment_model.pkl')
        sentiment_vectorizer = joblib.load('MODELS/vectorizer_sentiment.pkl')
        
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")

# ============= REQUEST MODELS =============

class ClassificationRequest(BaseModel):
    age: int
    gender: str
    hypertension: str
    diabetes: str
    diastolic_bp: float
    hdl_cholesterol: float
    ldl_cholesterol: float
    systolic_bp: float
    total_cholesterol: float

class RegressionRequest(BaseModel):
    encounter_class: str
    reason_description: str
    total_claim_cost: float
    payer_coverage: float
    cond_count: int
    proc_count: int
    med_count: int

class ClusteringRequest(BaseModel):
    age: int
    gender: str
    healthcare_expenses: float
    healthcare_coverage: float
    num_conditions: int
    num_procedures: int
    num_medications: int

class SentimentRequest(BaseModel):
    feedback_text: str

# ============= API ENDPOINTS =============

@app.get("/")
async def root():
    return {
        "message": "Healthcare AI API",
        "version": "1.0.0",
        "endpoints": {
            "classification": "/predict/risk",
            "regression": "/predict/los",
            "clustering": "/cluster/assign",
            "pneumonia": "/predict/pneumonia",
            "sentiment": "/analyze/sentiment"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "classification": classification_model is not None,
            "regression": regression_pipeline is not None,
            "clustering": clustering_model is not None,
            "pneumonia": pneumonia_model is not None,
            "sentiment": sentiment_model is not None
        }
    }

# ============= CLASSIFICATION ENDPOINT =============

@app.post("/predict/risk")
async def predict_disease_risk(request: ClassificationRequest):
    try:
        if classification_model is None or classification_scaler is None:
            raise HTTPException(status_code=503, detail="Classification model not loaded")
        
        # Encode inputs
        gender_encoded = 1 if request.gender.lower() == "male" else 0
        hypertension_encoded = 1 if request.hypertension.lower() == "yes" else 0
        diabetes_encoded = 1 if request.diabetes.lower() == "yes" else 0
        
        # Prepare numeric features for scaling
        numeric_features = np.array([[
            request.age,
            request.diastolic_bp,
            request.hdl_cholesterol,
            request.ldl_cholesterol,
            request.systolic_bp,
            request.total_cholesterol
        ]])
        
        # Scale numeric features
        numeric_scaled = classification_scaler.transform(numeric_features)
        
        # Combine scaled numeric + categorical
        input_scaled = np.concatenate([
            numeric_scaled[:, 0:1],
            [[gender_encoded]],
            [[hypertension_encoded]],
            [[diabetes_encoded]],
            numeric_scaled[:, 1:]
        ], axis=1)
        
        # Predict
        prediction = int(classification_model.predict(input_scaled)[0])
        probability = classification_model.predict_proba(input_scaled)[0].tolist()
        
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        confidence = probability[prediction] * 100
        
        return {
            "prediction": risk_level,
            "risk_score": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "low_risk": round(probability[0] * 100, 2),
                "high_risk": round(probability[1] * 100, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= REGRESSION ENDPOINT =============

@app.post("/predict/los")
async def predict_length_of_stay(request: RegressionRequest):
    try:
        if regression_pipeline is None:
            raise HTTPException(status_code=503, detail="Regression model not loaded")
        
        # Prepare input dataframe
        input_data = pd.DataFrame({
            'ENCOUNTERCLASS': [request.encounter_class],
            'REASONDESCRIPTION': [request.reason_description],
            'TOTAL_CLAIM_COST': [request.total_claim_cost],
            'PAYER_COVERAGE': [request.payer_coverage],
            'cond_count': [request.cond_count],
            'proc_count': [request.proc_count],
            'med_count': [request.med_count]
        })
        
        # Predict (log-transformed)
        los_log_pred = regression_pipeline.predict(input_data)[0]
        
        # Reverse log transform
        los_pred = float(np.expm1(los_log_pred))
        
        # Categorize stay
        if los_pred < 3:
            stay_category = "Short Stay"
        elif los_pred < 7:
            stay_category = "Medium Stay"
        else:
            stay_category = "Long Stay"
        
        return {
            "predicted_los_days": round(los_pred, 1),
            "stay_category": stay_category,
            "estimated_cost": round(los_pred * 1500, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= CLUSTERING ENDPOINT =============

@app.post("/cluster/assign")
async def assign_patient_cluster(request: ClusteringRequest):
    try:
        if clustering_model is None or clustering_scaler is None:
            raise HTTPException(status_code=503, detail="Clustering model not loaded")
        
        # Encode gender
        gender_encoded = 1 if request.gender.lower() == "male" else 0
        
        # Prepare input
        input_data = np.array([[
            request.age,
            gender_encoded,
            request.healthcare_expenses,
            request.healthcare_coverage,
            request.num_conditions,
            request.num_procedures,
            request.num_medications
        ]])
        
        # Scale
        input_scaled = clustering_scaler.transform(input_data)
        
        # Predict cluster
        cluster = int(clustering_model.predict(input_scaled)[0])
        
        # Cluster descriptions (customize based on your analysis)
        cluster_info = {
            0: "Low-Cost Routine Care",
            1: "Moderate Care",
            2: "High-Intensity Care",
            3: "Complex Chronic Care"
        }
        
        cluster_name = cluster_info.get(cluster, f"Cluster {cluster}")
        
        return {
            "cluster_id": cluster,
            "cluster_name": cluster_name,
            "optimal_k": int(clustering_optimal_k) if clustering_optimal_k else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= PNEUMONIA DETECTION ENDPOINT =============

@app.post("/predict/pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    try:
        if pneumonia_model is None:
            raise HTTPException(status_code=503, detail="Pneumonia model not loaded")
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 224x224
        image_resized = image.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(image_resized)
        img_array = img_array / 255.0
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Predict
        prediction = pneumonia_model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Class: 0 = NORMAL, 1 = PNEUMONIA
        if confidence > 0.5:
            result = "PNEUMONIA"
            result_confidence = confidence * 100
        else:
            result = "NORMAL"
            result_confidence = (1 - confidence) * 100
        
        return {
            "prediction": result,
            "confidence": round(result_confidence, 2),
            "raw_score": round(confidence, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= SENTIMENT ANALYSIS ENDPOINT =============

@app.post("/analyze/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        if sentiment_model is None or sentiment_vectorizer is None:
            raise HTTPException(status_code=503, detail="Sentiment model not loaded")
        
        # Vectorize input
        text_vectorized = sentiment_vectorizer.transform([request.feedback_text])
        
        # Predict
        prediction = int(sentiment_model.predict(text_vectorized)[0])
        probability = sentiment_model.predict_proba(text_vectorized)[0].tolist()
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[prediction] * 100
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "probabilities": {
                "negative": round(probability[0] * 100, 2),
                "positive": round(probability[1] * 100, 2)
            },
            "feedback_stats": {
                "word_count": len(request.feedback_text.split()),
                "char_count": len(request.feedback_text)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= MODEL INFO ENDPOINT =============

@app.get("/models/info")
async def get_model_info():
    return {
        "classification": {
            "model": "Random Forest Classifier",
            "features": ["Age", "Gender", "Hypertension", "Diabetes", "BP", "Cholesterol"],
            "target": "High Risk / Low Risk"
        },
        "regression": {
            "model": "Pipeline with Preprocessing",
            "features": ["Encounter Class", "Reason", "Cost", "Coverage", "Conditions", "Procedures", "Medications"],
            "target": "Length of Stay (days)"
        },
        "clustering": {
            "model": "K-Means Clustering",
            "features": ["Age", "Gender", "Expenses", "Coverage", "Conditions", "Procedures", "Medications"],
            "optimal_k": int(clustering_optimal_k) if clustering_optimal_k else None
        },
        "pneumonia": {
            "model": "Convolutional Neural Network",
            "input_size": "224x224 grayscale",
            "classes": ["Normal", "Pneumonia"]
        },
        "sentiment": {
            "model": "Text Classification Model",
            "classes": ["Negative", "Positive"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)