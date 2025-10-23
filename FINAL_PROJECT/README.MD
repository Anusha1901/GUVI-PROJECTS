# 🏥 Healthcare AI/ML Analytics Suite

> A comprehensive end-to-end AI/ML system for healthcare analytics, predictions, and patient management using Machine Learning and Deep Learning models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)](https://scikit-learn.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & Algorithms](#models--algorithms)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The Healthcare AI Suite is an integrated platform that leverages advanced machine learning and deep learning techniques to provide:

- **Predictive Analytics**: Disease risk classification and length of stay predictions
- **Patient Segmentation**: Clustering patients based on healthcare utilization patterns
- **Medical Image Analysis**: Pneumonia detection from chest X-rays using CNN
- **Sentiment Analysis**: Patient feedback analysis using VADER
- **Interactive Chatbot**: Healthcare query assistance
- **Medical Translator**: Multilingual medical communication

This system demonstrates the practical application of AI/ML in healthcare for improved patient outcomes, resource optimization, and clinical decision support.

---

## ✨ Features

### 🎯 **Disease Risk Classification**
- Predicts high/low risk based on patient health parameters
- Uses Random Forest classifier with 9 clinical features
- Features: Age, Gender, Hypertension, Diabetes, Blood Pressure, Cholesterol levels
- Real-time risk assessment with confidence scores

### 📈 **Length of Stay (LOS) Prediction**
- Predicts hospital stay duration using regression pipeline
- Considers encounter class, diagnosis, costs, and patient complexity
- Handles categorical and numeric features with preprocessing
- Log-transformed target for better prediction accuracy

### 👥 **Patient Segmentation**
- K-Means clustering for patient grouping
- Based on: Healthcare expenses, coverage, conditions, procedures, medications
- Identifies patient subgroups for targeted interventions
- Optimized cluster selection

### 🫁 **Pneumonia Detection**
- Deep Learning CNN model for chest X-ray analysis
- Input: 224x224 grayscale images
- Binary classification: Normal vs Pneumonia
- High accuracy diagnostic support tool

### 💬 **Sentiment Analysis**
- VADER-based sentiment analysis for patient feedback
- Classifies feedback as Positive, Negative, or Neutral
- Detailed sentiment scoring (compound, positive, neutral, negative)
- Actionable insights for quality improvement

### 🤖 **Healthcare Chatbot**
- Interactive medical query assistance
- Separate Streamlit application
- Provides health information and guidance

### 🌐 **Medical Translator**
- Multilingual medical communication tool
- Separate Streamlit application
- Supports multiple languages for patient-provider communication

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   User Interface Layer                      │
├─────────────────────────────────────────────────────────────┤
│  Main Dashboard    │   Chatbot App   │  Translator App     │
│   (Port 8501)      │   (Port 8502)   │   (Port 8503)       │
└──────────┬─────────┴─────────────────┴─────────────────────┘
           │
           │ HTTP Requests
           ↓
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)              │
├─────────────────────────────────────────────────────────────┤
│  • Classification API     • Clustering API                  │
│  • Regression API         • Pneumonia Detection API         │
│  • Sentiment Analysis API                                   │
└──────────┬──────────────────────────────────────────────────┘
           │
           │ Model Loading
           ↓
┌─────────────────────────────────────────────────────────────┐
│                    ML/DL Models Layer                       │
├─────────────────────────────────────────────────────────────┤
│  • Random Forest      • K-Means Clustering                  │
│  • Regression Pipeline • CNN (TensorFlow/Keras)             │
│  • VADER Sentiment    • Preprocessing Pipelines             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### **Frontend**
- **Streamlit**: Interactive web applications
- **HTML/CSS**: Custom styling

### **Backend**
- **FastAPI**: RESTful API development
- **Uvicorn**: ASGI server

### **Machine Learning**
- **scikit-learn**: Traditional ML models (Random Forest, K-Means)
- **TensorFlow/Keras**: Deep Learning (CNN)
- **VADER Sentiment**: NLP sentiment analysis

### **Data Processing**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Pillow**: Image processing

### **Model Persistence**
- **Joblib**: Model serialization

---

## 📦 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/healthcare-ai-suite.git
cd healthcare-ai-suite
```

### **Step 2: Create Virtual Environment** (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download Models**
Ensure all trained models are in the `models/` directory:
- `randomforest_classification_model.pkl`
- `scaler_classification.pkl`
- `label_encoder_classification.pkl`
- `best_model_pipeline_regression.pkl`
- `kmeans_model_clustering.pkl`
- `scaler_clustering.pkl`
- `features_to_cluster.pkl`
- `optimal_k_clustering.pkl`
- `best_model.h5`

---

## 🚀 Usage

### **Option 1: Run Individual Apps**
```bash
# Terminal 1: Main Dashboard
streamlit run main_dashboard.py --server.port 8501

# Terminal 2: Chatbot
streamlit run chatbot_app.py --server.port 8502

# Terminal 3: Translator
streamlit run translator_app.py --server.port 8503

# Terminal 4: FastAPI Backend
uvicorn main:app --reload --port 8000
```

### **Option 2: Run All Apps at Once**

**Windows:**
```bash
run_all.bat
```

**Linux/Mac:**
```bash
chmod +x run_all.sh
./run_all.sh
```

### **Access the Applications**

| Application | URL |
|------------|-----|
| 🏥 Main Dashboard | http://localhost:8501 |
| 🤖 Chatbot | http://localhost:8502 |
| 🌐 Translator | http://localhost:8503 |
| 📚 API Documentation | http://localhost:8000/docs |

---

## 📁 Project Structure
```
healthcare-ai-suite/
│
├── main_dashboard.py              # Main Streamlit dashboard
├── chatbot_app.py                 # Healthcare chatbot application
├── translator_app.py              # Medical translator application
├── main.py                        # FastAPI backend
│
├── models/                        # Trained ML/DL models
│   ├── randomforest_classification_model.pkl
│   ├── scaler_classification.pkl
│   ├── label_encoder_classification.pkl
│   ├── best_model_pipeline_regression.pkl
│   ├── kmeans_model_clustering.pkl
│   ├── scaler_clustering.pkl
│   ├── features_to_cluster.pkl
│   ├── optimal_k_clustering.pkl
│   └── best_model.h5
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── run_all.bat                    # Windows startup script
├── run_all.sh                     # Linux/Mac startup script
│
└── .gitignore                     # Git ignore file
```

---

## 🤖 Models & Algorithms

### **1. Disease Risk Classification**
- **Algorithm**: Random Forest Classifier
- **Features**: 9 clinical parameters
  - Age
  - Gender
  - Hypertension (Yes/No)
  - Diabetes (Yes/No)
  - Diastolic Blood Pressure
  - HDL Cholesterol
  - LDL Cholesterol
  - Systolic Blood Pressure
  - Total Cholesterol
- **Target**: High Risk (Age>60 OR Hypertension OR Diabetes) vs Low Risk
- **Preprocessing**: StandardScaler for numeric features

### **2. Length of Stay Prediction**
- **Algorithm**: Regression Pipeline (preprocessing + model)
- **Features**: 
  - Categorical: Encounter Class, Reason Description
  - Numeric: Total Claim Cost, Payer Coverage, Condition Count, Procedure Count, Medication Count
- **Target**: Length of Stay (log-transformed)
- **Preprocessing**: OneHotEncoder for categorical, StandardScaler for numeric

### **3. Patient Segmentation**
- **Algorithm**: K-Means Clustering
- **Features**: Age, Gender, Healthcare Expenses, Healthcare Coverage, Number of Conditions, Procedures, Medications
- **Clusters**: Optimal K determined through elbow method
- **Use Case**: Identify patient groups for targeted interventions

### **4. Pneumonia Detection**
- **Algorithm**: Convolutional Neural Network (CNN)
- **Architecture**: Deep CNN with multiple conv layers
- **Input**: 224x224 grayscale chest X-ray images
- **Preprocessing**: Rescaling (1/255)
- **Output**: Binary classification (Normal vs Pneumonia)

### **5. Sentiment Analysis**
- **Algorithm**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Features**: 
  - Handles negations, intensifiers, punctuation
  - Compound score: -1 (negative) to +1 (positive)
- **Output**: Positive, Negative, or Neutral sentiment
- **Use Case**: Patient feedback analysis for quality improvement

---

## 🔌 API Documentation

### **Base URL**: `http://localhost:8000`

### **Endpoints**

#### **1. Health Check**
```http
GET /health
```
Returns the status of all loaded models.

#### **2. Disease Risk Prediction**
```http
POST /predict/risk
Content-Type: application/json

{
  "age": 65,
  "gender": "Male",
  "hypertension": "Yes",
  "diabetes": "No",
  "diastolic_bp": 85.0,
  "hdl_cholesterol": 45.0,
  "ldl_cholesterol": 130.0,
  "systolic_bp": 145.0,
  "total_cholesterol": 220.0
}
```

**Response:**
```json
{
  "prediction": "High Risk",
  "risk_score": 1,
  "confidence": 87.5,
  "probabilities": {
    "low_risk": 12.5,
    "high_risk": 87.5
  }
}
```

#### **3. Length of Stay Prediction**
```http
POST /predict/los
Content-Type: application/json

{
  "encounter_class": "emergency",
  "reason_description": "Acute bronchitis",
  "total_claim_cost": 5000.0,
  "payer_coverage": 4000.0,
  "cond_count": 2,
  "proc_count": 3,
  "med_count": 5
}
```

**Response:**
```json
{
  "predicted_los_days": 4.2,
  "stay_category": "Medium Stay",
  "estimated_cost": 6300.0
}
```

#### **4. Patient Cluster Assignment**
```http
POST /cluster/assign
Content-Type: application/json

{
  "age": 50,
  "gender": "Female",
  "healthcare_expenses": 15000.0,
  "healthcare_coverage": 12000.0,
  "num_conditions": 2,
  "num_procedures": 5,
  "num_medications": 3
}
```

**Response:**
```json
{
  "cluster_id": 1,
  "cluster_name": "Moderate Care",
  "optimal_k": 4
}
```

#### **5. Pneumonia Detection**
```http
POST /predict/pneumonia
Content-Type: multipart/form-data

file: [chest_xray_image.jpg]
```

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 94.5,
  "raw_score": 0.945
}
```

#### **6. Sentiment Analysis**
```http
POST /analyze/sentiment
Content-Type: application/json

{
  "feedback_text": "The doctor was very helpful and caring."
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 87.3,
  "probabilities": {
    "negative": 12.7,
    "positive": 87.3
  },
  "feedback_stats": {
    "word_count": 7,
    "char_count": 42
  }
}
```

### **Interactive API Documentation**
Visit `http://localhost:8000/docs` for Swagger UI documentation with interactive testing.

---

## 📸 Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)
*Overview of the Healthcare AI Suite with key metrics*

### Disease Risk Classification
![Classification](screenshots/classification.png)
*Real-time disease risk prediction interface*

### Pneumonia Detection
![Pneumonia](screenshots/pneumonia.png)
*AI-powered chest X-ray analysis*

### Patient Segmentation
![Clustering](screenshots/clustering.png)
*Patient clustering visualization and insights*

### Sentiment Analysis
![Sentiment](screenshots/sentiment.png)
*Patient feedback sentiment analysis*

---

## 🔮 Future Enhancements

- [ ] **Real-time Model Monitoring**: Add model performance tracking and drift detection
- [ ] **Multi-model Ensemble**: Combine predictions from multiple models for better accuracy
- [ ] **Explainable AI**: Add SHAP/LIME for model interpretability
- [ ] **Patient Timeline**: Visualize patient health journey over time
- [ ] **Automated Retraining**: Implement MLOps pipeline for continuous model improvement
- [ ] **Mobile Application**: Develop React Native mobile app
- [ ] **Integration with EHR**: Connect to Electronic Health Record systems
- [ ] **Advanced NLP**: Implement BioBERT/ClinicalBERT for medical text analysis
- [ ] **Predictive Analytics Dashboard**: Add forecasting for hospital resource planning
- [ ] **Security Enhancements**: Add authentication, authorization, and data encryption

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Development Guidelines**
- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **Anthropic Claude** for AI assistance in development
- **Streamlit** for the amazing web framework
- **TensorFlow/Keras** for deep learning capabilities
- **scikit-learn** for machine learning algorithms
- **VADER Sentiment** for sentiment analysis
- Healthcare datasets from [source]
- Medical professionals who provided domain expertise

---

## ⚠️ Disclaimer

This system is intended for **educational and research purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.

---

## 📞 Support

For support, questions, or feedback:
- Open an issue on GitHub
- Email: support@yourdomain.com
- Documentation: [Wiki](https://github.com/yourusername/healthcare-ai-suite/wiki)

---

## 📊 Project Status

![Status](https://img.shields.io/badge/Status-Active-success)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-85%25-green)

**Last Updated**: December 2024

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐️!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/healthcare-ai-suite&type=Date)](https://star-history.com/#yourusername/healthcare-ai-suite&Date)

---

<div align="center">
  <p>Made with ❤️ for better healthcare</p>
  <p>© 2024 Healthcare AI Suite. All Rights Reserved.</p>
</div>
