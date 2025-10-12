import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_classification_models():
    try:
        model = joblib.load('MODELS/randomforest_classification_model.pkl')
        scaler = joblib.load('MODELS/scaler_classification.pkl')
        label_encoder = joblib.load('MODELS/label_encoder_classification.pkl')
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading classification models: {e}")
        return None, None, None

@st.cache_resource
def load_regression_model():
    try:
        pipeline = joblib.load('MODELS/best_model_pipeline_regression.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading regression model: {e}")
        return None

@st.cache_resource
def load_clustering_models():
    try:
        kmeans = joblib.load('MODELS/kmeans_model_clustering.pkl')
        scaler = joblib.load('MODELS/scaler_clustering.pkl')
        features = joblib.load('MODELS/features_to_cluster.pkl')
        optimal_k = joblib.load('MODELS/optimal_k_clustering.pkl')
        return kmeans, scaler, features, optimal_k
    except Exception as e:
        st.error(f"Error loading clustering models: {e}")
        return None, None, None, None

@st.cache_resource
def load_cnn_model():
    try:
        model = keras.models.load_model('MODELS/best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

@st.cache_resource
def load_sentiment_analyzer():
    """Load VADER sentiment analyzer"""
    return SentimentIntensityAnalyzer()

# Sidebar Navigation
st.sidebar.title("🏥 Healthcare AI Suite")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to:",
    [
        "🏠 Home",
        "🎯 Disease Risk Classification",
        "📈 Length of Stay Prediction",
        "👥 Patient Segmentation",
        "🫁 Pneumonia Detection",
        "💬 Sentiment Analysis",
        "🤖 Healthcare Chatbot",
        "🌐 Medical Translator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Select a section from above to get started")

# ============= HOME PAGE =============
if page == "🏠 Home":
    st.title("🏥 Healthcare AI/ML Analytics Suite")
    st.markdown("### Welcome to the Comprehensive Healthcare AI System")
    
    st.markdown("""
    This advanced AI-powered platform provides end-to-end healthcare analytics and predictions 
    using state-of-the-art machine learning and deep learning models.
    """)
    
    st.markdown("---")
    
    # Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯</h2>
            <h3>Risk Classification</h3>
            <p>Predict disease risk categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>📈</h2>
            <h3>LOS Prediction</h3>
            <p>Estimate hospital stay duration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>👥</h2>
            <h3>Patient Clustering</h3>
            <p>Discover patient subgroups</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>🫁</h2>
            <h3>Pneumonia Detection</h3>
            <p>AI-powered X-ray analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Section
    st.markdown("### 🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 Machine Learning Models
        - **Disease Risk Classification**: Random Forest classifier for high/low risk prediction
        - **Length of Stay Prediction**: Advanced regression pipeline with preprocessing
        - **Patient Segmentation**: K-Means clustering for patient grouping
        - **Sentiment Analysis**: Patient feedback classification
        """)
    
    with col2:
        st.markdown("""
        #### 🧠 Deep Learning Models
        - **Pneumonia Detection**: CNN model for chest X-ray analysis
        - **Medical Text Analysis**: BioBERT/ClinicalBERT integration
        - **Chatbot**: Healthcare query assistance
        - **Translator**: Multilingual medical communication
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 How to Use
    1. Select a feature from the **sidebar navigation**
    2. Enter the required patient information
    3. Click the **Predict** or **Analyze** button
    4. View detailed results and insights
    
    ### 🔒 Data Privacy
    All predictions are performed locally. No patient data is stored or transmitted.
    """)

# ============= CLASSIFICATION PAGE =============
elif page == "🎯 Disease Risk Classification":
    st.title("🎯 Disease Risk Classification")
    st.markdown("### Predict patient risk based on health parameters")
    
    model, scaler, label_encoder = load_classification_models()
    
    if model is not None and scaler is not None:
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50, help="Patient's age in years")
            gender = st.selectbox("Gender", ["Male", "Female"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        
        with col2:
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=140, value=80, help="mm Hg")
            hdl_cholesterol = st.number_input("HDL Cholesterol", min_value=20, max_value=100, value=50, help="mg/dL")
        
        with col3:
            ldl_cholesterol = st.number_input("LDL Cholesterol", min_value=40, max_value=250, value=100, help="mg/dL")
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120, help="mm Hg")
            total_cholesterol = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=200, help="mg/dL")
        
        st.markdown("---")
        
        if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
            try:
                # Encode gender
                gender_encoded = 1 if gender == "Male" else 0
                hypertension_encoded = 1 if hypertension == "Yes" else 0
                diabetes_encoded = 1 if diabetes == "Yes" else 0
                
                # Prepare numeric features only (for scaling)
                numeric_features = np.array([[
                    age,
                    diastolic_bp, 
                    hdl_cholesterol, 
                    ldl_cholesterol, 
                    systolic_bp, 
                    total_cholesterol
                ]])
                
                # Scale only numeric features
                numeric_scaled = scaler.transform(numeric_features)

                input_scaled = np.concatenate([
                    numeric_scaled[:, 0:1],  # Age (scaled)
                    [[gender_encoded]],       # Gender (not scaled)
                    [[hypertension_encoded]], # Hypertension (not scaled)
                    [[diabetes_encoded]],     # Diabetes (not scaled)
                    numeric_scaled[:, 1:]     # Rest of numeric features (scaled)
                ], axis=1)
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("### ⚠️ HIGH RISK")
                        st.markdown("""
                        **Risk Factors Detected:**
                        - Patient meets high-risk criteria
                        - Immediate medical attention recommended
                        """)
                    else:
                        st.success("### ✅ LOW RISK")
                        st.markdown("""
                        **Assessment:**
                        - Patient parameters within normal range
                        - Continue regular checkups
                        """)
                
                with col2:
                    st.metric("Confidence", f"{probability[prediction] * 100:.2f}%")
                    st.metric("Risk Score", f"{probability[1] * 100:.1f}%")
                
                # Feature importance note
                st.info("""
                **ℹ️ High Risk Definition:**
                Patient is classified as HIGH RISK if:
                - Age > 60 years, OR
                - Hypertension = Yes, OR
                - Diabetes = Yes
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.warning("⚠️ Models not loaded. Please ensure model files are in the 'models/' directory.")

# ============= REGRESSION PAGE =============
elif page == "📈 Length of Stay Prediction":
    st.title("📈 Length of Stay Prediction")
    st.markdown("### Predict hospital stay duration for patients")
    
    pipeline = load_regression_model()
    
    if pipeline is not None:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            encounter_class = st.selectbox(
                "Encounter Class", 
                ["emergency", "urgent care", "ambulatory", "inpatient", "outpatient", "wellness"]
            )
            reason = st.text_input(
                "Reason for Visit", 
                placeholder="e.g., Acute bronchitis",
                help="Description of the medical reason for encounter"
            )
            total_claim = st.number_input(
                "Total Claim Cost ($)", 
                min_value=0.0, 
                max_value=100000.0, 
                value=5000.0,
                step=100.0
            )
        
        with col2:
            payer_coverage = st.number_input(
                "Payer Coverage ($)", 
                min_value=0.0, 
                max_value=100000.0, 
                value=4000.0,
                step=100.0
            )
            cond_count = st.number_input(
                "Number of Conditions", 
                min_value=0, 
                max_value=20, 
                value=2
            )
            proc_count = st.number_input(
                "Number of Procedures", 
                min_value=0, 
                max_value=20, 
                value=3
            )
            med_count = st.number_input(
                "Number of Medications", 
                min_value=0, 
                max_value=30, 
                value=5
            )
        
        st.markdown("---")
        
        if st.button("🔍 Predict Length of Stay", type="primary", use_container_width=True):
            try:
                # Prepare input dataframe
                input_data = pd.DataFrame({
                    'ENCOUNTERCLASS': [encounter_class],
                    'REASONDESCRIPTION': [reason],
                    'TOTAL_CLAIM_COST': [total_claim],
                    'PAYER_COVERAGE': [payer_coverage],
                    'cond_count': [cond_count],
                    'proc_count': [proc_count],
                    'med_count': [med_count]
                })
                
                # Predict (log-transformed)
                los_log_pred = pipeline.predict(input_data)[0]
                
                # Reverse log transform
                los_pred = np.expm1(los_log_pred)
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Length of Stay", f"{los_pred:.1f} days")
                
                with col2:
                    if los_pred < 3:
                        stay_category = "Short Stay"
                        color = "🟢"
                    elif los_pred < 7:
                        stay_category = "Medium Stay"
                        color = "🟡"
                    else:
                        stay_category = "Long Stay"
                        color = "🔴"
                    st.metric("Stay Category", f"{color} {stay_category}")
                
                with col3:
                    estimated_cost = los_pred * 1500  # Approximate cost per day
                    st.metric("Estimated Total Cost", f"${estimated_cost:,.0f}")
                
                # Additional insights
                st.markdown("---")
                st.markdown("### 💡 Insights")
                
                if los_pred > 7:
                    st.warning("""
                    **Long Stay Detected**
                    - Consider discharge planning early
                    - Monitor patient progress closely
                    - Evaluate home care options
                    """)
                elif los_pred < 2:
                    st.info("""
                    **Short Stay Expected**
                    - Patient may be discharged quickly
                    - Prepare outpatient care instructions
                    """)
                else:
                    st.success("""
                    **Normal Stay Duration**
                    - Continue standard care protocols
                    - Monitor for any complications
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.warning("⚠️ Model not loaded. Please ensure model file is in the 'models/' directory.")

# ============= CLUSTERING PAGE =============
elif page == "👥 Patient Segmentation":
    st.title("👥 Patient Segmentation")
    st.markdown("### Discover which patient cluster a patient belongs to")
    
    kmeans, scaler, features, optimal_k = load_clustering_models()
    
    if kmeans is not None and scaler is not None:
        st.markdown("---")
        
        st.info(f"ℹ️ This system uses K-Means clustering with {optimal_k if optimal_k else 'optimal'} clusters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            healthcare_expenses = st.number_input(
                "Healthcare Expenses ($)", 
                min_value=0.0, 
                max_value=100000.0, 
                value=15000.0,
                step=500.0
            )
        
        with col2:
            healthcare_coverage = st.number_input(
                "Healthcare Coverage ($)", 
                min_value=0.0, 
                max_value=100000.0, 
                value=12000.0,
                step=500.0
            )
            num_conditions = st.number_input(
                "Number of Conditions", 
                min_value=0, 
                max_value=20, 
                value=2
            )
        
        with col3:
            num_procedures = st.number_input(
                "Number of Procedures", 
                min_value=0, 
                max_value=30, 
                value=5
            )
            num_medications = st.number_input(
                "Number of Medications", 
                min_value=0, 
                max_value=30, 
                value=3
            )
        
        st.markdown("---")
        
        if st.button("🔍 Find Patient Cluster", type="primary", use_container_width=True):
            try:
                # Encode gender
                gender_encoded = 1 if gender == "Male" else 0
                
                # Prepare input array
                input_data = np.array([[
                    age,
                    gender_encoded,
                    healthcare_expenses,
                    healthcare_coverage,
                    num_conditions,
                    num_procedures,
                    num_medications
                ]])
                
                # Scale the input
                input_scaled = scaler.transform(input_data)
                
                # Predict cluster
                cluster = kmeans.predict(input_scaled)[0]
                
                # Display results
                st.markdown("### 📊 Cluster Assignment")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #1f77b4; color: white; padding: 2rem; border-radius: 0.5rem; text-align: center;'>
                        <h1 style='color: white; margin: 0;'>Cluster {cluster}</h1>
                        <p style='margin: 0.5rem 0 0 0;'>Patient Group</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Cluster characteristics (example - customize based on your data)
                    cluster_info = {
                        0: {
                            "name": "Low-Cost Routine Care",
                            "description": "Patients with minimal healthcare needs and low expenses",
                            "avg_expense": "$5,000 - $10,000",
                            "characteristics": ["Few conditions", "Minimal procedures", "Low medication count"]
                        },
                        1: {
                            "name": "Moderate Care",
                            "description": "Patients with moderate healthcare utilization",
                            "avg_expense": "$10,000 - $25,000",
                            "characteristics": ["Multiple conditions", "Regular procedures", "Moderate medications"]
                        },
                        2: {
                            "name": "High-Intensity Care",
                            "description": "Patients requiring extensive medical attention",
                            "avg_expense": "$25,000 - $50,000",
                            "characteristics": ["Many conditions", "Frequent procedures", "High medication usage"]
                        },
                        3: {
                            "name": "Complex Chronic Care",
                            "description": "Patients with complex chronic conditions",
                            "avg_expense": "$50,000+",
                            "characteristics": ["Chronic diseases", "Intensive procedures", "Multiple medications"]
                        }
                    }
                    
                    # Get cluster info (use default if cluster number exceeds defined clusters)
                    info = cluster_info.get(cluster, {
                        "name": f"Cluster {cluster}",
                        "description": "Patient group with specific characteristics",
                        "avg_expense": "Variable",
                        "characteristics": ["Custom patient profile"]
                    })
                    
                    st.markdown(f"""
                    **{info['name']}**
                    
                    {info['description']}
                    
                    **Average Expense Range:** {info['avg_expense']}
                    
                    **Typical Characteristics:**
                    """)
                    for char in info['characteristics']:
                        st.markdown(f"- {char}")
                
                st.markdown("---")
                
                # Patient profile comparison
                st.markdown("### 📈 Patient Profile")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Healthcare Expenses", f"${healthcare_expenses:,.0f}")
                with col2:
                    st.metric("Coverage", f"${healthcare_coverage:,.0f}")
                with col3:
                    st.metric("Conditions", num_conditions)
                with col4:
                    st.metric("Procedures", num_procedures)
                
                # Recommendations
                st.markdown("---")
                st.markdown("### 💡 Clinical Recommendations")
                
                if healthcare_expenses > 30000:
                    st.warning("""
                    **High-Cost Patient**
                    - Consider care coordination programs
                    - Evaluate preventive care opportunities
                    - Review medication management
                    """)
                elif num_conditions > 5 or num_medications > 10:
                    st.info("""
                    **Complex Care Needs**
                    - Monitor for drug interactions
                    - Consider specialist consultations
                    - Implement care management plan
                    """)
                else:
                    st.success("""
                    **Standard Care Protocol**
                    - Continue routine monitoring
                    - Maintain preventive care schedule
                    - Encourage healthy lifestyle
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.warning("⚠️ Models not loaded. Please ensure model files are in the 'models/' directory.")

# ============= PNEUMONIA DETECTION PAGE =============
elif page == "🫁 Pneumonia Detection":
    st.title("🫁 Pneumonia Detection")
    st.markdown("### AI-powered chest X-ray analysis using Deep Learning")
    
    model = load_cnn_model()
    
    if model is not None:
        st.markdown("---")
        
        st.info("📤 Upload a chest X-ray image (grayscale, 224x224 pixels recommended)")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a chest X-ray image for pneumonia detection"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 📸 Uploaded X-ray")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🔍 Analysis")
                    
                    if st.button("Analyze X-ray", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            # Preprocess image
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
                            prediction = model.predict(img_array, verbose=0)
                            confidence = float(prediction[0][0])
                            
                            # Class: 0 = NORMAL, 1 = PNEUMONIA
                            if confidence > 0.5:
                                result = "PNEUMONIA"
                                result_confidence = confidence * 100
                                color = "error"
                            else:
                                result = "NORMAL"
                                result_confidence = (1 - confidence) * 100
                                color = "success"
                            
                            st.markdown("---")
                            
                            # Display results
                            if result == "PNEUMONIA":
                                st.error(f"### ⚠️ {result} DETECTED")
                            else:
                                st.success(f"### ✅ {result}")
                            
                            st.metric("Confidence", f"{result_confidence:.2f}%")
                            
                            # Progress bar for confidence
                            st.progress(result_confidence / 100)
                            
                            st.markdown("---")
                            
                            # Recommendations
                            st.markdown("### 💡 Recommendations")
                            
                            if result == "PNEUMONIA":
                                st.warning("""
                                **Action Required:**
                                - Immediate medical consultation recommended
                                - Further diagnostic tests may be needed
                                - Consider antibiotic treatment
                                - Monitor respiratory symptoms
                                
                                ⚠️ **Note:** This is an AI prediction. Always consult with a qualified healthcare professional for diagnosis.
                                """)
                            else:
                                st.info("""
                                **Assessment:**
                                - No signs of pneumonia detected
                                - Continue regular health monitoring
                                - Consult doctor if symptoms develop
                                
                                ℹ️ **Note:** This is an AI screening tool. Regular medical checkups are still recommended.
                                """)
            
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.info("Please ensure the image is a valid chest X-ray in JPG, JPEG, or PNG format.")
        
        else:
            st.markdown("""
            ### 📋 Instructions:
            1. Upload a chest X-ray image using the file uploader above
            2. Supported formats: JPG, JPEG, PNG
            3. Click "Analyze X-ray" to get results
            4. Review the prediction and recommendations
            
            ### 🎯 Model Information:
            - **Architecture:** Convolutional Neural Network (CNN)
            - **Input Size:** 224x224 pixels, Grayscale
            - **Classes:** Normal, Pneumonia
            - **Preprocessing:** Automatic rescaling (1/255)
            """)
    
    else:
        st.warning("⚠️ Model not loaded. Please ensure model file is in the 'models/' directory.")

# ============= SENTIMENT ANALYSIS PAGE =============
elif page == "💬 Sentiment Analysis":
    st.title("💬 Patient Feedback Sentiment Analysis")
    st.markdown("### Analyze patient feedback and reviews")
    
    analyzer = load_sentiment_analyzer()
    
    st.markdown("---")
    
    st.markdown("#### 📝 Enter Patient Feedback")
    
    feedback_text = st.text_area(
        "Patient Feedback",
        placeholder="Example: The doctor was very helpful and attentive. The facility was clean and well-maintained.",
        height=150,
        help="Enter patient feedback or review text"
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_button and feedback_text.strip():
        try:
            # Analyze with VADER
            scores = analyzer.polarity_scores(feedback_text)
            compound_score = scores['compound']
            
            # Determine sentiment based on compound score
            # compound score ranges from -1 (most negative) to +1 (most positive)
            if compound_score >= 0.05:
                prediction = 1
                sentiment = "POSITIVE"
                sentiment_emoji = "😊"
                sentiment_color = "#28a745"
                confidence = (compound_score + 1) / 2 * 100  # Convert to 0-100 scale
            elif compound_score <= -0.05:
                prediction = 0
                sentiment = "NEGATIVE"
                sentiment_emoji = "😞"
                sentiment_color = "#dc3545"
                confidence = abs(compound_score + 1) / 2 * 100  # Convert to 0-100 scale
            else:
                prediction = 2
                sentiment = "NEUTRAL"
                sentiment_emoji = "😐"
                sentiment_color = "#ffc107"
                confidence = 50
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Sentiment Analysis Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if prediction == 1:
                    st.success(f"### 😊 {sentiment}")
                elif prediction == 0:
                    st.error(f"### 😞 {sentiment}")
                else:
                    st.info(f"### 😐 {sentiment}")
                
                st.markdown(f"""
                <div style='background-color: {sentiment_color}; color: white; padding: 2rem; border-radius: 0.5rem; text-align: center;'>
                    <h1 style='color: white; margin: 0; font-size: 4rem;'>{sentiment_emoji}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Overall Confidence", f"{confidence:.2f}%")
                st.metric("Compound Score", f"{compound_score:.3f}")
                
                # Sentiment score bar
                st.markdown("**Sentiment Distribution:**")
                # Normalize compound score to 0-1 range for progress bar
                progress_value = (compound_score + 1) / 2
                st.progress(progress_value)
                
                st.markdown("---")
                st.markdown("**Detailed Scores:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("😊 Positive", f"{scores['pos']:.2f}")
                with col_b:
                    st.metric("😐 Neutral", f"{scores['neu']:.2f}")
                with col_c:
                    st.metric("😞 Negative", f"{scores['neg']:.2f}")
            
            st.markdown("---")
            
            # Insights
            st.markdown("### 💡 Insights & Actions")
            
            if prediction == 1:
                if compound_score > 0.5:
                    st.success("""
                    **Highly Positive Feedback**
                    - Patient is very satisfied with the service
                    - Consider this as a success case study
                    - Share positive experience with team
                    - Request testimonial if appropriate
                    """)
                else:
                    st.info("""
                    **Positive Feedback**
                    - Patient is generally satisfied
                    - Continue maintaining service quality
                    - Look for improvement opportunities
                    """)
            elif prediction == 0:
                if compound_score < -0.5:
                    st.error("""
                    **Highly Negative Feedback**
                    - Immediate attention required
                    - Contact patient for follow-up
                    - Investigate issues mentioned
                    - Implement corrective actions
                    """)
                else:
                    st.warning("""
                    **Negative Feedback**
                    - Address concerns promptly
                    - Review service quality
                    - Consider patient satisfaction survey
                    """)
            else:
                st.info("""
                **Neutral Feedback**
                - Mixed or balanced sentiment detected
                - Review specific concerns mentioned
                - Follow up if clarification needed
                """)
            
            # Word count and feedback stats
            st.markdown("---")
            st.markdown("### 📈 Feedback Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                word_count = len(feedback_text.split())
                st.metric("Word Count", word_count)
            
            with col2:
                char_count = len(feedback_text)
                st.metric("Character Count", char_count)
            
            with col3:
                sentence_count = feedback_text.count('.') + feedback_text.count('!') + feedback_text.count('?')
                st.metric("Sentences", max(1, sentence_count))
            
        except Exception as e:
            st.error(f"Error analyzing sentiment: {e}")
    
    elif analyze_button and not feedback_text.strip():
        st.warning("⚠️ Please enter some feedback text to analyze.")
    
    # Examples section
    st.markdown("---")
    st.markdown("### 📝 Example Feedback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Positive Example:**
        ```
        The staff was incredibly caring and 
        professional. The doctor took time to 
        explain everything clearly. I felt 
        well taken care of throughout my visit. 
        Highly recommend this facility!
        ```
        """)
    
    with col2:
        st.markdown("""
        **Negative Example:**
        ```
        Long wait times and poor communication. 
        The staff seemed disorganized. I waited 
        over 2 hours for my appointment. Not 
        satisfied with the service quality.
        ```
        """)
    
    st.markdown("---")
    st.info("""
    **ℹ️ About VADER Sentiment Analysis:**
    
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically designed for social media text and works well for short feedback. 
    It considers:
    - Sentiment polarity and intensity
    - Punctuation (e.g., "good" vs "good!!!")
    - Capitalization (e.g., "GREAT" vs "great")
    - Negations (e.g., "not good")
    - Intensifiers (e.g., "very bad" vs "bad")
    
    **Compound Score Range:**
    - Positive: ≥ 0.05
    - Neutral: -0.05 to 0.05
    - Negative: ≤ -0.05
    """)

# ============= CHATBOT PAGE =============
elif page == "🤖 Healthcare Chatbot":
    st.title("🤖 Healthcare Chatbot")
    st.markdown("### AI-powered medical query assistance")
    
    st.markdown("---")
    
    st.info("""
    **🚀 The Healthcare Chatbot is available as a separate application.**
    
    The chatbot provides:
    - 💬 Medical information and guidance
    - 🏥 Healthcare recommendations
    - 💊 Medication information
    - 📋 Symptom assessment
    - 🔍 Health-related queries
    """)
    
    st.markdown("---")
    
    # Placeholder for chatbot link
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem; 
                border-radius: 1rem; 
                text-align: center; 
                color: white;'>
        <h2 style='color: white; margin-bottom: 1rem;'>🤖 Launch Healthcare Chatbot</h2>
        <p style='font-size: 1.1rem; margin-bottom: 2rem;'>
            Click the button below to open the chatbot in a new window
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <p><strong>Chatbot URL will be available here</strong></p>
            <p style='color: #666;'>Example: http://localhost:8502</p>
        </div>
        """, unsafe_allow_html=True)
        
        # You can add the actual link once you have the port
        st.markdown("[🚀 Open Chatbot](http://localhost:8502)", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 Chatbot Features:
    
    #### 🩺 Medical Information
    - Disease information and symptoms
    - Treatment options and recommendations
    - Preventive care guidance
    
    #### 💊 Medication Assistance
    - Drug information and interactions
    - Dosage guidelines
    - Side effects information
    
    #### 🏥 Healthcare Navigation
    - Finding specialists
    - Understanding medical procedures
    - Healthcare system guidance
    
    #### ⚠️ Disclaimer
    This chatbot is for informational purposes only and should not replace 
    professional medical advice, diagnosis, or treatment.
    """)

# ============= TRANSLATOR PAGE =============
elif page == "🌐 Medical Translator":
    st.title("🌐 Medical Translator")
    st.markdown("### Multilingual medical communication tool")
    
    st.markdown("---")
    
    st.info("""
    **🚀 The Medical Translator is available as a separate application.**
    
    The translator provides:
    - 🌍 Multiple language support
    - 🏥 Medical terminology translation
    - 📄 Document translation
    - 💬 Real-time conversation translation
    - 🔊 Text-to-speech capabilities
    """)
    
    st.markdown("---")
    
    # Placeholder for translator link
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 3rem; 
                border-radius: 1rem; 
                text-align: center; 
                color: white;'>
        <h2 style='color: white; margin-bottom: 1rem;'>🌐 Launch Medical Translator</h2>
        <p style='font-size: 1.1rem; margin-bottom: 2rem;'>
            Click the button below to open the translator in a new window
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <p><strong>Translator URL will be available here</strong></p>
            <p style='color: #666;'>Example: http://localhost:8503</p>
        </div>
        """, unsafe_allow_html=True)
        
        # You can add the actual link once you have the port
        st.markdown("[🚀 Open Translator](http://localhost:8503)", unsafe_allow_html=True)

        
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 Translator Features:
    
    #### 🌍 Language Support
    - English, Spanish, Hindi, French, German, Chinese, and more
    - Medical terminology in multiple languages
    - Cultural context consideration
    
    #### 🏥 Medical Translation
    - Patient instructions
    - Medical reports and documents
    - Prescription information
    - Diagnostic results
    
    #### 💬 Communication Tools
    - Real-time conversation translation
    - Voice input and output
    - Common medical phrases library
    
    #### 📄 Document Translation
    - Medical records
    - Lab reports
    - Treatment plans
    - Consent forms
    
    #### ⚠️ Note
    Translations are AI-generated and should be verified by qualified 
    medical interpreters for critical communications.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Healthcare AI/ML Analytics Suite</strong></p>
    <p>Powered by Advanced Machine Learning and Deep Learning Models</p>
    <p style='font-size: 0.9rem;'>⚠️ For informational purposes only. Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)