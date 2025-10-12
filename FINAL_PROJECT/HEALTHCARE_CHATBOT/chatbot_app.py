# Healthcare Chatbot - Streamlit Application
# Complete Interactive Healthcare Assistant

import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Healthcare Assistant Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
        color: #000000; 
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 5px solid #4CAF50;
        color: #000000;
    }
    .symptom-badge {
        background-color: #ff9800;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
        margin: 5px;
        font-size: 0.9em;
    }
    .disease-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .urgency-high {
        color: #f44336;
        font-weight: bold;
        font-size: 1.2em;
    }
    .urgency-moderate {
        color: #ff9800;
        font-weight: bold;
        font-size: 1.2em;
    }
    .urgency-low {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2em;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        margin: 5px 0;
    }
    .confidence-fill {
        background-color: #4CAF50;
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 0.8em;
        line-height: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================

@st.cache_resource
def load_models_and_data():
    """Load all models and preprocessed data"""
    
    # Load disease information
    with open('disease_info_dict.pkl', 'rb') as f:
        disease_info_dict = pickle.load(f)
    
    # Load symptom severity
    with open('symptom_severity_dict.pkl', 'rb') as f:
        symptom_severity_dict = pickle.load(f)
    
    # Load symptom to diseases mapping
    with open('symptom_to_diseases.pkl', 'rb') as f:
        symptom_to_diseases = pickle.load(f)
    
    # Load FAQ database
    with open('faq_database.pkl', 'rb') as f:
        faq_df = pickle.load(f)
    
    # Load FAQ embeddings
    faq_embeddings = np.load('faq_embeddings.npy')
    
    # Load sentence transformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    return disease_info_dict, symptom_severity_dict, symptom_to_diseases, faq_df, faq_embeddings, embedder

# Load all data
with st.spinner("🔄 Loading healthcare assistant..."):
    disease_info_dict, symptom_severity_dict, symptom_to_diseases, faq_df, faq_embeddings, embedder = load_models_and_data()

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def extract_symptoms(user_input, known_symptoms):
    """Extract symptoms from user input text"""
    user_input = user_input.lower().strip()
    detected_symptoms = []
    
    for symptom in known_symptoms:
        symptom_words = symptom.split()
        if all(word in user_input for word in symptom_words):
            detected_symptoms.append(symptom)
        elif len(symptom_words) == 1 and symptom in user_input:
            detected_symptoms.append(symptom)
    
    return detected_symptoms

def match_diseases(input_symptoms, disease_symptoms_dict, top_k=5):
    """Match user symptoms to diseases"""
    if not input_symptoms:
        return []
    
    disease_scores = []
    
    for disease, disease_symptoms in disease_symptoms_dict.items():
        matched_symptoms = set(input_symptoms) & set(disease_symptoms)
        
        if matched_symptoms:
            precision = len(matched_symptoms) / len(disease_symptoms)
            recall = len(matched_symptoms) / len(input_symptoms)
            
            if precision + recall > 0:
                confidence = 2 * (precision * recall) / (precision + recall)
            else:
                confidence = 0
            
            disease_scores.append({
                'disease': disease,
                'confidence': confidence,
                'matched_symptoms': list(matched_symptoms),
                'match_count': len(matched_symptoms)
            })
    
    disease_scores.sort(key=lambda x: (x['confidence'], x['match_count']), reverse=True)
    return disease_scores[:top_k]

def assess_urgency(symptoms, symptom_severity_dict):
    """Assess urgency level based on symptom severity"""
    if not symptoms:
        return "UNKNOWN", 0, "No symptoms provided"
    
    total_score = 0
    symptom_count = 0
    
    for symptom in symptoms:
        if symptom in symptom_severity_dict:
            total_score += symptom_severity_dict[symptom]
            symptom_count += 1
    
    if symptom_count > 0:
        avg_severity = total_score / symptom_count
    else:
        avg_severity = 0
    
    if avg_severity >= 6 or total_score >= 20:
        urgency = "HIGH"
        color = "🔴"
        recommendation = "Seek immediate medical attention or visit ER"
    elif avg_severity >= 4 or total_score >= 12:
        urgency = "MODERATE"
        color = "🟡"
        recommendation = "Consult a doctor within 24 hours"
    else:
        urgency = "LOW"
        color = "🟢"
        recommendation = "Monitor symptoms and rest. Consult doctor if symptoms worsen"
    
    explanation = f"Average severity: {avg_severity:.1f} | Total score: {total_score}"
    
    return urgency, avg_severity, f"{color} {recommendation}"

def perform_triage(user_input, disease_info_dict, symptom_severity_dict, symptom_to_diseases):
    """Main function to perform complete symptom triage"""
    all_symptoms = list(symptom_to_diseases.keys())
    detected_symptoms = extract_symptoms(user_input, all_symptoms)
    
    if not detected_symptoms:
        return {
            'status': 'error',
            'message': 'No recognizable symptoms found. Please describe your symptoms clearly (e.g., "I have fever, headache, and cough").',
            'detected_symptoms': []
        }
    
    disease_symptoms_only = {d: info['symptoms'] for d, info in disease_info_dict.items()}
    matched_diseases = match_diseases(detected_symptoms, disease_symptoms_only, top_k=5)
    
    if not matched_diseases:
        return {
            'status': 'error',
            'message': 'Could not match symptoms to any known conditions.',
            'detected_symptoms': detected_symptoms
        }
    
    urgency, severity_score, recommendation = assess_urgency(detected_symptoms, symptom_severity_dict)
    
    results = {
        'status': 'success',
        'detected_symptoms': detected_symptoms,
        'urgency': urgency,
        'severity_score': severity_score,
        'recommendation': recommendation,
        'possible_conditions': []
    }
    
    for disease_match in matched_diseases:
        disease_name = disease_match['disease']
        disease_details = {
            'name': disease_name,
            'confidence': disease_match['confidence'],
            'matched_symptoms': disease_match['matched_symptoms'],
            'description': disease_info_dict[disease_name]['description'],
            'precautions': disease_info_dict[disease_name]['precautions']
        }
        results['possible_conditions'].append(disease_details)
    
    return results

def search_faq(user_question, faq_df, faq_embeddings, embedder, top_k=3):
    """Search FAQ database using semantic similarity"""
    question_embedding = embedder.encode([user_question])[0]
    
    similarities = cosine_similarity(
        question_embedding.reshape(1, -1),
        faq_embeddings
    )[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'question': faq_df.iloc[idx]['question'],
            'answer': faq_df.iloc[idx]['answer'],
            'category': faq_df.iloc[idx]['category'],
            'similarity': similarities[idx]
        })
    
    return results

def detect_intent(user_input):
    """Detect if user is describing symptoms or asking a general question"""
    symptom_keywords = ['have', 'feel', 'experiencing', 'suffering', 'pain', 'ache', 'hurts', 
                       'fever', 'cough', 'headache', 'nausea', 'vomiting', 'diarrhea', 
                       'itching', 'rash', 'swelling', 'bleeding', 'dizzy', 'tired']
    
    faq_keywords = ['what', 'how', 'when', 'why', 'should', 'can', 'is', 'are', 
                   'tell me', 'explain', 'difference', 'mean']
    
    user_lower = user_input.lower()
    
    symptom_score = sum(1 for keyword in symptom_keywords if keyword in user_lower)
    faq_score = sum(1 for keyword in faq_keywords if keyword in user_lower)
    
    # If user mentions specific symptoms, prioritize triage
    if symptom_score > faq_score:
        return 'symptom_triage'
    else:
        return 'faq'

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        'role': 'assistant',
        'content': 'Hello! 👋 I\'m your Healthcare Assistant. I can help you with:\n\n🔹 **Symptom Assessment** - Describe your symptoms and I\'ll help assess them\n🔹 **Health Questions** - Ask me any general health questions\n\nHow can I assist you today?',
        'timestamp': datetime.now()
    })

if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🏥 Healthcare Assistant")
    st.markdown("---")
    
    st.subheader("📊 System Information")
    st.info(f"✓ {len(disease_info_dict)} diseases in database")
    st.info(f"✓ {len(symptom_to_diseases)} symptoms tracked")
    st.info(f"✓ {len(faq_df)} FAQs available")
    
    st.markdown("---")
    
    st.subheader("💡 Tips for Best Results")
    st.markdown("""
    **For Symptom Assessment:**
    - Be specific about your symptoms
    - Mention duration and severity
    - Example: "I have high fever, headache, and vomiting"
    
    **For General Questions:**
    - Ask clear, specific questions
    - Example: "How much water should I drink?"
    """)
    
    st.markdown("---")
    
    st.subheader("🔄 Actions")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.session_state.messages.append({
            'role': 'assistant',
            'content': 'Chat history cleared! How can I help you?',
            'timestamp': datetime.now()
        })
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("⚠️ Disclaimer")
    st.warning("""
    This chatbot is for informational purposes only and does not replace professional medical advice. 
    Always consult a healthcare provider for medical concerns.
    """)
    
    st.markdown("---")
    st.caption("Healthcare Assistant v1.0")
    st.caption(f"Conversations: {st.session_state.conversation_count}")

# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

st.title("🏥 Healthcare Assistant Chatbot")
st.markdown("Your AI-powered health companion for symptom assessment and health questions")

# Display chat messages
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>👤 You</b><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <b>🤖 Healthcare Assistant</b><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now()
    })
    
    st.session_state.conversation_count += 1
    
    # Detect intent
    intent = detect_intent(user_input)
    
    # Process based on intent
    if intent == 'symptom_triage':
        # Perform symptom triage
        with st.spinner("🔍 Analyzing your symptoms..."):
            triage_result = perform_triage(user_input, disease_info_dict, 
                                          symptom_severity_dict, symptom_to_diseases)
        
        if triage_result['status'] == 'error':
            response = f"❌ {triage_result['message']}"
        else:
            # Format response
            response = "## 🔍 Symptom Analysis Results\n\n"
            
            # Detected symptoms
            response += "### Detected Symptoms:\n"
            for symptom in triage_result['detected_symptoms']:
                response += f'<span class="symptom-badge">{symptom}</span>'
            response += "\n\n"
            
            # Urgency assessment
            urgency_class = f"urgency-{triage_result['urgency'].lower()}"
            response += f"### Urgency Level:\n"
            response += f'<p class="{urgency_class}">{triage_result["recommendation"]}</p>\n\n'
            
            # Possible conditions
            response += "### Possible Conditions:\n\n"
            for i, condition in enumerate(triage_result['possible_conditions'][:3], 1):
                confidence_pct = condition['confidence'] * 100
                response += f'<div class="disease-card">'
                response += f"**{i}. {condition['name']}**\n\n"
                response += f'<div class="confidence-bar"><div class="confidence-fill" style="width: {confidence_pct}%">{confidence_pct:.1f}%</div></div>\n\n'
                response += f"**Matched Symptoms:** {', '.join(condition['matched_symptoms'])}\n\n"
                
                if condition['description']:
                    response += f"**About:** {condition['description'][:200]}...\n\n"
                
                if condition['precautions']:
                    response += "**Recommended Precautions:**\n"
                    for prec in condition['precautions'][:3]:
                        response += f"- {prec}\n"
                
                response += '</div>\n\n'
            
            response += "\n\n---\n\n"
            response += "💡 **Remember:** This is a preliminary assessment. Please consult a healthcare professional for proper diagnosis and treatment."
    
    else:
        # Search FAQ
        with st.spinner("🔍 Searching for answers..."):
            faq_results = search_faq(user_input, faq_df, faq_embeddings, embedder, top_k=3)
        
        best_match = faq_results[0]
        
        if best_match['similarity'] > 0.6:
            # High confidence match
            response = f"## 💡 {best_match['question']}\n\n"
            response += f"{best_match['answer']}\n\n"
            response += f"*Category: {best_match['category']}*\n\n"
            
            # Show related questions if similarity is not too high
            if best_match['similarity'] < 0.9 and len(faq_results) > 1:
                response += "\n### Related Questions:\n"
                for result in faq_results[1:3]:
                    if result['similarity'] > 0.4:
                        response += f"- {result['question']}\n"
        else:
            # Lower confidence - show multiple options
            response = "I found some related information:\n\n"
            for i, result in enumerate(faq_results, 1):
                response += f"**{i}. {result['question']}**\n"
                response += f"{result['answer'][:150]}...\n\n"
            
            response += "\nWould you like more details on any of these topics?"
    
    # Add bot response to chat
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
        'timestamp': datetime.now()
    })
    
    # Rerun to display new messages
    st.rerun()

# =============================================================================
# QUICK ACTION BUTTONS
# =============================================================================

st.markdown("---")
st.subheader("🚀 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🤒 Common Cold Symptoms"):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'I have runny nose, continuous sneezing, and mild fever',
            'timestamp': datetime.now()
        })
        st.rerun()

with col2:
    if st.button("💊 Medication Questions"):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'Can I take medicine with an empty stomach?',
            'timestamp': datetime.now()
        })
        st.rerun()

with col3:
    if st.button("🏃 Exercise Advice"):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'How often should I exercise?',
            'timestamp': datetime.now()
        })
        st.rerun()

with col4:
    if st.button("🚨 Emergency Signs"):
        st.session_state.messages.append({
            'role': 'user',
            'content': 'When should I go to the emergency room?',
            'timestamp': datetime.now()
        })
        st.rerun()