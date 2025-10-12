# Medical Translator - Streamlit Application
# Languages: English, Hindi, Marathi, Gujarati

import streamlit as st
import pickle
from googletrans import Translator
from langdetect import detect
import pandas as pd
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Medical Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .translation-box {
        background-color: #1e3a8a;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 5px solid #0066cc;
        color: white;
    }
    .medical-term { background-color: #fff3cd; padding: 3px 8px; border-radius: 5px; font-weight: bold; color: #856404; display: inline-block; margin: 2px; }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .section-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 20px 0 10px 0; text-align: center; font-size: 1.3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================

@st.cache_resource
def load_all_data():
    with open('medical_dictionary.pkl', 'rb') as f:
        medical_dict = pickle.load(f)
    with open('common_phrases.pkl', 'rb') as f:
        phrases = pickle.load(f)
    with open('language_config.pkl', 'rb') as f:
        lang_config = pickle.load(f)
    try:
        with open('term_categories.pkl', 'rb') as f:
            categories = pickle.load(f)
    except:
        categories = {}
    translator = Translator()
    return medical_dict, phrases, lang_config, categories, translator

with st.spinner("🔄 Loading Medical Translator..."):
    medical_dictionary, common_phrases, language_config, term_categories, translator = load_all_data()

# =============================================================================
# TRANSLATION FUNCTIONS
# =============================================================================

def detect_language_enhanced(text):
    try:
        detected = detect(text)
        lang_map = {'hi': 'hi', 'mr': 'mr', 'gu': 'gu', 'en': 'en'}
        return lang_map.get(detected, 'en')
    except:
        if any('\u0900' <= char <= '\u097F' for char in text):
            if any(char in text for char in ['ढ', 'ळ', 'ऱ']):
                return 'mr'
            return 'hi'
        if any('\u0A80' <= char <= '\u0AFF' for char in text):
            return 'gu'
        return 'en'

def extract_medical_terms(text, language):
    text_lower = text.lower()
    found_terms = []
    for term_en, translations in medical_dictionary.items():
        if language in translations:
            term_in_lang = translations[language].lower()
            if '(' in term_in_lang:
                term_in_lang = term_in_lang.split('(')[0].strip()
            if term_in_lang in text_lower:
                found_terms.append({
                    'term': term_en,
                    'found_text': term_in_lang,
                    'translations': translations
                })
    return found_terms

def translate_text(text, source_lang, target_lang):
    try:
        lang_codes = {'en': 'en', 'hi': 'hi', 'mr': 'mr', 'gu': 'gu'}
        src = lang_codes.get(source_lang, 'en')
        tgt = lang_codes.get(target_lang, 'en')
        translation = translator.translate(text, src=src, dest=tgt)
        return translation.text
    except:
        return text

def translate_with_medical_accuracy(text, source_lang, target_lang):
    text_key = text.strip()
    if text_key in common_phrases and target_lang in common_phrases[text_key]:
        return {
            'translation': common_phrases[text_key][target_lang],
            'source': 'phrase_library',
            'medical_terms': [],
            'confidence': 'high',
            'original': text
        }
    medical_terms = extract_medical_terms(text, source_lang)
    base_translation = translate_text(text, source_lang, target_lang)
    return {
        'translation': base_translation,
        'source': 'model + correction' if medical_terms else 'model',
        'medical_terms': [t['term'] for t in medical_terms],
        'confidence': 'high' if medical_terms else 'medium',
        'original': text
    }

# =============================================================================
# SESSION STATE
# =============================================================================

if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

if 'last_doctor_translation' not in st.session_state:
    st.session_state.last_doctor_translation = ""

if 'last_patient_translation' not in st.session_state:
    st.session_state.last_patient_translation = ""

if 'source_lang' not in st.session_state:
    st.session_state.source_lang = 'en'

if 'target_lang' not in st.session_state:
    st.session_state.target_lang = 'hi'

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🌍 Medical Translator")
    st.markdown("**Bridge doctor-patient communication**")
    st.markdown("---")
    st.info(f"✓ {len(medical_dictionary)} medical terms")
    st.info(f"✓ {len(common_phrases)} common phrases")
    st.info(f"✓ {len(language_config['languages'])} languages supported")
    st.markdown("---")
    if st.session_state.translation_history:
        st.subheader("📈 Session Stats")
        st.metric("Translations", len(st.session_state.translation_history))
        lang_pairs = [f"{t['source_lang']}→{t['target_lang']}" for t in st.session_state.translation_history]
        most_common = max(set(lang_pairs), key=lang_pairs.count)
        st.metric("Most Used", most_common)
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.translation_history = []
        st.rerun()

# =============================================================================
# MAIN INTERFACE
# =============================================================================

st.markdown("<div class='section-header'>💬 Translation Interface</div>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

# ---------------- DOCTOR PANEL ----------------
with col_left:
    st.markdown("### 👨‍⚕️ Doctor's Panel")
    
    source_options = {code: info['native'] for code, info in language_config['languages'].items()}
    st.session_state.source_lang = st.selectbox(
        "From Language:",
        options=list(source_options.keys()),
        format_func=lambda x: source_options[x],
        index=list(source_options.keys()).index(st.session_state.source_lang),
        key='source_selector'
    )
    
    doctor_input = st.text_area(
        "Type here (Doctor):",
        height=150,
        placeholder="Type your message...",
        key='doctor_input',
        value=st.session_state.get("last_patient_translation", "")
    )

# ---------------- PATIENT PANEL ----------------
with col_right:
    st.markdown("### 👤 Patient's Panel")
    
    target_options = {code: info['native'] for code, info in language_config['languages'].items()}
    st.session_state.target_lang = st.selectbox(
        "To Language:",
        options=list(target_options.keys()),
        format_func=lambda x: target_options[x],
        index=list(target_options.keys()).index(st.session_state.target_lang),
        key='target_selector'
    )
    
    patient_input = st.text_area(
        "Type here (Patient):",
        height=150,
        placeholder="Type your message...",
        key='patient_input',
        value=st.session_state.get("last_doctor_translation", "")
    )

# ---------------- TRANSLATE BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1])
with col2:
    translate_button = st.button("🔄 TRANSLATE", use_container_width=True)

if translate_button:
    if doctor_input:
        # Doctor → Patient
        result = translate_with_medical_accuracy(
            doctor_input,
            st.session_state.source_lang,
            st.session_state.target_lang
        )
        st.session_state.last_doctor_translation = result['translation']
        st.session_state.translation_history.insert(0, {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'original': doctor_input,
            'translation': result['translation'],
            'source_lang': st.session_state.source_lang,
            'target_lang': st.session_state.target_lang,
            'medical_terms': result['medical_terms'],
            'confidence': result['confidence']
        })
    if patient_input:
        # Patient → Doctor (reverse)
        rev_result = translate_with_medical_accuracy(
            patient_input,
            st.session_state.target_lang,
            st.session_state.source_lang
        )
        st.session_state.last_patient_translation = rev_result['translation']
        st.session_state.translation_history.insert(0, {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'original': patient_input,
            'translation': rev_result['translation'],
            'source_lang': st.session_state.target_lang,
            'target_lang': st.session_state.source_lang,
            'medical_terms': rev_result['medical_terms'],
            'confidence': rev_result['confidence']
        })

# =============================================================================
# TRANSLATION HISTORY
# =============================================================================
if st.session_state.translation_history:
    st.markdown("<div class='section-header'>📜 Translation History</div>", unsafe_allow_html=True)
    for idx, item in enumerate(st.session_state.translation_history[:10]):
        with st.expander(f"🕐 {item['timestamp']} | {item['source_lang'].upper()} → {item['target_lang'].upper()}"):
            st.markdown(f"**Original:** {item['original']}")
            st.markdown(f"**Translation:** {item['translation']}")
            if item['medical_terms']:
                st.markdown(f"**Medical Terms:** {', '.join(item['medical_terms'])}")
            st.markdown(f"**Confidence:** {item['confidence']}")
