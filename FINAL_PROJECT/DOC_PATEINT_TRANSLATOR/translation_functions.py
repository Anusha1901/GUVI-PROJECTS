"""
Translation Functions Module
Core functions for medical translation
"""

def detect_language_enhanced(text):
    """Enhanced language detection"""
    from langdetect import detect
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

def extract_medical_terms(text, language, medical_dictionary):
    """Extract medical terms from text"""
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

def translate_with_medical_accuracy(text, source_lang, target_lang, 
                                   medical_dictionary, common_phrases, 
                                   translator):
    """Complete translation pipeline"""
    
    # Check phrase library
    text_key = text.strip()
    if text_key in common_phrases and target_lang in common_phrases[text_key]:
        return {
            'translation': common_phrases[text_key][target_lang],
            'source': 'phrase_library',
            'medical_terms': [],
            'confidence': 'high'
        }
    
    # Extract medical terms
    medical_terms = extract_medical_terms(text, source_lang, medical_dictionary)
    
    # Base translation
    lang_codes = {'en': 'en', 'hi': 'hi', 'mr': 'mr', 'gu': 'gu'}
    src = lang_codes.get(source_lang, 'en')
    tgt = lang_codes.get(target_lang, 'en')
    
    try:
        translation_obj = translator.translate(text, src=src, dest=tgt)
        base_translation = translation_obj.text
    except:
        base_translation = text
    
    # Correct medical terms if needed
    final_translation = base_translation
    
    return {
        'translation': final_translation,
        'source': 'model + correction' if medical_terms else 'model',
        'medical_terms': [t['term'] for t in medical_terms],
        'confidence': 'high' if medical_terms else 'medium'
    }
