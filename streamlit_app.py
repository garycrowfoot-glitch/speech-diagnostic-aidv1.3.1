import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import difflib
from fpdf import FPDF
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Speech Diagnostic Support Tool",
    page_icon="🎙️",
    layout="wide"
)

# Load configuration and reference data
@st.cache_data
def load_reference_phrases():
    """Load comprehensive reference phrases from CSV"""
    try:
        df = pd.read_csv('reference_phrases_diagnostic.csv')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.warning("⚠️ reference_phrases_diagnostic.csv not found. Using basic phrases.")
        data = {
            'phrase': ['The cat sat on the mat', 'She sells seashells by the seashore'],
            'expected_IPA': ['/ðə kæt sæt ɒn ðə mæt/', '/ʃi sɛlz siːʃɛlz baɪ ðə siːʃɔː/'],
            'phoneme_breakdown': ['ð ə | k æ t | s æ t | ɒ n | ð ə | m æ t', 'ʃ i | s ɛ l z | s iː ʃ ɛ l z | b aɪ | ð ə | s iː ʃ ɔː'],
            'example_distortion_patterns': ['/θ/→/f/, /t/→glottal stop', '/ʃ/→/s/ (lisp), /z/→/d/']
        }
        return pd.DataFrame(data)

@st.cache_data
def load_sensitivity_config():
    """Load sensitivity thresholds"""
    try:
        with open('sensitivity_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "high_confidence": 0.85,
            "moderate_confidence": 0.65,
            "low_confidence": 0.45
        }

@st.cache_data
def load_speech_rules():
    """Load comprehensive speech disorder rules"""
    try:
        df = pd.read_csv('speech_rule_mapping.csv')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.warning("⚠️ speech_rule_mapping.csv not found. Using basic rule set.")
        data = {
            'Condition': ['Articulation Disorder', 'Phonological Disorder'],
            'Typical phonetic/phonological patterns': ['Difficulty producing specific speech sounds', 'Patterned simplifications of sounds'],
            'Rule mapping (example)': ['r→w substitution', 'cluster reduction'],
            'Clinical notes': ['Motor-based errors', 'Language-based phonological knowledge difference'],
            'Age_of_concern': ['Any age', 'Typically noticed in childhood'],
            'Confidence_notes': ['High for detection', 'High when multiple processes observed']
        }
        return pd.DataFrame(data)

def transcribe_audio_to_text(audio_file):
    """
    Transcribe audio file to text using OpenAI Whisper API
    Returns: (transcribed_text, success_flag, error_message)
    """
    try:
        import openai
        
        # Check if API key is set
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return None, False, "OpenAI API key not configured. Please add OPENAI_API_KEY to Streamlit secrets."
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Transcribe using Whisper
            with open(tmp_path, 'rb') as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="en"
                )
            
            return transcript.text, True, None
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except ImportError:
        return None, False, "OpenAI library not installed. Please add 'openai' to requirements.txt"
    except Exception as e:
        return None, False, f"Transcription error: {str(e)}"

def text_to_ipa(text):
    """
    Convert English text to IPA using epitran
    Returns: (ipa_string, success_flag, error_message)
    """
    try:
        import epitran
        
        # Initialize epitran for English (Australian or General American)
        # Australian English would be 'eng-Latn-aus' if available, fallback to general
        epi = epitran.Epitran('eng-Latn')
        
        # Convert to IPA
        ipa = epi.transliterate(text)
        
        # Wrap in forward slashes
        return f"/{ipa}/", True, None
        
    except ImportError:
        return None, False, "Epitran library not installed. Please add 'epitran' to requirements.txt"
    except Exception as e:
        return None, False, f"IPA conversion error: {str(e)}"

def audio_to_ipa_pipeline(audio_file, reference_phrase):
    """
    Complete pipeline: Audio → Text → IPA
    Returns: (ipa_string, transcribed_text, success_flag, error_message)
    """
    # Step 1: Transcribe audio to text
    transcribed_text, success, error = transcribe_audio_to_text(audio_file)
    
    if not success:
        return None, None, False, error
    
    # Step 2: Convert text to IPA
    ipa_string, success, error = text_to_ipa(transcribed_text)
    
    if not success:
        return None, transcribed_text, False, error
    
    return ipa_string, transcribed_text, True, None

def clean_ipa_for_pdf(text):
    """Convert IPA characters to ASCII-safe alternatives for PDF generation"""
    if not text:
        return ""
    
    text = str(text).strip('/')
    
    ipa_map = {
        'ð': 'dh', 'θ': 'th', 'ʃ': 'sh', 'ʒ': 'zh',
        'ŋ': 'ng', 'ɹ': 'r', 'ɔː': 'aw', 'ɔ': 'o',
        'æ': 'a', 'ɛ': 'e', 'ɪ': 'i', 'ʌ': 'u',
        'ə': 'uh', 'iː': 'ee', 'uː': 'oo', 'ɑː': 'ah',
        'eɪ': 'ay', 'aɪ': 'eye', 'ɔɪ': 'oy', 'aʊ': 'ow',
        'əʊ': 'oh', 'ɪə': 'ear', 'eə': 'air', 'ʊə': 'oor',
        'ʊ': 'oo', 'ɒ': 'o', 'ɡ': 'g', 'ʧ': 'ch', 'ʤ': 'j',
        'ː': ':', 'ˈ': "'", 'ˌ': ',', '→': '->', '̃': '~',
        'ɑ': 'ah', 'ɜ': 'er', 'ɜː': 'er', 'ɔ': 'aw',
    }
    
    result = text
    for ipa_char in sorted(ipa_map.keys(), key=len, reverse=True):
        result = result.replace(ipa_char, ipa_map[ipa_char])
    
    result = ''.join(char if ord(char) < 128 else '?' for char in result)
    return result

def safe_pdf_text(text):
    """Ensure text is safe for PDF encoding"""
    if not text:
        return ""
    
    text = str(text)
    text = clean_ipa_for_pdf(text)
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        return text.encode('latin-1', errors='replace').decode('latin-1')

def parse_distortion_patterns(pattern_string):
    """Parse the example distortion patterns from reference data"""
    patterns = []
    if pd.isna(pattern_string) or not pattern_string:
        return patterns
    
    items = pattern_string.split(',')
    for item in items:
        item = item.strip()
        if '→' in item or '->' in item:
            separator = '→' if '→' in item else '->'
            parts = item.split(separator)
            source = parts[0].strip().strip('/')
            target_full = parts[1].strip()
            target = target_full.split('(')[0].strip().strip('/')
            description = ''
            if '(' in target_full:
                description = target_full.split('(')[1].strip(')')
            patterns.append({
                'source': source,
                'target': target,
                'description': description,
                'full': item
            })
    return patterns

def compare_ipa_transcriptions(produced_ipa, expected_ipa):
    """Compare produced IPA with expected IPA and identify differences"""
    produced = produced_ipa.strip('/')
    expected = expected_ipa.strip('/')
    
    matcher = difflib.SequenceMatcher(None, expected, produced)
    differences = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            differences.append({
                'type': tag,
                'expected': expected[i1:i2],
                'produced': produced[j1:j2],
                'position': i1
            })
    
    similarity = matcher.ratio()
    return differences, similarity

def identify_patterns(differences, speech_rules, reference_patterns):
    """Identify potential speech disorder patterns using clinical rules and reference patterns"""
    identified_patterns = []
    
    ref_pattern_dict = {}
    for rp in reference_patterns:
        ref_pattern_dict[rp['source']] = rp
    
    for diff in differences:
        if diff['type'] == 'replace':
            expected = diff['expected']
            produced = diff['produced']
            
            matching_ref = None
            for source, ref_p in ref_pattern_dict.items():
                if expected == source or expected in source:
                    matching_ref = ref_p
                    break
            
            # Gliding patterns
            if (expected in ['r', 'ɹ'] and produced == 'w') or (expected == 'l' and produced in ['w', 'j']):
                rule = speech_rules[speech_rules['Condition'] == 'Gliding']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Gliding',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': matching_ref['description'] if matching_ref else 'Common developmental pattern'
                    })
            
            # Stopping patterns
            elif expected in ['s', 'z', 'f', 'v', 'θ', 'ð', 'ʃ', 'ʒ', 'ʧ', 'ʤ'] and produced in ['t', 'd', 'p', 'b', 'k', 'g']:
                rule = speech_rules[speech_rules['Condition'] == 'Stopping']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Stopping',
                        'pattern': f'{expected}→{produced} (fricative/affricate to stop)',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': matching_ref['description'] if matching_ref else 'Common in young children'
                    })
            
            # Fronting (Velar Fronting)
            elif expected in ['k', 'g', 'ŋ'] and produced in ['t', 'd', 'n']:
                rule = speech_rules[speech_rules['Condition'] == 'Fronting (Velar Fronting)']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Fronting (Velar Fronting)',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': matching_ref['description'] if matching_ref else 'Developmental pattern'
                    })
            
            # Backing
            elif expected in ['t', 'd', 's', 'n'] and produced in ['k', 'g', 'ŋ']:
                rule = speech_rules[speech_rules['Condition'] == 'Backing']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Backing',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Concerning',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': matching_ref['description'] if matching_ref else 'Less common developmentally'
                    })
            
            # Labialization
            elif expected in ['s', 'z', 't', 'd', 'θ', 'ð'] and produced in ['f', 'v', 'p', 'b']:
                rule = speech_rules[speech_rules['Condition'] == 'Labialization']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Labialization',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Mild-Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': matching_ref['description'] if matching_ref else 'May be phonetic or dialectal'
                    })
            
            # Voicing errors
            elif (expected in ['p', 't', 'k', 'f', 's', 'θ', 'ʃ', 'ʧ'] and produced in ['b', 'd', 'g', 'v', 'z', 'ð', 'ʒ', 'ʤ']) or \
                 (expected in ['b', 'd', 'g', 'v', 'z', 'ð', 'ʒ', 'ʤ'] and produced in ['p', 't', 'k', 'f', 's', 'θ', 'ʃ', 'ʧ']):
                rule = speech_rules[speech_rules['Condition'] == 'Voicing (Devoicing/Voicing Errors)']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Voicing Error',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Mild',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': matching_ref['description'] if matching_ref else 'Context-dependent'
                    })
    
    # Check for deletion patterns
    for diff in differences:
        if diff['type'] == 'delete' and diff['expected']:
            rule = speech_rules[speech_rules['Condition'] == 'Final Consonant Deletion']
            if not rule.empty:
                identified_patterns.append({
                    'condition': 'Final Consonant Deletion',
                    'pattern': f"Omission of /{diff['expected']}/",
                    'example': rule.iloc[0]['Rule mapping (example)'],
                    'clinical_notes': rule.iloc[0]['Clinical notes'],
                    'age_concern': rule.iloc[0]['Age_of_concern'],
                    'severity': 'Moderate',
                    'confidence': rule.iloc[0]['Confidence_notes'],
                    'reference_context': 'Affects intelligibility'
                })
                break
    
    return identified_patterns

def calculate_confidence_level(similarity_score, config):
    """Calculate confidence level based on similarity score"""
    if similarity_score >= config['high_confidence']:
        return "High Confidence", "success"
    elif similarity_score >= config['moderate_confidence']:
        return "Moderate Confidence", "warning"
    elif similarity_score >= config['low_confidence']:
        return "Low Confidence", "error"
    else:
        return "Very Low Confidence", "error"

def generate_pdf_report(analysis_results, clinician_notes=""):
    """Generate comprehensive PDF report with ASCII-safe text"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Speech Diagnostic Support Report", ln=True, align='C')
    pdf.ln(5)
    
    # Version and Copyright
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(0, 5, "Version 1.3 | Copyright 2024 Gary Crowfoot", ln=True, align='C')
    pdf.cell(0, 5, "Contact: gary.crowfoot@newcastle.edu.au", ln=True, align='C')
    pdf.ln(5)
    
    # Disclaimer
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This tool provides pattern analysis and confidence scoring only. Diagnosis remains the responsibility of a qualified speech pathologist.")
    pdf.ln(5)
    
    # Timestamp
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Analysis Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Analysis Summary", ln=True)
    pdf.set_font("Arial", '', 10)
    
    phrase_safe = safe_pdf_text(analysis_results['phrase'])
    expected_ipa_clean = safe_pdf_text(analysis_results['expected_ipa'])
    produced_ipa_clean = safe_pdf_text(analysis_results['produced_ipa'])
    
    pdf.cell(0, 8, f"Reference Phrase: {phrase_safe}", ln=True)
    
    # Add transcribed text if available
    if 'transcribed_text' in analysis_results and analysis_results['transcribed_text']:
        transcribed_safe = safe_pdf_text(analysis_results['transcribed_text'])
        pdf.cell(0, 8, f"Transcribed Text: {transcribed_safe}", ln=True)
    
    pdf.cell(0, 8, f"Expected IPA (simplified): {expected_ipa_clean}", ln=True)
    pdf.cell(0, 8, f"Produced IPA (simplified): {produced_ipa_clean}", ln=True)
    pdf.cell(0, 8, f"Similarity Score: {analysis_results['similarity']:.2%}", ln=True)
    pdf.cell(0, 8, f"Confidence Level: {analysis_results['confidence']}", ln=True)
    pdf.ln(5)
    
    # Phoneme Breakdown
    if 'phoneme_breakdown' in analysis_results and analysis_results['phoneme_breakdown']:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, "Phoneme Structure:", ln=True)
        pdf.set_font("Arial", '', 9)
        breakdown_clean = safe_pdf_text(analysis_results['phoneme_breakdown'])
        pdf.multi_cell(0, 5, breakdown_clean)
        pdf.ln(3)
    
    # Differences
    if analysis_results['differences']:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Identified Differences", ln=True)
        pdf.set_font("Arial", '', 10)
        for i, diff in enumerate(analysis_results['differences'], 1):
            exp_clean = safe_pdf_text(diff['expected'])
            prod_clean = safe_pdf_text(diff['produced'])
            pdf.cell(0, 6, f"{i}. Expected: '{exp_clean}' -> Produced: '{prod_clean}'", ln=True)
        pdf.ln(5)
    
    # Clinical Patterns
    if analysis_results['patterns']:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinical Pattern Analysis", ln=True)
        pdf.set_font("Arial", '', 10)
        for i, pattern in enumerate(analysis_results['patterns'], 1):
            pdf.set_font("Arial", 'B', 10)
            condition_safe = safe_pdf_text(pattern['condition'])
            severity_safe = safe_pdf_text(pattern['severity'])
            pdf.cell(0, 6, f"{i}. {condition_safe} - {severity_safe}", ln=True)
            
            pdf.set_font("Arial", '', 9)
            pattern_clean = safe_pdf_text(pattern['pattern'])
            example_clean = safe_pdf_text(pattern['example'])
            clinical_notes_clean = safe_pdf_text(pattern['clinical_notes'])
            age_concern_clean = safe_pdf_text(pattern['age_concern'])
            confidence_clean = safe_pdf_text(pattern['confidence'])
            
            pdf.cell(0, 5, f"   Pattern: {pattern_clean}", ln=True)
            pdf.multi_cell(0, 5, f"   Clinical Notes: {clinical_notes_clean}")
            pdf.cell(0, 5, f"   Age of Concern: {age_concern_clean}", ln=True)
            pdf.cell(0, 5, f"   Confidence: {confidence_clean}", ln=True)
            if 'reference_context' in pattern:
                context_clean = safe_pdf_text(pattern['reference_context'])
                pdf.multi_cell(0, 5, f"   Context: {context_clean}")
            pdf.ln(2)
    
    # Clinician Notes
    if clinician_notes:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinician Notes", ln=True)
        pdf.set_font("Arial", '', 10)
        notes_safe = safe_pdf_text(clinician_notes)
        pdf.multi_cell(0, 6, notes_safe)
    
    # Add note about IPA conversion
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, "Note: IPA symbols have been converted to ASCII-readable format for PDF compatibility. Refer to the web interface for precise phonetic notation.")
    
    return pdf.output(dest='S').encode('latin-1')

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Main App Layout
st.title('🎙️ Speech Diagnostic Support Tool')

# Version and Copyright
col_left, col_right = st.columns([3, 1])
with col_left:
    st.caption("Version 1.3 | © 2024 Gary Crowfoot")
with col_right:
    st.caption("📧 [Contact](mailto:gary.crowfoot@newcastle.edu.au)")

st.info("**For research or collaboration enquiries, please contact:** gary.crowfoot@newcastle.edu.au")

# Disclaimer at top
st.warning("⚠️ **DISCLAIMER:** This tool provides pattern analysis and confidence scoring only. Diagnosis remains the responsibility of a qualified speech pathologist.")

st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Display sensitivity thresholds
    config = load_sensitivity_config()
    st.subheader("Confidence Thresholds")
    st.write(f"🟢 High: ≥ {config['high_confidence']:.0%}")
    st.write(f"🟡 Moderate: ≥ {config['moderate_confidence']:.0%}")
    st.write(f"🔴 Low: ≥ {config['low_confidence']:.0%}")
    
    st.markdown("---")
    
    # Display reference information
    st.subheader("Reference Database")
    reference_df = load_reference_phrases()
    st.write(f"**Test Phrases:** {len(reference_df)}")
    
    speech_rules = load_speech_rules()
    st.write(f"**Clinical Conditions:** {len(speech_rules)}")
    
    with st.expander("View All Test Phrases"):
        st.dataframe(reference_df[['phrase']], height=300)
    
    st.markdown("---")
    
    # API Status
    st.subheader("⚙️ System Status")
    try:
        import openai
        st.success("✅ OpenAI (Whisper)")
    except:
        st.error("❌ OpenAI not installed")
    
    try:
        import epitran
        st.success("✅ Epitran (IPA)")
    except:
        st.error("❌ Epitran not installed")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Select Reference Phrase")
    reference_df = load_reference_phrases()
    
    phrase_options = reference_df['phrase'].tolist()
    
    selected_phrase = st.selectbox(
        "Choose a diagnostic test phrase:",
        options=phrase_options,
        key='phrase_selector',
        help="Select a phrase for the patient to read or repeat",
        format_func=lambda x: x
    )
    
    selected_row = reference_df[reference_df['phrase'] == selected_phrase].iloc[0]
    
    st.info(f"**Expected IPA:** {selected_row['expected_IPA']}")
    
    # Show phoneme breakdown
    if 'phoneme_breakdown' in selected_row and pd.notna(selected_row['phoneme_breakdown']):
        with st.expander("📊 View Phoneme Breakdown"):
            st.code(selected_row['phoneme_breakdown'], language=None)
    
    # Show example patterns
    if 'example_distortion_patterns' in selected_row and pd.notna(selected_row['example_distortion_patterns']):
        with st.expander("🔍 Common Distortion Patterns for This Phrase"):
            st.write(selected_row['example_distortion_patterns'])

with col2:
    st.header("Step 2: Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Upload patient audio recording",
        type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
        help="Supported formats: WAV, MP3, OGG, M4A, FLAC"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        st.caption(f"File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

# Analysis section
st.markdown("---")
st.header("Step 3: Analyze Speech")

col_analyze, col_info = st.columns([1, 2])

with col_analyze:
    analyze_button = st.button("🔍 Run Real-Time Analysis", type="primary", disabled=not uploaded_file)

with col_info:
    st.caption("🎙️ Real-time transcription using OpenAI Whisper + Epitran IPA conversion")

if analyze_button:
    with st.spinner("🎧 Transcribing audio..."):
        # Parse reference distortion patterns
        reference_patterns = parse_distortion_patterns(
            selected_row.get('example_distortion_patterns', '')
        )
        
        # Perform real audio transcription and analysis
        produced_ipa, transcribed_text, success, error = audio_to_ipa_pipeline(uploaded_file, selected_row)
        
        if not success:
            st.error(f"❌ **Error:** {error}")
            st.info("💡 **Setup Instructions:**\n\n"
                   "1. Add to `requirements.txt`:\n"
                   "   ```\n"
                   "   openai\n"
                   "   epitran\n"
                   "   ```\n\n"
                   "2. Add OpenAI API key to Streamlit Secrets:\n"
                   "   - Go to App Settings → Secrets\n"
                   "   - Add: `OPENAI_API_KEY = \"your-key-here\"`\n\n"
                   "3. Redeploy the app")
        else:
            st.success(f"✅ Transcription complete: \"{transcribed_text}\"")
            
            with st.spinner("🔬 Analyzing phonetic patterns..."):
                import time
                time.sleep(0.5)
                
                # Perform comparison and pattern analysis
                differences, similarity = compare_ipa_transcriptions(produced_ipa, selected_row['expected_IPA'])
                patterns = identify_patterns(differences, speech_rules, reference_patterns)
                confidence, confidence_type = calculate_confidence_level(similarity, config)
                
                # Store results
                st.session_state.analysis_results = {
                    'phrase': selected_phrase,
                    'expected_ipa': selected_row['expected_IPA'],
                    'produced_ipa': produced_ipa,
                    'transcribed_text': transcribed_text,
                    'phoneme_breakdown': selected_row.get('phoneme_breakdown', ''),
                    'differences': differences,
                    'similarity': similarity,
                    'patterns': patterns,
                    'confidence': confidence,
                    'confidence_type': confidence_type,
                    'reference_patterns': reference_patterns
                }
                st.session_state.analysis_complete = True
                st.rerun()

# Display results
if st.session_state.analysis_complete and st.session_state.analysis_results:
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    results = st.session_state.analysis_results
    
    # Summary metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Similarity Score", f"{results['similarity']:.1%}")
    
    with metric_col2:
        confidence_color = {
            'success': '🟢',
            'warning': '🟡',
            'error': '🔴'
        }
        st.metric("Confidence Level", 
                 f"{confidence_color.get(results['confidence_type'], '⚪')} {results['confidence']}")
    
    with metric_col3:
        st.metric("Patterns Detected", len(results['patterns']))
    
    with metric_col4:
        if 'transcribed_text' in results:
            word_count = len(results['transcribed_text'].split())
            st.metric("Words Transcribed", word_count)
    
    # Detailed results in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Transcription", "🔬 Phoneme Analysis", "🔍 Clinical Patterns", "📊 Comparison", "📄 Report & Notes"])
    
    with tab1:
        st.subheader("Audio Transcription Results")
        
        if 'transcribed_text' in results and results['transcribed_text']:
            st.markdown("**Transcribed Text:**")
            st.info(f"🎤 \"{results['transcribed_text']}\"")
        
        st.markdown("**Reference Phrase:**")
        st.success(f"📖 \"{results['phrase']}\"")
        
        # Text comparison
        if 'transcribed_text' in results and results['transcribed_text']:
            text_similarity = difflib.SequenceMatcher(None, 
                                                     results['phrase'].lower(), 
                                                     results['transcribed_text'].lower()).ratio()
            st.metric("Text Accuracy", f"{text_similarity:.1%}")
    
    with tab2:
        st.subheader("Phonetic Transcription")
        
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("**Expected IPA:**")
            st.code(results['expected_ipa'], language=None)
        
        with comp_col2:
            st.markdown("**Produced IPA:**")
            st.code(results['produced_ipa'], language=None)
        
        if results.get('phoneme_breakdown'):
            st.markdown("---")
            st.markdown("**Word-Level Phoneme Breakdown:**")
            st.code(results['phoneme_breakdown'], language=None)
            st.caption("Phonemes are grouped by word boundaries (separated by |)")
        
        # Show reference patterns for this phrase
        if results.get('reference_patterns'):
            st.markdown("---")
            st.subheader("Expected Distortion Patterns for This Phrase")
            for rp in results['reference_patterns']:
                st.write(f"• **{rp['full']}**")
                if rp['description']:
                    st.caption(f"  _{rp['description']}_")
    
    with tab3:
        st.subheader("Clinical Pattern Analysis")
        
        if results['patterns']:
            for i, pattern in enumerate(results['patterns'], 1):
                with st.expander(f"Pattern {i}: {pattern['condition']} - {pattern['severity']}", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Observed Pattern:** {pattern['pattern']}")
                        st.write(f"**Example:** {pattern['example']}")
                        st.write(f"**Severity:** {pattern['severity']}")
                    
                    with col_b:
                        st.write(f"**Age of Concern:** {pattern['age_concern']}")
                        st.write(f"**Confidence:** {pattern['confidence']}")
                    
                    st.markdown("**Clinical Notes:**")
                    st.info(pattern['clinical_notes'])
                    
                    if 'reference_context' in pattern:
                        st.markdown("**Context:**")
                        st.caption(pattern['reference_context'])
        else:
            st.info("No specific clinical patterns identified. This may indicate typical speech production or require further assessment with additional samples.")
        
        # Show clinical database reference
        if results['patterns']:
            st.markdown("---")
            st.subheader("📚 Related Clinical Information")
            conditions_detected = [p['condition'] for p in results['patterns']]
            related_rules = speech_rules[speech_rules['Condition'].isin(conditions_detected)]
            if not related_rules.empty:
                st.dataframe(related_rules, hide_index=True)
    
    with tab4:
        st.subheader("Detailed Phoneme Comparison")
        
        if results['differences']:
            st.markdown("**Phoneme-Level Differences:**")
            diff_data = []
            for diff in results['differences']:
                diff_data.append({
                    'Type': diff['type'].title(),
                    'Expected': diff['expected'] if diff['expected'] else '(none)',
                    'Produced': diff['produced'] if diff['produced'] else '(none)',
                    'Position': diff['position']
                })
            st.dataframe(pd.DataFrame(diff_data), hide_index=True)
            
            # Visual similarity bar
            st.markdown("---")
            st.markdown("**Phonetic Similarity:**")
            st.progress(results['similarity'])
            st.caption(f"Overall match: {results['similarity']:.1%}")
        else:
            st.success("✅ No significant phoneme-level differences detected!")
    
    with tab5:
        st.subheader("Clinical Notes")
        
        clinician_notes = st.text_area(
            "Add clinical observations and notes:",
            height=150,
            placeholder="Enter any additional observations, context, or clinical insights...\n\nExample:\n- Client history and background\n- Testing environment and conditions\n- Audio quality observations\n- Additional observations during assessment\n- Recommended follow-up actions\n- Other relevant clinical information"
        )
        
        st.markdown("---")
        st.subheader("Download Report")
        
        col_pdf, col_csv = st.columns(2)
        
        with col_pdf:
            if st.button("📥 Generate PDF Report", type="secondary"):
                pdf_data = generate_pdf_report(results, clinician_notes)
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=f"speech_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col_csv:
            # Prepare detailed CSV data
            csv_rows = []
            for i, pattern in enumerate(results['patterns'], 1):
                csv_rows.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Reference Phrase': results['phrase'],
                    'Transcribed Text': results.get('transcribed_text', ''),
                    'Expected IPA': results['expected_ipa'],
                    'Produced IPA': results['produced_ipa'],
                    'Phoneme Breakdown': results.get('phoneme_breakdown', ''),
                    'Similarity Score': f"{results['similarity']:.2%}",
                    'Confidence Level': results['confidence'],
                    'Pattern Number': i,
                    'Condition': pattern['condition'],
                    'Pattern': pattern['pattern'],
                    'Severity': pattern['severity'],
                    'Age of Concern': pattern['age_concern'],
                    'Clinical Notes': pattern['clinical_notes'],
                    'Reference Context': pattern.get('reference_context', ''),
                    'Clinician Notes': clinician_notes
                })
            
            if not csv_rows:
                csv_rows.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Reference Phrase': results['phrase'],
                    'Transcribed Text': results.get('transcribed_text', ''),
                    'Expected IPA': results['expected_ipa'],
                    'Produced IPA': results['produced_ipa'],
                    'Phoneme Breakdown': results.get('phoneme_breakdown', ''),
                    'Similarity Score': f"{results['similarity']:.2%}",
                    'Confidence Level': results['confidence'],
                    'Pattern Number': 0,
                    'Condition': 'None detected',
                    'Pattern': 'N/A',
                    'Severity': 'N/A',
                    'Age of Concern': 'N/A',
                    'Clinical Notes': 'No patterns identified',
                    'Reference Context': '',
                    'Clinician Notes': clinician_notes
                })
            
            csv_data = pd.DataFrame(csv_rows)
            csv_buffer = io.StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"speech_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Speech Diagnostic Support Tool v1.3 | © 2024 Gary Crowfoot</p>
    <p>For clinical use by qualified speech pathologists only</p>
    <p>This tool is a prototype for pattern analysis and should not replace professional clinical judgment</p>
    <p>For research or collaboration enquiries: <a href="mailto:gary.crowfoot@newcastle.edu.au">gary.crowfoot@newcastle.edu.au</a></p>
    </div>
    """,
    unsafe_allow_html=True
)