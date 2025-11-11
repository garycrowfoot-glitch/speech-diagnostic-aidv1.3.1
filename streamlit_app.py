# streamlit_app.py
import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import difflib
from fpdf import FPDF
import tempfile
import os
import numpy as np

# Optional heavy deps are imported lazily:
# torch/transformers/torchaudio/librosa used only when phoneme ASR is enabled.

# Page configuration
st.set_page_config(page_title="Speech Diagnostic Support Tool", page_icon="🎙️", layout="wide")

# ----------------------------
# Helpers & Data Loaders
# ----------------------------

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
            'phrase': [
                'The cat sat on the mat',
                'She sells seashells by the seashore'
            ],
            'expected_IPA': [
                '/ðə kæt sæt ɒn ðə mæt/',
                '/ʃi sɛlz siːʃɛlz baɪ ðə siːʃɔː/'
            ],
            'phoneme_breakdown': [
                'ð ə \n k æ t \n s æ t \n ɒ n \n ð ə \n m æ t',
                'ʃ i \n s ɛ l z \n s iː ʃ ɛ l z \n b aɪ \n ð ə \n s iː ʃ ɔː'
            ],
            'example_distortion_patterns': [
                '/θ/→/f/, /t/→glottal stop',
                '/ʃ/→/s/ (lisp), /z/→/d/'
            ]
        }
        return pd.DataFrame(data)

@st.cache_data
def load_sensitivity_config():
    """Confidence thresholds"""
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
    """Load clinical rules"""
    try:
        df = pd.read_csv('speech_rule_mapping.csv')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.warning("⚠️ speech_rule_mapping.csv not found. Using basic rule set.")
        data = {
            'Condition': ['Articulation Disorder', 'Phonological Disorder', 'Gliding', 'Stopping', 'Final Consonant Deletion'],
            'Typical phonetic/phonological patterns': [
                'Difficulty producing specific speech sounds',
                'Patterned simplifications of sounds',
                'Approximant → glide (r/ɹ→w, l→w/j)',
                'Fricative/affricate → stop (s→t, ʃ→t, etc.)',
                'Omission of final consonant'
            ],
            'Rule mapping (example)': [
                'r→w substitution',
                'cluster reduction',
                'ɹ→w',
                'ʃ→t',
                'C(final)→∅'
            ],
            'Clinical notes': [
                'Motor-based errors',
                'Language-based phonological knowledge difference',
                'Common developmental pattern; monitor age',
                'Often developmental; impacts intelligibility',
                'Impacts intelligibility, monitor age'
            ],
            'Age_of_concern': ['Any age', 'Typically in childhood', '3–5 yrs', '3–5 yrs', '3–5 yrs'],
            'Confidence_notes': ['High', 'High when multiple processes', 'Moderate', 'Moderate', 'Moderate']
        }
        return pd.DataFrame(data)

# ----------------------------
# Diagnostics Helpers
# ----------------------------

def ipa_diagnostics(ipa_text: str):
    """Check IPA Unicode ranges for sanity."""
    s = (ipa_text or "").strip().strip('/')
    result = {
        "length": len(s),
        "ascii_only": all(ord(ch) < 128 for ch in s),
        "has_ipa_ext": any(0x0250 <= ord(ch) <= 0x02AF for ch in s),
        "has_modifiers": any(0x02B0 <= ord(ch) <= 0x02FF for ch in s),
        "has_combining_diacritics": any(0x0300 <= ord(ch) <= 0x036F for ch in s),
        "sample_non_ascii": "".join(ch for ch in s if ord(ch) >= 128)[:30],
        "codepoints": " ".join(f"U+{ord(ch):04X}" for ch in s[:50])
    }
    return result

# ----------------------------
# OpenAI Whisper Transcription (text)
# ----------------------------

def transcribe_audio_to_text(audio_file):
    """
    Transcribe audio file to text using OpenAI Whisper or gpt-4o-transcribe API.
    Returns: (transcribed_text, success_flag, error_message)
    """
    try:
        import openai
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return None, False, "OpenAI API key not configured. Please add OPENAI_API_KEY to Streamlit secrets."
        client = openai.OpenAI(api_key=api_key)

        # Write temp file preserving suffix
        orig_suffix = os.path.splitext(getattr(audio_file, 'name', 'input.wav'))[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=orig_suffix) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        try:
            with open(tmp_path, 'rb') as audio:
                model = st.secrets.get("TRANSCRIBE_MODEL", "whisper-1")
                transcript = client.audio.transcriptions.create(
                    model=model,
                    file=audio,
                    language="en",
                    response_format="text"
                )
            return transcript, True, None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except ImportError:
        return None, False, "OpenAI library not installed. Please add 'openai' to requirements.txt"
    except Exception as e:
        return None, False, f"Transcription error: {str(e)}"

# ----------------------------
# Phoneme ASR (Audio -> Phones) — Option 1 (default)
# ----------------------------

@st.cache_resource(show_spinner=False)
def load_phoneme_asr_model():
    """
    Load a phoneme-level ASR model from Hugging Face if configured in secrets.PHONEME_MODEL_ID.
    Returns (processor, model, inventory_type) or (None, None, None).
    inventory_type: 'ARPABET' (common) or 'UNKNOWN' (assumed IPA-like tokens)
    """
    model_id = st.secrets.get("PHONEME_MODEL_ID", "")
    if not model_id:
        return None, None, None
    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        vocab = processor.tokenizer.get_vocab() if hasattr(processor, "tokenizer") else {}
        arpabet_keys = {'AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY',
                        'F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY',
                        'P','R','S','SH','T','TH','UH','UW','V','W','Y','Z','ZH'}
        inv = 'ARPABET' if any(k in vocab for k in arpabet_keys) else 'UNKNOWN'
        return processor, model, inv
    except Exception as e:
        st.warning(f"Phoneme ASR model load failed: {type(e).__name__}: {e}")
        return None, None, None

ARPABET_TO_IPA = {
    "AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ","AW":"aʊ","AY":"aɪ",
    "B":"b","CH":"ʧ","D":"d","DH":"ð","EH":"ɛ","ER":"ɝ","EY":"eɪ",
    "F":"f","G":"ɡ","HH":"h","IH":"ɪ","IY":"i","JH":"ʤ","K":"k",
    "L":"l","M":"m","N":"n","NG":"ŋ","OW":"oʊ","OY":"ɔɪ",
    "P":"p","R":"ɹ","S":"s","SH":"ʃ","T":"t","TH":"θ","UH":"ʊ",
    "UW":"u","V":"v","W":"w","Y":"j","Z":"z","ZH":"ʒ",
}

def bytes_to_waveform(file_bytes, sr_target=16000):
    import librosa
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr_target, mono=True)
    return y, sr

def audio_to_phones_via_asr(file_bytes):
    """
    Return (phones_string, inventory_type, success, error) from phoneme ASR if configured.
    phones_string: space-delimited; inventory may be 'ARPABET' or 'UNKNOWN' (assumed IPA-like).
    """
    processor, model, inv = load_phoneme_asr_model()
    if processor is None or model is None:
        return None, None, False, "Phoneme ASR model not configured."
    try:
        import torch
        y, sr = bytes_to_waveform(file_bytes)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)
        seq = processor.batch_decode(ids)[0]
        seq = " ".join(seq.split())  # normalize spaces
        return seq, inv, True, None
    except Exception as e:
        return None, None, False, f"Phoneme ASR error: {type(e).__name__}: {e}"

def arpabet_to_ipa_sequence(arp_seq: str) -> str:
    ipa_tokens = []
    for token in arp_seq.split():
        base = ''.join([c for c in token if c.isalpha()])  # strip stress digits (e.g., AE1)
        ipa_tokens.append(ARPABET_TO_IPA.get(base, base))
    return ' '.join(ipa_tokens)

# ----------------------------
# G2P / Phonemizer fallback (Text -> IPA phones)
# ----------------------------

def text_to_ipa_phones(text: str, dialect='en-us'):
    """Use phonemizer (espeak-ng backend) to produce IPA; fallback to eng_to_ipa."""
    text = (text or '').strip()
    if not text:
        return ""
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        sep = Separator(phone=' ', word=' | ')
        ipa = phonemize(text, language=dialect, backend='espeak', separator=sep,
                        strip=False, preserve_punctuation=False, with_stress=True)
        return ipa
    except Exception:
        try:
            import eng_to_ipa as eipa
            return eipa.convert(text)
        except Exception:
            return text  # last resort

# ----------------------------
# Comparison & Pattern Analysis
# ----------------------------

def parse_distortion_patterns(pattern_string):
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
    identified = []
    ref_map = {rp['source']: rp for rp in reference_patterns}
    for diff in differences:
        if diff['type'] == 'replace':
            expected = diff['expected']
            produced = diff['produced']
            matching_ref = None
            for source, rp in ref_map.items():
                if expected == source or expected in source:
                    matching_ref = rp
                    break
            # Gliding
            if (expected in ['r', 'ɹ'] and produced == 'w') or (expected == 'l' and produced in ['w', 'j']):
                rule = speech_rules[speech_rules['Condition'] == 'Gliding']
                if not rule.empty:
                    identified.append({
                        'condition': 'Gliding',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': (matching_ref or {}).get('description', 'Common developmental pattern')
                    })
            # Stopping
            elif expected in ['s','z','f','v','θ','ð','ʃ','ʒ','ʧ','ʤ'] and produced in ['t','d','p','b','k','g']:
                rule = speech_rules[speech_rules['Condition'] == 'Stopping']
                if not rule.empty:
                    identified.append({
                        'condition': 'Stopping',
                        'pattern': f'{expected}→{produced} (fricative/affricate → stop)',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes'],
                        'reference_context': (matching_ref or {}).get('description', 'Common in young children')
                    })
    # Final consonant deletion (simple heuristic)
    for diff in differences:
        if diff['type'] == 'delete' and diff['expected']:
            rule = speech_rules[speech_rules['Condition'] == 'Final Consonant Deletion']
            if not rule.empty:
                identified.append({
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
    return identified

def calculate_confidence_level(similarity_score, config):
    if similarity_score >= config['high_confidence']:
        return "High Confidence", "success"
    elif similarity_score >= config['moderate_confidence']:
        return "Moderate Confidence", "warning"
    elif similarity_score >= config['low_confidence']:
        return "Low Confidence", "error"
    else:
        return "Very Low Confidence", "error"

# ----------------------------
# PDF helpers
# ----------------------------

def clean_ipa_for_pdf(text):
    if not text:
        return ""
    text = str(text).strip('/')
    ipa_map = {
        'ð':'dh','θ':'th','ʃ':'sh','ʒ':'zh','ŋ':'ng','ɹ':'r','ɔː':'aw','ɔ':'o',
        'æ':'a','ɛ':'e','ɪ':'i','ʌ':'u','ə':'uh','iː':'ee','uː':'oo','ɑː':'ah',
        'eɪ':'ay','aɪ':'eye','ɔɪ':'oy','aʊ':'ow','əʊ':'oh','ɪə':'ear','eə':'air','ʊə':'oor',
        'ʊ':'oo','ɒ':'o','ɡ':'g','ʧ':'ch','ʤ':'j','ː':':','ˈ':"'",'ˌ':','
    }
    result = text
    for ipa_char in sorted(ipa_map.keys(), key=len, reverse=True):
        result = result.replace(ipa_char, ipa_map[ipa_char])
    result = ''.join(char if ord(char) < 128 else '?' for char in result)
    return result

def safe_pdf_text(text):
    if not text:
        return ""
    text = str(text)
    text = clean_ipa_for_pdf(text)
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        return text.encode('latin-1', errors='replace').decode('latin-1')

# ----------------------------
# Report
# ----------------------------

def generate_pdf_report(analysis_results, clinician_notes=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Speech Diagnostic Support Report", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(0, 5, "Version 1.4 \n Copyright 2025 Gary Crowfoot", ln=True, align='C')
    pdf.cell(0, 5, "Contact: gary.crowfoot@newcastle.edu.au", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This tool provides pattern analysis and confidence scoring only. Diagnosis remains the responsibility of a qualified speech pathologist.")
    pdf.ln(5)

    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Analysis Summary", ln=True)
    pdf.set_font("Arial", '', 10)
    phrase_safe = safe_pdf_text(analysis_results['phrase'])
    expected_ipa_clean = safe_pdf_text(analysis_results['expected_ipa'])
    produced_ipa_clean = safe_pdf_text(analysis_results['produced_ipa'])
    pdf.cell(0, 8, f"Reference Phrase: {phrase_safe}", ln=True)

    if 'transcribed_text' in analysis_results and analysis_results['transcribed_text']:
        transcribed_safe = safe_pdf_text(analysis_results['transcribed_text'])
        pdf.cell(0, 8, f"Transcribed Text: {transcribed_safe}", ln=True)
    pdf.cell(0, 8, f"Expected IPA (simplified): {expected_ipa_clean}", ln=True)
    pdf.cell(0, 8, f"Produced IPA (simplified): {produced_ipa_clean}", ln=True)
    pdf.cell(0, 8, f"Similarity Score: {analysis_results['similarity']:.2%}", ln=True)
    pdf.cell(0, 8, f"Confidence Level: {analysis_results['confidence']}", ln=True)
    pdf.ln(5)

    if analysis_results['differences']:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Identified Differences", ln=True)
        pdf.set_font("Arial", '', 10)
        for i, diff in enumerate(analysis_results['differences'], 1):
            exp_clean = safe_pdf_text(diff['expected'])
            prod_clean = safe_pdf_text(diff['produced'])
            pdf.cell(0, 6, f"{i}. Expected: '{exp_clean}' -> Produced: '{prod_clean}'", ln=True)
        pdf.ln(5)

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
            clinical_notes_clean = safe_pdf_text(pattern['clinical_notes'])
            age_concern_clean = safe_pdf_text(pattern['age_concern'])
            confidence_clean = safe_pdf_text(pattern['confidence'])
            pdf.cell(0, 5, f" Pattern: {pattern_clean}", ln=True)
            pdf.multi_cell(0, 5, f" Clinical Notes: {clinical_notes_clean}")
            pdf.cell(0, 5, f" Age of Concern: {age_concern_clean}", ln=True)
            pdf.cell(0, 5, f" Confidence: {confidence_clean}", ln=True)
            if 'reference_context' in pattern:
                context_clean = safe_pdf_text(pattern['reference_context'])
                pdf.multi_cell(0, 5, f" Context: {context_clean}")
            pdf.ln(2)

    if clinician_notes:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinician Notes", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, safe_pdf_text(clinician_notes))

    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, "Note: IPA symbols have been converted to ASCII-readable format for PDF compatibility. Refer to the web interface for precise phonetic notation.")
    return pdf.output(dest='S').encode('latin-1')

# ----------------------------
# Session state
# ----------------------------
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# ----------------------------
# Main UI
# ----------------------------
st.title('🎙️ Speech Diagnostic Support Tool')
col_left, col_right = st.columns([3, 1])
with col_left:
    st.caption("Version 1.4 \n © 2025 Gary Crowfoot")
with col_right:
    st.caption("📧 gary.crowfoot@newcastle.edu.au")

st.info("**For research or collaboration enquiries, please contact:** gary.crowfoot@newcastle.edu.au")
st.warning("⚠️ **DISCLAIMER:** This tool provides pattern analysis and confidence scoring only. Diagnosis remains the responsibility of a qualified speech pathologist.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    config = load_sensitivity_config()
    st.subheader("Confidence Thresholds")
    st.write(f"🟢 High: ≥ {config['high_confidence']:.0%}")
    st.write(f"🟡 Moderate: ≥ {config['moderate_confidence']:.0%}")
    st.write(f"🔴 Low: ≥ {config['low_confidence']:.0%}")
    st.markdown("---")

    st.subheader("Reference Database")
    reference_df = load_reference_phrases()
    st.write(f"**Test Phrases:** {len(reference_df)}")
    speech_rules = load_speech_rules()
    st.write(f"**Clinical Conditions:** {len(speech_rules)}")
    with st.expander("View All Test Phrases"):
        st.dataframe(reference_df[['phrase']], height=300)
    st.markdown("---")

    st.subheader("⚙️ System Status")
    try:
        import openai
        st.success("✅ OpenAI (Transcription)")
    except Exception:
        st.error("❌ OpenAI not installed")

    proc, mdl, inv = load_phoneme_asr_model()
    if proc is not None and mdl is not None:
        st.success(f"✅ Phoneme ASR model loaded ({inv or 'unknown inventory'})")
    else:
        st.warning("⚠️ Phoneme ASR not configured. Falling back to G2P after transcription.")

    st.markdown("---")
    st.subheader("Phone Source")
    phone_source = st.radio(
        "Select phone source",
        ["Phoneme ASR (audio→phones)", "Dictionary (G2P)"],
        index=0  # Option 1 is default
    )
    dialect = st.radio("IPA dialect (for G2P)", ["en-us", "en-gb"], index=0)

# Main content
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

    if 'phoneme_breakdown' in selected_row and pd.notna(selected_row['phoneme_breakdown']):
        with st.expander("📊 View Phoneme Breakdown"):
            st.code(selected_row['phoneme_breakdown'], language=None)
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

st.markdown("---")
st.header("Step 3: Analyze Speech")
col_analyze, col_info = st.columns([1, 2])
with col_analyze:
    analyze_button = st.button("🔍 Run Real-Time Analysis", type="primary", disabled=not uploaded_file)
with col_info:
    st.caption("🎙️ Pipeline: Audio → Phones (ASR) → IPA → Comparison")

if analyze_button:
    with st.spinner("🎧 Processing audio..."):
        # Transcribe (for display & fallback)
        transcribed_text, ts_success, ts_error = transcribe_audio_to_text(uploaded_file)
        if not ts_success:
            st.error(f"❌ **Transcription Error:** {ts_error or 'Unknown error.'}")
            st.stop()

        # Parse expected distortion patterns
        reference_patterns = parse_distortion_patterns(selected_row.get('example_distortion_patterns', ''))

        produced_ipa = ""
        backend_used = ""
        if phone_source.startswith("Phoneme ASR"):
            phones, inv, ok, err = audio_to_phones_via_asr(uploaded_file.getvalue())
            if not ok:
                st.warning(f"⚠️ Phoneme ASR unavailable: {err}. Using G2P fallback.")
                ipa_phones = text_to_ipa_phones(transcribed_text, dialect=dialect)
                produced_ipa = f"/{ipa_phones}/"
                backend_used = f"G2P ({dialect})"
            else:
                # Convert inventory to IPA if needed
                if inv == 'ARPABET':
                    produced_ipa = f"/{arpabet_to_ipa_sequence(phones)}/"
                else:
                    produced_ipa = f"/{phones}/"  # assume already IPA-like
                backend_used = f"Phoneme ASR ({inv or 'unknown'})"
        else:
            ipa_phones = text_to_ipa_phones(transcribed_text, dialect=dialect)
            produced_ipa = f"/{ipa_phones}/"
            backend_used = f"G2P ({dialect})"

        differences, similarity = compare_ipa_transcriptions(produced_ipa, selected_row['expected_IPA'])
        patterns = identify_patterns(differences, speech_rules, reference_patterns)
        confidence, confidence_type = calculate_confidence_level(similarity, config)

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
            'reference_patterns': reference_patterns,
            'backend_used': backend_used,
        }
        st.session_state.analysis_complete = True
        st.rerun()

# Results
if st.session_state.analysis_complete and st.session_state.analysis_results:
    st.markdown("---")
    st.header("📊 Analysis Results")
    results = st.session_state.analysis_results

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Similarity Score", f"{results['similarity']:.1%}")
    with metric_col2:
        confidence_color = {'success': '🟢', 'warning': '🟡', 'error': '🔴'}
        st.metric("Confidence Level", f"{confidence_color.get(results['confidence_type'], '⚪')} {results['confidence']}")
    with metric_col3:
        st.metric("Patterns Detected", len(results['patterns']))
    with metric_col4:
        word_count = len((results.get('transcribed_text') or '').split())
        st.metric("Words Transcribed", word_count)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Transcription", "🔬 Phoneme/IPA", "🔍 Clinical Patterns", "📈 Comparison", "📄 Report & Notes"])

    with tab1:
        st.subheader("Audio Transcription Results")
        st.markdown("**Transcribed Text:**")
        st.info(f"🎤 \"{results.get('transcribed_text', '')}\"")
        st.markdown("**Reference Phrase:**")
        st.success(f"📖 \"{results['phrase']}\"")
        text_similarity = difflib.SequenceMatcher(None, results['phrase'].lower(), (results.get('transcribed_text') or '').lower()).ratio()
        st.metric("Text Accuracy", f"{text_similarity:.1%}")

    with tab2:
        st.subheader("Phonetic Transcription")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("**Expected IPA:**")
            st.code(results['expected_ipa'], language=None)
        with comp_col2:
            st.markdown(f"**Produced IPA ({results.get('backend_used','')})**")
            st.code(results['produced_ipa'], language=None)

        st.markdown("---")
        st.markdown("### 🔎 IPA Diagnostics")
        diag = ipa_diagnostics(results['produced_ipa'])
        st.write(f"- Length: {diag['length']}")
        st.write(f"- ASCII only: {diag['ascii_only']}")
        st.write(f"- Has IPA Extensions (U+0250–U+02AF): {diag['has_ipa_ext']}")
        st.write(f"- Has modifier letters (U+02B0–U+02FF): {diag['has_modifiers']}")
        st.write(f"- Has combining diacritics (U+0300–U+036F): {diag['has_combining_diacritics']}")
        if diag['sample_non_ascii']:
            st.write(f"- Sample non-ASCII characters: `{diag['sample_non_ascii']}`")
        with st.expander("Show code points (first 50 characters)"):
            st.code(diag['codepoints'])

        if results.get('phoneme_breakdown'):
            st.markdown("---")
            st.markdown("**Word-Level Phoneme Breakdown:**")
            st.code(results['phoneme_breakdown'], language=None)
            st.caption("Phonemes are grouped by word boundaries (separated by \\n)")

        if results.get('reference_patterns'):
            st.markdown("---")
            st.subheader("Expected Distortion Patterns for This Phrase")
            for rp in results['reference_patterns']:
                st.write(f"• **{rp['full']}**")
                if rp['description']:
                    st.caption(f" _{rp['description']}_")

    with tab3:
        st.subheader("Clinical Pattern Analysis")
        if results['patterns']:
            for i, pattern in enumerate(results['patterns'], 1):
                with st.expander(f"Pattern {i}: {pattern['condition']} - {pattern['severity']}", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Observed Pattern:** {pattern['pattern']}")
                        st.write(f"**Example:** {pattern.get('example','')}")
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
            diff_rows = []
            for diff in results['differences']:
                diff_rows.append({
                    'Type': diff['type'].title(),
                    'Expected': diff['expected'] if diff['expected'] else '(none)',
                    'Produced': diff['produced'] if diff['produced'] else '(none)',
                    'Position': diff['position']
                })
            st.dataframe(pd.DataFrame(diff_rows), hide_index=True)
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
            placeholder=(
                "Enter any additional observations, context, or clinical insights...\n\n"
                "Example:\n- Client history and background\n- Testing environment and conditions\n"
                "- Audio quality observations\n- Additional observations during assessment\n"
                "- Recommended follow-up actions\n- Other relevant clinical information"
            )
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
            rows = []
            for i, pattern in enumerate(results['patterns'], 1):
                rows.append({
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
            if not rows:
                rows.append({
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
            csv_data = pd.DataFrame(rows)
            buf = io.StringIO()
            csv_data.to_csv(buf, index=False)
            st.download_button(
                label="Download CSV",
                data=buf.getvalue(),
                file_name=f"speech_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Speech Diagnostic Support Tool v1.4 \n © 2025 Gary Crowfoot</p>
    <p>For clinical use by qualified speech pathologists only</p>
    <p>This tool is a prototype for pattern analysis and should not replace professional clinical judgment</p>
    <p>For research or collaboration enquiries: <a href="mailto:gary.crowfoot@newcastle.edu.au">ot@newcastle.edu.au</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
