import streamlit as st
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import sqlite3

st.set_page_config(page_title="Exepert review AI Agent", page_icon="🔬", layout="wide")

# DB Initialization
def init_db():
    conn = sqlite3.connect("spectroscopy_results.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS llm_result (
            sample_number TEXT PRIMARY KEY,
            peaks TEXT,
            result TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Query DB
def get_existing_result(sample_number):
    if not sample_number:
        return None, None
    conn = sqlite3.connect("spectroscopy_results.db")
    c = conn.cursor()
    c.execute("SELECT peaks, result FROM llm_result WHERE sample_number = ?", (str(sample_number),))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0]), json.loads(row[1])
    return None, None

# Save to DB
def save_result(sample_number, peaks, result):
    if not sample_number:
        return
    conn = sqlite3.connect("spectroscopy_results.db")
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO llm_result (sample_number, peaks, result)
        VALUES (?, ?, ?)
    ''', (str(sample_number), json.dumps(peaks), json.dumps(result)))
    conn.commit()
    conn.close()

def show_results(result):
    st.subheader("🤖 Analysis Results")
    conf_color = "green" if result.get('confidence', '').lower() == 'high' else "orange" if result.get('confidence', '').lower() == 'medium' else "red"
    st.markdown(f"### **Predicted Drug:** <span style='color:#1E88E5;'>{result.get('predicted_drug', 'None')}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence:** <span style='color:{conf_color}; font-weight:bold;'>{result.get('confidence', 'N/A').title()}</span>", unsafe_allow_html=True)
    st.info(f"**Reasoning:** {result.get('reasoning', '')}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Matched Peaks", "Missing Critical Peaks", "Alternative Candidates", "JSON Data"])
    with tab1:
        if result.get("matched_peaks"):
            st.dataframe(result["matched_peaks"], width='stretch')
        else:
            st.write("No peaks matched.")
    with tab2:
        if result.get("missing_critical_peaks"):
            st.warning(", ".join(map(str, result["missing_critical_peaks"])))
        else:
            st.success("No missing critical peaks!")
    with tab3:
        if result.get("alternative_candidates"):
            st.json(result["alternative_candidates"])
        else:
            st.write("None")
    with tab4:
        st.json(result)

# Import required functions from provided oak_identification.py
# Make sure oak_identification.py and its dependencies are in Python path
try:
    from oak_identification import flourencense_identify, preprocess
except Exception as e:
    st.error(f"Failed to import from oak_identification: {e}")

# Try importing Gemini Client; use legacy if Client not found
try:
    from google import genai
    HAS_NEW_GENAI = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_NEW_GENAI = False
    except ImportError:
        HAS_NEW_GENAI = None

# Initialize Gemini setup
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

PROMINENCE = 0.05

def find_xcal_in_json(data):
    """
    Recursively search for 'xcal' and some form of sample number in the JSON data.
    """
    if isinstance(data, dict):
        xcal = None
        sample_num = None
        for k, v in data.items():
            if k.lower() == 'xcal' or k.lower() == 'excal':
                xcal = v
            if 'number' in k.lower() or 'sample' in k.lower() or 'id' in k.lower():
                sample_num = v
        if xcal is not None:
            return xcal, sample_num
        
        # If not found at this level, recurse
        for k, v in data.items():
            result = find_xcal_in_json(v)
            if result[0] is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_xcal_in_json(item)
            if result[0] is not None:
                return result
    return None, None

def generate_llm_analysis(sample_peaks, kb):
    prompt = f"""You are a forensic Raman spectroscopy expert.

Your task is to analyze a sample spectrum and determine the most likely drug based on the provided knowledge base.

========================
INPUT SAMPLE PEAKS
========================
{json.dumps(sample_peaks, indent=2)}

========================
KNOWLEDGE BASE
========================
{json.dumps(kb, indent=2)}

========================
INSTRUCTIONS
========================

1. Compare sample peaks with each drug in the knowledge base
2. Consider:   
   - Peak matching within tolerance
   - Importance levels (critical > high > medium > low)
   - Presence of key peak ranges
   - Signature rules
3. Identify:   
   - Best matching drug
   - Alternative possible drugs (if any)
4. Clearly explain reasoning:   
   - Which peaks matched
   - Which critical peaks are present/missing
   - Why selected drug is correct
5. If confidence is low, strictly say "Inconclusive"

========================
OUTPUT FORMAT (STRICT JSON)
========================

{{
  "predicted_drug": "",
  "confidence": "high/medium/low",
  "reasoning": "",
  "matched_peaks": [],
  "missing_critical_peaks": [],
  "alternative_candidates": []
}}
"""
    if HAS_NEW_GENAI is True:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)
    elif HAS_NEW_GENAI is False:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    else:
        return {"error": "Generative AI module not available."}


def process_xcal(xcal):
    # Ensure xcal is a list of floats
    if isinstance(xcal, str):
        # If it's a stringified JSON list, strip brackets and parse
        import ast
        try:
            # Try to parse it as a JSON array string
            parsed_xcal = json.loads(xcal)
            if isinstance(parsed_xcal, list):
                xcal = [float(x) for x in parsed_xcal]
            else:
                xcal = [float(x) for x in xcal.replace('[', '').replace(']', '').replace(',', ' ').split()]
        except Exception:
            # Fallback string parsing
            xcal = [float(x) for x in xcal.replace('[', '').replace(']', '').replace(',', ' ').split()]
    else:
        # If it's a list, convert elements to float
        xcal = [float(x) for x in xcal]

    is_fluorescent = flourencense_identify(xcal)
    st.write(f"**Fluorescence Check:** {'Fluorescent' if is_fluorescent else 'Not Fluorescent'}")
    
    if is_fluorescent:
        offset = 400
        # If length is less than expected, we handle gracefully
        # Original logic assumes length to be sufficient for 200..2601 mapping
        try:
            x_series = pd.Series(xcal, index=np.arange(200, 200+len(xcal)))
            intensity_array = preprocess(x_series, smoothing=True, idx=(400, 1700))
        except Exception as e:
            st.warning(f"Using fallback preprocessing due to: {e}")
            intensity_array = preprocess(xcal)
    else:
        offset = 200
        intensity_array = preprocess(xcal, null=True)
        
    intensity_array = np.array(intensity_array)
    peaks, properties = find_peaks(intensity_array, prominence=PROMINENCE)
    
    peak_data = []
    # If the offset preprocessing returned a sliced array, we must ensure peaks track physical position well.
    # We will use the offset that is defined by whether it's fluorescent or not
    
    for idx, prom in zip(peaks, properties["prominences"]):
        peak_data.append({
            "position": int(idx + offset),
            "intensity": float(intensity_array[idx]),
            "prominence": float(prom)
        })
        
    return peak_data

st.title("🔬 Expert  Review AI Agent")
st.markdown("Upload a JSON containing sample information and `xcal` points to analyze the spectral signature automatically.")
st.divider()

kb_path = "Updated_Knowledge_base.json"
if os.path.exists(kb_path):
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
else:
    st.error(f"Knowledge base not found at {kb_path}")
    st.stop()

uploaded_file = st.file_uploader("Upload Spectrum JSON", type=["json"])

if uploaded_file is not None:
    data = json.load(uploaded_file)
    xcal, sample_num = find_xcal_in_json(data)
    
    if xcal is None:
        st.error("Could not find 'xcal' data in the uploaded JSON.")
    else:
        # st.success(f"Data successfully extracted! Sample Number: {sample_num if sample_num else 'N/A'}")
        
        existing_peaks, existing_result = get_existing_result(sample_num)
        
        col1, col2 = st.columns([1, 2], gap="large")
        
        if existing_result:
            st.info("✅ Found existing result in database for this sample! Showing cached results.")
            
            with col1:
                st.subheader("📈 Extracted Peaks")
                st.dataframe(existing_peaks, width='stretch')
            
            with col2:
                show_results(existing_result)
        else:
            with st.spinner("Processing Spectral Data..."):
                extracted_peaks = process_xcal(xcal)
            
            with col1:
                st.subheader("📈 Extracted Peaks")
                st.dataframe(extracted_peaks, width='stretch')
                
            with col2:
                st.write("### Determining Drug Identity with LLM (RAG)")
                with st.spinner("Calling LLM..."):
                    try:
                        result = generate_llm_analysis(extracted_peaks, kb_data)
                        save_result(sample_num, extracted_peaks, result)
                        show_results(result)
                    except Exception as e:
                        st.error(f"Error communicating with LLM: {e}")
