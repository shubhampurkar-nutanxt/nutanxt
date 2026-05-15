import streamlit as st
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import sqlite3
import pymysql

st.set_page_config(page_title="Expert review AI Agent", page_icon="🔬", layout="wide")
######################### rds db for scan fetch

def connect_db():
    con = pymysql.connect(host='oak-centralized-db.clyejxhqj10e.us-west-1.rds.amazonaws.com',user='shubham', password = "nutanxt@5499",db="oak_db")
    return con

def db_execute_query(query, data_feed=None, fetch=False):
    sqlClient = connect_db()
    sqlObj = sqlClient.cursor(pymysql.cursors.DictCursor)
    try:
        if data_feed:
            sqlObj.execute(query, data_feed)
        else:
            sqlObj.execute(query)
        if fetch:
            result = sqlObj.fetchall()
            return result
        sqlClient.commit()
    except pymysql.err.IntegrityError as e:
        sqlClient.rollback()
        return str(e)
    finally:
        sqlClient.close()
    return sqlObj.rowcount


def get_scan_id(id):
    product_name= "IDMaterial"
    query = """SELECT st.scan_id,st.xcal,
idt.sample_number FROM scan_tb st LEFT JOIN identification_tb idt  ON st.scan_id = idt.scan_id where st.product_name = %s
AND st.scan_id in %s;
    """
    data_feed = (product_name,id)
    sc_dt = db_execute_query(query, data_feed=data_feed, fetch=True)
    return sc_dt

##########################################

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
        peaks = json.loads(row[0])
        result = json.loads(row[1])
        if isinstance(result, str):
            result = json.loads(result)
        return peaks, result
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
    confidence_label = (result.get('confidence') or '').lower()
    conf_color = "green" if confidence_label == 'high' else "orange" if confidence_label == 'medium' else "red"
    confidence_score = result.get('confidence_score', None)

    predicted = (result.get('predicted_drug') or '').strip()
    if not predicted or predicted.lower() in {"none", "null", "n/a"} or confidence_label == "low":
        predicted = "Inconclusive"
        drug_color = "#b45309"
    else:
        drug_color = "#1E88E5"

    st.markdown(f"### **Predicted Drug:** <span style='color:{drug_color};'>{predicted}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence Level:** <span style='color:{conf_color}; font-weight:bold;'>{(result.get('confidence') or 'N/A').title()}</span>", unsafe_allow_html=True)
    if confidence_score is not None:
        try:
            score = int(confidence_score)
            st.markdown(f"**Confidence Score:** {score}%")
            st.progress(score / 100)
        except (ValueError, TypeError):
            pass
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
try:
    from oak_identification import (
        flourencense_identify,
        preprocess,
        structure_score,
        compute_score,
        get_pass_fail,
    )
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

# ---------------------------------------------------------------------------
# Gemini API keys — loaded from Streamlit secrets for cloud deployment.
# The app rotates to the next key on a 429/quota error.
#
# In Streamlit Cloud, configure secrets as either:
#   GEMINI_API_KEYS = ["AIza...key1", "AIza...key2", "AIza...key3"]
# or a single key:
#   GEMINI_API_KEY = "AIza...key1"
# For local dev, GEMINI_API_KEYS env var (comma-separated) also works.
# ---------------------------------------------------------------------------
def _load_gemini_keys():
    try:
        if "GEMINI_API_KEYS" in st.secrets:
            val = st.secrets["GEMINI_API_KEYS"]
            if isinstance(val, str):
                return [k.strip() for k in val.split(",") if k.strip()]
            return [str(k).strip() for k in val if str(k).strip()]
        if "GEMINI_API_KEY" in st.secrets:
            return [str(st.secrets["GEMINI_API_KEY"]).strip()]
    except Exception:
        pass
    env_keys = os.environ.get("GEMINI_API_KEYS", "").strip()
    if env_keys:
        return [k.strip() for k in env_keys.split(",") if k.strip()]
    env_single = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_single:
        return [env_single]
    return []

GEMINI_API_KEYS = _load_gemini_keys()

_KEY_INDEX = 0  # current key pointer; advances on rate-limit errors


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Gemini rate-limit / quota errors across SDK versions."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg:
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, "429"):
        return True
    return False


def _call_gemini(prompt: str, generation_config: dict | None = None) -> str:
    """Call Gemini with automatic key rotation on rate-limit errors.
    Each key gets one attempt; raises the last error if all keys fail.
    Returns the raw response.text."""
    global _KEY_INDEX
    if not GEMINI_API_KEYS:
        raise RuntimeError("No Gemini API keys configured.")
    cfg = generation_config or {'response_mime_type': 'application/json'}

    last_err = None
    for _ in range(len(GEMINI_API_KEYS)):
        key = GEMINI_API_KEYS[_KEY_INDEX]
        try:
            if HAS_NEW_GENAI is True:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=cfg,
                )
                return response.text
            elif HAS_NEW_GENAI is False:
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    prompt,
                    generation_config=cfg,
                )
                return response.text
            raise RuntimeError("Generative AI module not available.")
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e):
                next_idx = (_KEY_INDEX + 1) % len(GEMINI_API_KEYS)
                st.warning(
                    f"Gemini key #{_KEY_INDEX + 1} rate-limited. "
                    f"Rotating to key #{next_idx + 1}/{len(GEMINI_API_KEYS)}…"
                )
                _KEY_INDEX = next_idx
                continue
            raise
    raise last_err if last_err else RuntimeError("All Gemini keys failed.")


PROMINENCE = 0.03

# ---------------------------------------------------------------------------
# Middleware: shortlist KB drugs to reduce token usage
# ---------------------------------------------------------------------------
KB_TO_DB_NAME = {
    "Methamphetamine": "Methamphetamine",
    "Amphetamine": "Amphetamine",
    "Fentanyl": "Fentanyl",
    "Cocaine": "Cocaine",
    "CocaineHCl": "Cocainehcl",
    "Heroin": "Heroin",
    "Oxycodone": "Oxycodone",
    "MDMA": "Mdma",
    "Codeine": "Codeine",
    "Alprazolam": "Alprazolam",
    "Xylazine": "Xylazine",
    "Morphine": "Morphine",
    "Methylsulfonylmethane": "Methylsulfonylmethane",
}

def _score_one_drug(xcal, db_name, is_fluorescent):
    out = {"structure": 0.0, "compute": 0.0, "peaks_found": [],
           "pass_fail_score": 0.0, "pass_fail_result": "fail"}
    try:
        ss = structure_score(xcal, db_name, flourense=is_fluorescent)
        if isinstance(ss, dict):
            out["structure"] = float(ss.get("percent", 0) or 0)
    except Exception:
        pass
    try:
        cs = compute_score(xcal, db_name, flourense=is_fluorescent)
        if isinstance(cs, dict):
            out["compute"] = float(cs.get("score", 0) or 0)
            peaks_found = cs.get("peaks_found", []) or []
            out["peaks_found"] = [int(p) for p in peaks_found]
    except Exception:
        pass
    try:
        pf = get_pass_fail(xcal, db_name, flourense=is_fluorescent)
        if isinstance(pf, dict):
            out["pass_fail_score"] = float(pf.get("score", 0) or 0)
            out["pass_fail_result"] = pf.get("drugname", "xyz")
    except Exception:
        pass
    return out


def shortlist_kb(xcal, kb_data, is_fluorescent):
    """Score every drug, return (filtered_kb, score_rows, selected_names, strategy).

    Strategy: top 3 by compute_score + top 2 by structure_score (deduped).
    """
    rows = []
    for entry in kb_data:
        kb_name = entry.get("drug_name")
        db_name = KB_TO_DB_NAME.get(kb_name)
        if not db_name:
            rows.append({"kb_name": kb_name, "db_name": None,
                         "structure": 0.0, "compute": 0.0,
                         "pass_fail_score": 0.0, "pass_fail_result": "skip"})
            continue
        scores = _score_one_drug(xcal, db_name, is_fluorescent)
        rows.append({"kb_name": kb_name, "db_name": db_name, **scores})

    top_comp = sorted(rows, key=lambda r: r["compute"], reverse=True)[:3]
    top_struct = sorted(rows, key=lambda r: r["structure"], reverse=True)[:2]
    seen, selected = set(), []
    for r in top_comp + top_struct:
        if r["kb_name"] and r["kb_name"] not in seen:
            seen.add(r["kb_name"])
            selected.append(r["kb_name"])
    strategy = "top 3 compute + top 2 structure"

    filtered_kb = [d for d in kb_data if d.get("drug_name") in selected]
    return filtered_kb, rows, selected, strategy


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
            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                if k.lower() == 'sample_number':
                    sample_num = v
                elif sample_num is None and ('number' in k.lower() or 'sample' in k.lower() or 'id' in k.lower()):
                    sample_num = v
        if xcal is not None:
            return xcal, sample_num

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

def generate_llm_analysis(sample_peaks, kb, drug_scores=None):
    drug_scores = drug_scores or []
    prompt = f"""You are a forensic Raman spectroscopy expert.

Your task is to analyze a sample spectrum and determine the most likely drug based on the provided knowledge base.

========================
INPUT SAMPLE PEAKS
========================
{json.dumps(sample_peaks, indent=2)}

========================
PRE-SCREENING SCORES (for the shortlisted drugs)
========================
These scores were computed by a deterministic spectral matcher BEFORE you. Use them
as strong evidence alongside the peak/KB analysis:
- "structure_score" : structural similarity of the sample to the reference (0–100, higher is better).
- "compute_score"   : peak-matching score — how well sample peaks align with the reference's golden peaks (0–100, higher is better).
- "peaks_found"     : the sample peak positions (cm⁻¹) that matched this drug's golden peaks.

{json.dumps(drug_scores, indent=2)}

========================
KNOWLEDGE BASE (only the shortlisted candidates)
========================
{json.dumps(kb, indent=2)}

========================
INSTRUCTIONS
========================

1. Compare sample peaks with each drug in the knowledge base.
2. Combine evidence from BOTH sources:
   - The KB rules (peak matching within tolerance, critical > high > medium > low importance,
     key peak ranges, signature rules, confusion analysis).
   - The pre-screening "structure_score" and "compute_score" (peak score) above 40
     a drug with clearly higher structure AND compute scores than the others is the
     strongest candidate; a drug with low compute_score and few peaks_found is unlikely
     even if a couple of generic peaks coincide.
3. Identify:
   - Best matching drug.
   - Alternative possible drugs (if any).
4. Clearly explain reasoning:
   - Which peaks matched (cite peaks_found).
   - Which critical peaks are present/missing.
   - How the structure and compute scores influenced your decision.
   - Why the selected drug is correct vs the runner-ups.
5. If confidence is low (no drug stands out on either KB rules or scores, or all
   pre-screening structure_score and compute_score are ~0), you MUST set
   "predicted_drug" to the exact string "Inconclusive" (not empty, not null) and
   set "confidence" to "low".

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
def generate_llm_analysis2(sample_peaks, kb, drug_scores=None):
    drug_scores = drug_scores or []

    prompt = f"""
You are a forensic Raman spectroscopy expert.

Your task is to analyze a sample Raman spectrum and determine whether it matches one
of the shortlisted drugs from the provided knowledge base.

This is an OPEN-SET identification system.
The sample may belong to a substance that is NOT present in the knowledge base.
Do NOT force a drug prediction just because candidates are provided.

========================
INPUT SAMPLE PEAKS
========================
{json.dumps(sample_peaks, indent=2)}

========================
PRE-SCREENING SCORES
========================
These scores were computed by a deterministic spectral matcher before you.

Definitions:
- structure_score: overall structural similarity to the reference spectrum, 0–100.
- compute_score: peak-matching score against golden/reference peaks, 0–100.
- peaks_found: sample peaks that matched this drug's golden/reference peaks.

Use these scores as strong evidence, but do not rely only on the highest score.
A drug with the highest score among weak candidates is NOT automatically a valid detection.

{json.dumps(drug_scores, indent=2)}

========================
KNOWLEDGE BASE
========================
The KB contains:
- golden_signature
- critical_peaks
- key_peaks
- peak_patterns
- shift_model
- overlap_handling
- confusion_analysis
- decision_strategy

Use all of these fields for reasoning.

{json.dumps(kb, indent=2)}

========================
CORE DECISION PRINCIPLE
========================

Your decision must combine BOTH:
1. Deterministic scores: structure_score, compute_score, peaks_found.
2. Knowledge-base evidence: critical peaks, high-importance peaks, required peak patterns,
   overlap/confusion rules, and missing diagnostic peaks.

You must classify the sample as:
- a specific drug only when the evidence is strong enough, OR
- "Inconclusive" when the sample may be outside the known KB.

========================
OPEN-SET REJECTION RULES
========================

Return "Inconclusive" when the shortlisted candidates are weak or ambiguous.

A candidate is weak if:
- compute_score < 40
- structure_score is very low
- only generic/common peaks match
- required critical/signature peaks are missing
- the matched peaks also overlap with multiple other drugs
- the KB decision_strategy is not satisfied
- the candidate is only the "closest available" drug but not a strong match

IMPORTANT:
If all candidates have compute_score < 40, you should normally return "Inconclusive".

However, compute_score < 40 can still be accepted ONLY if there is very strong KB evidence.
Use the LOW-SCORE OVERRIDE rules below.

========================
LOW-SCORE OVERRIDE RULES
========================

If the best candidate has compute_score < 40, you may still predict that drug ONLY when
ALL of the following conditions are satisfied:

1. The candidate has a clear KB-supported signature match:
   - either the required peak_pattern is satisfied,
   - or at least one critical peak plus two high-importance/key peaks are present,
   - or the KB explicitly says one critical peak is sufficient only when most golden
     signature peaks also match.

2. The matched peaks are diagnostic, not generic:
   - Do not rely only on common Raman peaks such as ~620, ~1000, ~1450, or ~1600
     unless the KB says their combination is drug-specific.
   - If overlap_handling says a peak is ambiguous, reduce confidence.

3. Missing critical peaks must be handled strictly:
   - If the KB says a peak is essential, missing that peak should usually prevent prediction.
   - If the KB says missing one critical peak reduces confidence but does not reject,
     you may continue only if other strong signature evidence is present.
   - Do not explain away missing critical peaks using fluorescence, tablet effects,
     or SERS suppression unless the input explicitly indicates those effects.

4. The best candidate must clearly beat runner-ups:
   - It must have stronger KB evidence than all alternatives.
   - A small score difference is not enough.
   - If another candidate has similar scores or similar peak overlap, return "Inconclusive"
     or include it as an alternative with lower confidence.

5. Confidence cannot be "high" when compute_score < 40.
   - If accepted under low-score override, confidence must be "medium" at maximum.
   - If evidence is present but not enough, return "Inconclusive" with confidence "low".

========================
CONFIDENCE RULES
========================

High confidence:
- compute_score >= 50
- structure_score is reasonably strong
- required critical/signature pattern is satisfied
- missing critical peaks do not contradict the KB
- candidate clearly beats alternatives

Medium confidence:
- compute_score >= 40 with good KB support, OR
- compute_score < 40 but LOW-SCORE OVERRIDE is fully satisfied
- at least one critical peak and multiple high/key peaks match
- no major contradiction exists

Low confidence:
- weak scores
- missing essential peaks
- only generic peaks match
- conflicting candidates exist
- sample may be outside the KB

If confidence is low, predicted_drug MUST be "Inconclusive".

========================
SPECIAL RULES FOR FALSE POSITIVE PREVENTION
========================

1. Do not select a drug only because it has the highest compute_score.
2. Do not select a drug only because a few peaks match.
3. Do not select a drug when matched peaks are mostly generic or shared across drug families.
4. Do not use KB notes like "fluorescence may affect Raman" or "SERS may suppress carbonyl"
   as an excuse unless the input sample metadata explicitly confirms fluorescence/SERS.
5. If the sample could be a non-KB material, cutting agent, tablet excipient, or unknown
   pharmaceutical, return "Inconclusive".
6. If the best candidate has compute_score < 40 and structure_score < 20, prediction is allowed
   only if LOW-SCORE OVERRIDE is strongly satisfied. Otherwise return "Inconclusive".

========================
ANALYSIS STEPS
========================

For each candidate:
1. List matched peaks from peaks_found.
2. Compare matched peaks against critical_peaks and key_peaks.
3. Check whether required peak_patterns are satisfied.
4. Check missing critical peaks.
5. Check overlap_handling and confusion_analysis.
6. Compare structure_score and compute_score with other candidates.
7. Decide whether this is:
   - strong match
   - weak but possible match
   - rejected candidate

Final decision:
- Select a drug only if it passes score + KB evidence requirements.
- Otherwise return "Inconclusive".

========================
OUTPUT FORMAT
========================

Return STRICT JSON only.

{{
  "predicted_drug": "",
  "confidence": "high/medium/low",
  "decision_type": "strong_match/low_score_override/inconclusive",
  "reasoning": "",
  "matched_peaks": [],
  "missing_critical_peaks": [],
  "kb_evidence_summary": {{
    "critical_peaks_matched": [],
    "high_key_peaks_matched": [],
    "patterns_satisfied": [],
    "patterns_failed": [],
    "generic_or_ambiguous_peaks": [],
    "contradictions": []
  }},
  "score_evidence_summary": {{
    "best_candidate_compute_score": 0,
    "best_candidate_structure_score": 0,
    "score_interpretation": ""
  }},
  "alternative_candidates": []
}}
"""

    def _unwrap(parsed):
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return parsed

    def _normalize_inconclusive(res):
        if not isinstance(res, dict):
            return res
        pred = (res.get("predicted_drug") or "").strip()
        conf = (res.get("confidence") or "").strip().lower()
        if not pred or pred.lower() in {"none", "null", "n/a", "inconclusive"} or conf == "low":
            res["predicted_drug"] = "Inconclusive"
            if not conf:
                res["confidence"] = "low"
        return res
    generation_config = {
        'temperature': 0.0,
        'top_p': 1.0,
        'seed': 42,  # Any integer works as a fixed value
        'response_mime_type': 'application/json'
    }
    try:
        text = _call_gemini(prompt, generation_config)
    except Exception as e:
        return {"error": f"Generative AI call failed: {e}"}
    return _normalize_inconclusive(_unwrap(json.loads(text)))


def process_xcal(xcal):
    # Coerce xcal into a list of floats. This is type normalization — required
    # because RDS returns the spectrum as a JSON-encoded string and a JSON
    # upload may contain string-typed numbers; flourencense_identify calls
    # np.array(x).min() which fails on string dtype.
    if isinstance(xcal, str):
        try:
            parsed_xcal = json.loads(xcal)
            if isinstance(parsed_xcal, list):
                xcal = [float(v) for v in parsed_xcal]
            else:
                xcal = [float(v) for v in xcal.replace('[', '').replace(']', '').replace(',', ' ').split()]
        except Exception:
            xcal = [float(v) for v in xcal.replace('[', '').replace(']', '').replace(',', ' ').split()]
    else:
        xcal = [float(v) for v in xcal]

    # Pass the raw spectrum directly to the detector — no smoothing, no slicing.
    print("type of xcal:", type(xcal))
    print("length of xcal:", len(xcal))
    is_fluorescent = bool(flourencense_identify(xcal) or flourencense_identify(xcal,just_peaks=True))
    print(f"Fluorescence detected: {is_fluorescent}")
    badge = "🟠 Fluorescent" if is_fluorescent else "🟢 Not Fluorescent"
    st.markdown(
        f"**Fluorescence Check:** {badge} "
        f"<span style='color:#64748b;font-size:12px'>· spectrum length {len(xcal)}</span>",
        unsafe_allow_html=True,
    )

    if is_fluorescent:
        offset = 400
        try:
            x_series = pd.Series(xcal, index=np.arange(200, 200+len(xcal)))
            intensity_array = preprocess(x_series, smoothing=True, idx=(400, 1700))
            print("Running fluorescent preprocessing with smoothing and slicing...")
        except Exception as e:
            st.warning(f"Using fallback preprocessing due to: {e}")
            intensity_array = preprocess(xcal)
    else:
        offset = 200
        print("Running non-fluorescent preprocessing...")
        intensity_array = preprocess(xcal)
    
    intensity_array = np.array(intensity_array)
    peaks, properties = find_peaks(intensity_array, prominence=PROMINENCE)
    print(f"Found peaks at indices: {peaks}")
    peak_data = []
    for idx, prom in zip(peaks, properties["prominences"]):
        peak_data.append({
            "position": int(idx + offset),
            "intensity": float(intensity_array[idx]),
            "prominence": float(prom)
        })
    print(f"Extracted {len(peak_data)} peaks with prominence > {PROMINENCE} and peaks {peak_data}")
    return peak_data, xcal, bool(is_fluorescent)

st.title("Expert Review Agent")
st.markdown("Upload a JSON file **or** enter a Scan ID to analyze the spectral signature automatically.")
st.divider()

kb_path = "Updated_Knowledge_base.json"
if os.path.exists(kb_path):
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
else:
    st.error(f"Knowledge base not found at {kb_path}")
    st.stop()

input_method = st.radio("Select Input Method", ["Upload JSON File", "Enter Scan ID"], horizontal=True)

xcal = None
sample_num = None

if input_method == "Upload JSON File":
    uploaded_file = st.file_uploader("Upload Spectrum JSON", type=["json"])
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        xcal, sample_num = find_xcal_in_json(data)
        if xcal is None:
            st.error("Could not find 'xcal' data in the uploaded JSON.")

else:  # Enter Scan ID
    scan_id_input = st.text_input("Enter Scan ID")
    fetch_btn = st.button("Fetch from Database")
    if fetch_btn:
        if not scan_id_input.strip():
            st.warning("Please enter a Scan ID.")
        else:
            with st.spinner("Fetching data from RDS..."):
                try:
                    print(f"Fetching data for Scan ID: {scan_id_input.strip()}")
                    rows = get_scan_id((scan_id_input.strip(),))
                    print(f"data keys {rows[0].keys() if rows else 'No rows returned'}")
                except Exception as e:
                    st.error(f"Database error: {e}")
                    rows = None
            if rows is not None:
                if not rows:
                    st.error(f"No data found for Scan ID: {scan_id_input.strip()}")
                else:
                    row = rows[0]
                    xcal = eval(row.get('xcal'))
                    sample_num = row.get('sample_number')
                    if xcal is None:
                        st.error("No xcal data found for this Scan ID.")
                    else:
                        st.success(f"Data fetched! Sample Number: {sample_num if sample_num else 'N/A'}")

if xcal is not None:
    existing_peaks, existing_result = get_existing_result(sample_num)

    col1, col2 = st.columns([1, 2], gap="large")

    if existing_result:
        print(f"Found existing result for sample {sample_num}")
        with col1:
            st.subheader("📈 Extracted Peaks")
            st.dataframe(existing_peaks, width='stretch')
        with col2:
            show_results(existing_result)
    else:
        with st.spinner("Processing Spectral Data..."):
            print("Processing xcal data...",len(xcal))
            extracted_peaks, xcal_floats, is_fluorescent = process_xcal(xcal)

        with col1:
            st.subheader("📈 Extracted Peaks")
            st.dataframe(extracted_peaks, width='stretch')

        with col2:
            with st.spinner("Pre-screening knowledge base (13 drugs)..."):
                filtered_kb, score_rows, selected_names, strategy = shortlist_kb(
                    xcal_floats, kb_data, is_fluorescent
                )

            # Fluorescence often hides in baseline noise that the detector misses.
            # If we ran in NON-fluorescent mode and every drug scored zero on both
            # structure and compute, retry the middleware as fluorescent — usually
            # the scores jump and the LLM gets useful context.
            if (not is_fluorescent
                    and score_rows
                    and all((r["structure"] + r["compute"]) <= 0.01 for r in score_rows)):
                st.warning(
                    "All drugs scored ~0 in non-fluorescent mode — retrying as fluorescent."
                )
                with st.spinner("Re-scoring with fluorescent preprocessing..."):
                    filtered_kb, score_rows, selected_names, strategy = shortlist_kb(
                        xcal_floats, kb_data, True
                    )
                is_fluorescent = True
                print("filtered_kb after retry:", filtered_kb)
            # Compact score payload for the LLM — only the shortlisted drugs.
            drug_scores_for_llm = [
                {
                    "drug_name": r["kb_name"],
                    "structure_score": round(r["structure"], 2),
                    "compute_score": round(r["compute"], 2),
                    "peaks_found": r.get("peaks_found", []),
                }
                for r in score_rows if r["kb_name"] in selected_names
            ]
            drug_scores_for_llm.sort(
                key=lambda d: (d["structure_score"] + d["compute_score"]),
                reverse=True,
            )

            st.write("### Determining Drug Identity with LLM (RAG)")
            with st.spinner("Calling LLM..."):
                try:
                    result = generate_llm_analysis2(
                        extracted_peaks, filtered_kb, drug_scores=drug_scores_for_llm
                    )
                    save_result(sample_num, extracted_peaks, result)
                    show_results(result)
                except Exception as e:
                    st.error(f"Error communicating with LLM: {e}")
