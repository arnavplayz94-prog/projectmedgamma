"""
MedGamma Backend Server with Gemini AI Integration
===================================================
Flask wrapper for the bby.ipynb notebook with Google Gemini API for AI-generated clinical notes.

This backend:
1. Processes wearable data to create structured clinical context
2. Uses Gemini API to generate SOAP notes, HPI, and chatbot responses
3. Enforces clinical safety guardrails (no diagnoses, probabilistic language)
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: google-generativeai not installed. Run: pip install google-generativeai")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ============================================================================
# GEMINI CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Initialize Gemini client
gemini_model = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        print("[OK] Gemini API initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini: {e}")
        gemini_model = None
elif not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY not set. Set it with: $env:GEMINI_API_KEY = 'your-key'")

# Clinical safety system prompt
CLINICAL_SYSTEM_PROMPT = """You are a clinical AI assistant for the MedGamma wearable health monitoring system.

MANDATORY SAFETY RULES - YOU MUST FOLLOW THESE:
1. NEVER provide a definitive diagnosis
2. NEVER prescribe medications or specific treatments
3. ALWAYS use probabilistic language: "may suggest", "could indicate", "appears consistent with"
4. ALWAYS frame findings as "considerations" not conclusions
5. ALWAYS remind that clinical correlation is required
6. NEVER claim certainty about any medical condition

You analyze structured wearable sensor data (heart rate, EDA, temperature, SpO2, etc.) and provide:
- Objective summaries of the data
- Possible considerations based on patterns
- Areas that may warrant clinical attention

You are NOT a replacement for clinical judgment. All outputs require review by a licensed clinician."""

# ============================================================================
# STRUCTURED CONTEXT - Output from notebook processing
# ============================================================================

def get_structured_context(entity_id: str) -> Dict[str, Any]:
    """
    Get structured clinical context for a subject.
    This simulates what the notebook would produce from wearable data.
    """
    np.random.seed(hash(entity_id) % 2**32)
    
    # Current vitals (factual data from wearables)
    current_vitals = {
        "hr_mean": float(np.random.uniform(65, 95)),
        "hr_variability": float(np.random.uniform(30, 60)),
        "eda_mean": float(np.random.uniform(0.1, 0.5)),
        "temp": float(np.random.uniform(36.2, 37.5)),
        "spo2": float(np.random.uniform(94, 99)),
        "resp_rate": float(np.random.uniform(12, 20)),
        "bp_systolic": float(np.random.uniform(110, 145)),
        "bp_diastolic": float(np.random.uniform(70, 95)),
    }
    
    # Trend descriptors (factual, non-interpretive)
    trends = {
        "hr_trend": np.random.choice(["stable", "increasing", "decreasing"]),
        "eda_trend": np.random.choice(["stable", "elevated", "decreasing"]),
        "temp_trend": np.random.choice(["stable", "elevated", "decreasing"]),
    }
    
    # Significant deviations from baseline
    deviations = []
    if current_vitals["hr_mean"] > 90:
        deviations.append({
            "signal": "hr_mean",
            "deviation_pct": float((current_vitals["hr_mean"] - 75) / 75 * 100),
            "direction": "above_baseline"
        })
    if current_vitals["eda_mean"] > 0.4:
        deviations.append({
            "signal": "eda_mean",
            "deviation_pct": float((current_vitals["eda_mean"] - 0.25) / 0.25 * 100),
            "direction": "above_baseline"
        })
    
    # Time context
    time_context = {
        "data_start": (datetime.now() - timedelta(days=7)).isoformat(),
        "data_end": datetime.now().isoformat(),
        "measurement_count": int(np.random.randint(150, 300)),
    }
    
    return {
        "entity_id": entity_id,
        "current_vitals": current_vitals,
        "trends": trends,
        "deviations": deviations,
        "time_context": time_context,
        "processed_at": datetime.now().isoformat()
    }


# Sample subjects
MOCK_SUBJECTS = [
    {"id": "S01", "name": "John Smith", "age": 45, "condition": "Post-cardiac surgery"},
    {"id": "S02", "name": "Maria Garcia", "age": 62, "condition": "Hypertension monitoring"},
    {"id": "S03", "name": "James Wilson", "age": 38, "condition": "Stress assessment"},
    {"id": "S04", "name": "Emily Chen", "age": 55, "condition": "Cardiac rehabilitation"},
    {"id": "S05", "name": "Michael Brown", "age": 70, "condition": "Heart failure monitoring"},
    {"id": "PT-1234", "name": "John Smith", "age": 62, "condition": "Cardiac monitoring"},
    {"id": "PT-5678", "name": "Maria Garcia", "age": 45, "condition": "Respiratory observation"},
    {"id": "PT-9012", "name": "James Johnson", "age": 55, "condition": "Post-op recovery"},
    {"id": "PT-3456", "name": "Sarah Williams", "age": 38, "condition": "Neuro observation"},
]


# ============================================================================
# GEMINI-POWERED GENERATION
# ============================================================================

def generate_ai_clinical_notes_gemini(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI clinical documentation using Gemini API.
    """
    entity_id = context.get("entity_id", "Unknown")
    vitals = context.get("current_vitals", {})
    trends = context.get("trends", {})
    deviations = context.get("deviations", [])
    time_context = context.get("time_context", {})
    
    # Find patient info
    patient_info = next((p for p in MOCK_SUBJECTS if p["id"] == entity_id), None)
    patient_name = patient_info.get("name", entity_id) if patient_info else entity_id
    patient_condition = patient_info.get("condition", "Unknown") if patient_info else "Unknown"
    patient_age = patient_info.get("age", "Unknown") if patient_info else "Unknown"
    
    # Build the prompt for Gemini
    prompt = f"""Generate clinical documentation for the following patient based on wearable monitoring data.

PATIENT INFORMATION:
- ID: {entity_id}
- Name: {patient_name}
- Age: {patient_age}
- Primary Condition: {patient_condition}

CURRENT VITAL SIGNS (from wearable sensors):
- Heart Rate: {vitals.get('hr_mean', 0):.1f} bpm (Trend: {trends.get('hr_trend', 'stable')})
- Heart Rate Variability: {vitals.get('hr_variability', 0):.1f} ms
- SpO2: {vitals.get('spo2', 0):.1f}%
- Temperature: {vitals.get('temp', 0):.2f}°C
- Respiratory Rate: {vitals.get('resp_rate', 0):.1f}/min
- Blood Pressure: {vitals.get('bp_systolic', 0):.0f}/{vitals.get('bp_diastolic', 0):.0f} mmHg
- Electrodermal Activity (EDA): {vitals.get('eda_mean', 0):.3f} µS (Trend: {trends.get('eda_trend', 'stable')})

MONITORING PERIOD:
- From: {time_context.get('data_start', 'N/A')[:10]}
- To: {time_context.get('data_end', 'N/A')[:10]}
- Data Points: {time_context.get('measurement_count', 0)}

DETECTED DEVIATIONS FROM BASELINE:
{json.dumps(deviations, indent=2) if deviations else "None detected"}

Generate the following sections. Remember to use probabilistic language and NEVER diagnose:

1. SOAP NOTE:
   - Subjective: What the monitoring data suggests about the patient's state
   - Objective: Factual vital sign readings and trends
   - Assessment: Possible considerations based on the data (NOT diagnoses)
   - Plan: Suggested monitoring or follow-up actions

2. HPI (History of Present Illness): Brief narrative of the monitoring period

3. PATIENT SUMMARY: Concise overview suitable for handoff

4. CLINICAL IMPRESSION: Possible considerations (list format, 3-5 items)

Format your response as JSON with keys: subjective, objective, assessment, plan, hpi, patient_summary, clinical_impression (as a list)"""

    if gemini_model:
        try:
            # Call Gemini API
            response = gemini_model.generate_content(
                [CLINICAL_SYSTEM_PROMPT, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent clinical output
                    max_output_tokens=2000,
                )
            )
            
            # Parse response
            response_text = response.text
            
            # Try to extract JSON from response
            try:
                # Handle markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                parsed = json.loads(response_text)
                
                return {
                    "soap_note": {
                        "subjective": parsed.get("subjective") or parsed.get("soap", {}).get("subjective", "Unable to generate"),
                        "objective": parsed.get("objective") or parsed.get("soap", {}).get("objective", "Unable to generate"),
                        "assessment": parsed.get("assessment") or parsed.get("soap", {}).get("assessment", "Unable to generate"),
                        "plan": parsed.get("plan") or parsed.get("soap", {}).get("plan", "Unable to generate")
                    },
                    "hpi": parsed.get("hpi", "Unable to generate"),
                    "assessment_summary": parsed.get("assessment_summary") or parsed.get("assessment", "Unable to generate"),
                    "patient_summary": parsed.get("patient_summary", "Unable to generate"),
                    "clinical_impression": {
                        "differentials": parsed.get("clinical_impression", []),
                        "narrative": "PRELIMINARY CLINICAL IMPRESSION (AI-Generated):\n\n" + 
                                   "\n".join([f"• {item}" for item in parsed.get("clinical_impression", [])])
                    },
                    "disclaimer": "This is an AI-generated suggestion and must be reviewed, edited, or rejected by a licensed clinician.",
                    "label": "AI-Suggested (Review Required)",
                    "generated_at": datetime.now().isoformat(),
                    "model": "gemini-2.5-flash-lite"
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw text
                return {
                    "soap_note": {
                        "subjective": response_text[:500] if len(response_text) > 500 else response_text,
                        "objective": "See subjective section",
                        "assessment": "See subjective section",
                        "plan": "See subjective section"
                    },
                    "hpi": response_text,
                    "assessment_summary": response_text,
                    "patient_summary": response_text,
                    "clinical_impression": {
                        "differentials": ["Raw AI response - structured parsing failed"],
                        "narrative": response_text
                    },
                    "disclaimer": "This is an AI-generated suggestion and must be reviewed, edited, or rejected by a licensed clinician.",
                    "label": "AI-Suggested (Review Required)",
                    "generated_at": datetime.now().isoformat(),
                    "model": "gemini-2.5-flash-lite"
                }
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fall back to rule-based generation
            return generate_ai_clinical_notes_fallback(context)
    else:
        # No Gemini available, use fallback
        return generate_ai_clinical_notes_fallback(context)


def generate_ai_clinical_notes_fallback(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback rule-based generation when Gemini is unavailable.
    """
    entity_id = context.get("entity_id", "Unknown")
    vitals = context.get("current_vitals", {})
    trends = context.get("trends", {})
    deviations = context.get("deviations", [])
    
    subjective = "Patient data obtained from continuous wearable monitoring. "
    subjective += "No subjective complaints documented during this monitoring period."
    
    objective = f"Vitals (wearable-derived): "
    objective += f"HR {vitals.get('hr_mean', 0):.0f} bpm ({trends.get('hr_trend', 'stable')}), "
    objective += f"SpO2 {vitals.get('spo2', 0):.0f}%, "
    objective += f"Temp {vitals.get('temp', 0):.1f}°C, "
    objective += f"RR {vitals.get('resp_rate', 0):.0f}/min, "
    objective += f"BP {vitals.get('bp_systolic', 0):.0f}/{vitals.get('bp_diastolic', 0):.0f} mmHg. "
    objective += f"EDA {vitals.get('eda_mean', 0):.2f} µS ({trends.get('eda_trend', 'stable')})."
    
    assessment = "Based on wearable data analysis: "
    if deviations:
        assessment += "Observed deviations from baseline may warrant further evaluation. "
    else:
        assessment += "Vital parameters appear within expected ranges based on available data. "
    assessment += "Clinical correlation recommended."
    
    plan = "1. Continue wearable monitoring as indicated. "
    plan += "2. Review trends at next clinical encounter. "
    if deviations:
        plan += "3. Consider evaluation if deviations persist or worsen. "
    
    hpi = f"This is a continuous monitoring report for patient {entity_id}. "
    hpi += f"Data collected over {context.get('time_context', {}).get('measurement_count', 0)} measurements."
    
    patient_summary = f"Patient {entity_id} - Wearable Monitoring Summary\n"
    patient_summary += f"Status: {'Deviations noted' if deviations else 'Within expected ranges'}"
    
    differentials = ["Normal physiological variation", "Activity-related changes"]
    if vitals.get("hr_mean", 0) > 90:
        differentials.insert(0, "Possible physiological stress response")
    
    return {
        "soap_note": {
            "subjective": subjective,
            "objective": objective,
            "assessment": assessment,
            "plan": plan
        },
        "hpi": hpi,
        "assessment_summary": assessment,
        "patient_summary": patient_summary,
        "clinical_impression": {
            "differentials": differentials,
            "narrative": "PRELIMINARY CLINICAL IMPRESSION (Rule-Based Fallback):\n" + "\n".join([f"• {d}" for d in differentials])
        },
        "disclaimer": "This is an AI-generated suggestion and must be reviewed, edited, or rejected by a licensed clinician.",
        "label": "AI-Suggested (Review Required)",
        "generated_at": datetime.now().isoformat(),
        "model": "fallback-rule-based"
    }


def generate_chat_response_gemini(message: str, context: Dict, patient_name: str, 
                                   condition: str) -> str:
    """
    Generate a clinical chat response using Gemini API.
    """
    vitals = context.get("current_vitals", {})
    trends = context.get("trends", {})
    deviations = context.get("deviations", [])
    
    prompt = f"""A clinician is asking about a patient. Answer their question based on the monitoring data.

PATIENT: {patient_name}
CONDITION: {condition}

CURRENT VITALS:
- Heart Rate: {vitals.get('hr_mean', 0):.1f} bpm (Trend: {trends.get('hr_trend', 'stable')})
- SpO2: {vitals.get('spo2', 0):.1f}%
- Temperature: {vitals.get('temp', 0):.2f}°C
- Respiratory Rate: {vitals.get('resp_rate', 0):.1f}/min
- Blood Pressure: {vitals.get('bp_systolic', 0):.0f}/{vitals.get('bp_diastolic', 0):.0f} mmHg
- EDA: {vitals.get('eda_mean', 0):.3f} µS (Trend: {trends.get('eda_trend', 'stable')})

DEVIATIONS FROM BASELINE:
{json.dumps(deviations, indent=2) if deviations else "None detected"}

CLINICIAN'S QUESTION: {message}

Provide a helpful, concise response. Remember:
- Use probabilistic language
- Do NOT diagnose
- Reference specific values from the data
- End with a reminder that clinical judgment is required"""

    if gemini_model:
        try:
            response = gemini_model.generate_content(
                [CLINICAL_SYSTEM_PROMPT, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=800,
                )
            )
            
            response_text = response.text.strip()
            
            # Ensure disclaimer is present
            if "clinical judgment" not in response_text.lower():
                response_text += "\n\n---\n*AI-assisted insight. Clinical judgment required.*"
            
            return response_text
            
        except Exception as e:
            print(f"Gemini chat error: {e}")
            return f"DEBUG ERROR: {str(e)}"
            # return generate_chat_response_fallback(message, context, patient_name, condition, vitals, trends, deviations)
    else:
        return generate_chat_response_fallback(message, context, patient_name, condition, vitals, trends, deviations)


def generate_chat_response_fallback(message: str, context: Dict, patient_name: str, 
                                     condition: str, vitals: Dict, trends: Dict, 
                                     deviations: List) -> str:
    """
    Fallback rule-based chat response when Gemini is unavailable.
    """
    message_lower = message.lower()
    response_parts = []
    
    if any(word in message_lower for word in ["vital", "vitals", "current", "status"]):
        response_parts.append(f"**Current Vital Signs for {patient_name}:**")
        response_parts.append(f"• Heart Rate: {vitals.get('hr_mean', 0):.0f} bpm (Trend: {trends.get('hr_trend', 'stable')})")
        response_parts.append(f"• SpO2: {vitals.get('spo2', 0):.0f}%")
        response_parts.append(f"• Temperature: {vitals.get('temp', 0):.1f}°C")
        response_parts.append(f"• Blood Pressure: {vitals.get('bp_systolic', 0):.0f}/{vitals.get('bp_diastolic', 0):.0f} mmHg")
    elif any(word in message_lower for word in ["trend", "concern", "worry", "problem"]):
        response_parts.append(f"**Analysis for {patient_name}:**")
        if deviations:
            response_parts.append("Observed deviations that may warrant attention:")
            for dev in deviations:
                signal = dev["signal"].replace("_", " ").title()
                response_parts.append(f"• {signal}: {dev['deviation_pct']:.1f}% {dev['direction'].replace('_', ' ')}")
        else:
            response_parts.append("No significant deviations from baseline detected.")
    else:
        response_parts.append(f"**{patient_name} ({condition}):**")
        response_parts.append(f"Current HR: {vitals.get('hr_mean', 0):.0f} bpm, SpO2: {vitals.get('spo2', 0):.0f}%")
        if deviations:
            response_parts.append("Note: Some parameters show deviation from baseline.")
    
    response_parts.append("\n---\n*AI-assisted insight (fallback mode). Clinical judgment required.*")
    
    return "\n".join(response_parts)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "MedGamma Clinical AI Backend",
        "version": "2.0.0",
        "gemini_available": gemini_model is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get list of all available subjects."""
    return jsonify({
        "subjects": MOCK_SUBJECTS,
        "count": len(MOCK_SUBJECTS)
    })


@app.route('/api/context/<entity_id>', methods=['GET'])
def get_context(entity_id: str):
    """Get structured clinical context for a subject."""
    context = get_structured_context(entity_id)
    return jsonify(context)


@app.route('/api/generate-notes', methods=['POST'])
def generate_notes():
    """
    Generate AI clinical documentation from context using Gemini.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if "context" in data:
            context = data["context"]
        elif "entity_id" in data:
            context = get_structured_context(data["entity_id"])
        else:
            return jsonify({"error": "Must provide entity_id or context"}), 400
        
        # Generate AI clinical notes (uses Gemini if available)
        notes = generate_ai_clinical_notes_gemini(context)
        
        return jsonify({
            "success": True,
            "context": context,
            "notes": notes
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def clinical_chat():
    """
    Clinical AI Assistant chat endpoint using Gemini.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        message = data.get("message", "").strip()
        entity_id = data.get("entity_id")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        if not entity_id:
            return jsonify({
                "response": "Please select a patient first. I need patient-specific data to provide meaningful clinical insights.\n\n*AI-assisted insight. Clinical judgment required.*",
                "type": "error"
            })
        
        # Get patient context
        context = get_structured_context(entity_id)
        
        # Find patient info
        patient_info = next((p for p in MOCK_SUBJECTS if p["id"] == entity_id), None)
        patient_name = patient_info.get("name", entity_id) if patient_info else entity_id
        patient_condition = patient_info.get("condition", "Unknown") if patient_info else "Unknown"
        
        # Generate response (uses Gemini if available)
        response = generate_chat_response_gemini(message, context, patient_name, patient_condition)
        
        return jsonify({
            "response": response,
            "type": "success",
            "entity_id": entity_id,
            "model": "gemini-2.5-flash-lite" if gemini_model else "fallback",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MedGamma Clinical AI Backend (Gemini-Powered)")
    print("=" * 60)
    print(f"Gemini API: {'[OK] Available' if gemini_model else '[X] Not configured'}")
    if not gemini_model:
        print(f"  Set API key: $env:GEMINI_API_KEY = 'your-key'")
    print(f"Starting server at http://localhost:5000")
    print(f"API Endpoints:")
    print(f"  GET  /api/health           - Health check")
    print(f"  GET  /api/subjects         - List all subjects")
    print(f"  GET  /api/context/<id>     - Get structured context")
    print(f"  POST /api/generate-notes   - Generate AI clinical notes")
    print(f"  POST /api/chat             - Clinical AI Assistant chat")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
