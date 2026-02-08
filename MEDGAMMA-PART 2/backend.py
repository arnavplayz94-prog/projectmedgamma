"""
MedGamma Backend Server - Synthetic Patient Data + BBY AI Agent
================================================================
Flask backend integrating the advanced BBY clinical AI agent
with the synthetic patient dataset.

ALL AI outputs are derived ONLY from the synthetic dataset.
NO mock, fallback, or hallucinated data is permitted.
"""

import os
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import patient data module
from patient_data import (
    SYNTHETIC_PATIENTS,
    get_patient,
    get_all_patients,
    get_patient_ids,
    validate_patient_exists,
    build_patient_context,
    build_subjective_data,
    build_objective_data,
    patient_summary_text
)

# Import BBY Advanced Clinical AI Agent
import bby

app = Flask(__name__)
CORS(app)

# ============================================================================
# GEMINI API WRAPPER
# ============================================================================

CLINICAL_SYSTEM_PROMPT = """You are a clinical documentation assistant for the MedGamma system.

MANDATORY RULES:
1. Use ONLY the patient data provided - do NOT invent or assume any information
2. Do NOT provide definitive diagnoses - use probabilistic language
3. Do NOT prescribe treatments - suggest clinical considerations only
4. Be uncertainty-aware and traceable to data fields
5. All outputs must be explainable from the input data

FORMATTING:
- Use clear, professional medical terminology
- Structure outputs with clear sections
- Include disclaimers about AI-generated content
"""

def call_gemini(system_prompt: str, user_prompt: str) -> str:
    """
    Call Gemini API for natural language generation.
    Raises RuntimeError if API fails - NO fallback.
    """
    if not gemini_model:
        raise RuntimeError("Gemini API not configured. Set GEMINI_API_KEY environment variable.")
    
    try:
        response = gemini_model.generate_content(
            [system_prompt, user_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
            )
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {str(e)}")


# ============================================================================
# SESSION CONTEXT
# ============================================================================

session_contexts: Dict[str, Dict[str, Any]] = {}


def get_session_context(patient_id: str) -> Dict[str, Any]:
    """Get or create session context for a patient."""
    if patient_id not in session_contexts:
        validate_patient_exists(patient_id)
        session_contexts[patient_id] = {
            "patient_id": patient_id,
            "context": build_patient_context(patient_id),
            "generated_notes": None,
            "chat_history": [],
            "created_at": datetime.now().isoformat()
        }
    return session_contexts[patient_id]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with system status."""
    return jsonify({
        "status": "ok" if gemini_model else "degraded",
        "service": "MedGamma Backend (Synthetic Data)",
        "gemini_available": gemini_model is not None,
        "patients_loaded": len(SYNTHETIC_PATIENTS),
        "patient_ids": get_patient_ids(),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/patients', methods=['GET'])
def list_patients():
    """List all available patients from the synthetic dataset."""
    patients = []
    for pid, data in SYNTHETIC_PATIENTS.items():
        # Build full context to include risk assessment
        context = build_patient_context(pid)
        
        patients.append({
            "patient_id": pid,
            "demographics": data["demographics"],
            "medical_history": data["medical_history"],
            "risk_assessment": context["risk_assessment"],
            "vitals": data["vitals"]
        })
        
    return jsonify({
        "patients": patients,
        "count": len(patients),
        "data_source": "synthetic_dataset"
    })


@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient_details(patient_id: str):
    """Get full patient data from synthetic dataset."""
    try:
        validate_patient_exists(patient_id)
        patient = get_patient(patient_id)
        context = build_patient_context(patient_id)
        return jsonify({
            "patient": patient,
            "context": context,
            "data_source": "synthetic_dataset"
        })
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "error_type": "PATIENT_NOT_FOUND"
        }), 404


@app.route('/api/generate-notes', methods=['POST'])
def generate_notes():
    """
    Generate clinical notes using synthetic patient data + BBY AI Agent.
    Uses the advanced 5-point clinical format from bby.py.
    All outputs are derived from the dataset - no fabrication.
    """
    if not bby.gemini_model:
        return jsonify({
            "error": "Gemini API not configured. Set GEMINI_API_KEY.",
            "error_type": "API_NOT_CONFIGURED"
        }), 500
    
    try:
        data = request.get_json()
        if not data or "patient_id" not in data:
            return jsonify({
                "error": "patient_id is required",
                "error_type": "VALIDATION_ERROR"
            }), 400
        
        patient_id = data["patient_id"]
        
        # Validate patient exists
        validate_patient_exists(patient_id)
        
        # Build structured context from dataset
        context = build_patient_context(patient_id)
        subjective = build_subjective_data(patient_id)
        objective = build_objective_data(patient_id)
        summary_text = patient_summary_text(patient_id)
        
        # Build a recovery summary structure for the BBY agent
        # This bridges patient_data.py format to bby.py expected format
        bby_summary = {
            "entity_id": patient_id,
            "trend": context["risk_assessment"]["level"],  # Map risk level to trend
            "signals_used": ["hr_mean", "bp_systolic", "spo2"],
            "overall": {
                "median_slope": None,  # Not available in synthetic data
                "median_time_to_baseline_min": None,
                "median_excess_auc": None,
                "mean_rebound_rate": None
            },
            "per_signal": {},
            "risk_flags": [{"flag": f, "evidence": "From patient data"} for f in context["risk_assessment"]["factors"]],
            "uncertainty": {"level": "medium", "reasons": ["Using synthetic dataset averages"]},
            "tag_count": 0
        }
        
        # Generate comprehensive clinical note using BBY agent's format
        bby_compact = bby.compact_for_llm(bby_summary)
        summary_json = json.dumps(bby_compact, ensure_ascii=False, indent=2)
        
        # Use BBY agent's USER_TEMPLATE for the advanced 5-point format
        clinical_prompt = bby.USER_TEMPLATE.format(
            patient_context=f"Patient {patient_id}: {context['demographics']['age']}-year-old {context['demographics']['sex']}, "
                           f"BMI {context['bmi']}, presenting with: {', '.join(context['symptoms']) if context['symptoms'] else 'No symptoms reported'}. "
                           f"Medical history: {', '.join(context['medical_history']) if context['medical_history'] else 'None'}. "
                           f"Vitals: HR {context['vitals']['heart_rate_avg']} bpm, BP {context['vitals']['blood_pressure']}, SpO2 {context['vitals']['spo2_percent']}%.",
            summary_json=summary_json
        )
        
        # Call Gemini using BBY agent's llm_gemini function
        clinical_note = bby.llm_gemini(bby.SYSTEM_PROMPT, clinical_prompt)
        
        # Also generate a traditional SOAP note for compatibility
        soap_prompt = f"""
Based ONLY on the following patient data, generate a SOAP note.
Do NOT invent any information not present in the data.

{summary_text}

Generate a SOAP note with these sections:
1. SUBJECTIVE: Patient-reported symptoms and history
2. OBJECTIVE: Vitals, measurements, wearable data
3. ASSESSMENT: Clinical impression based ONLY on the data (use probabilistic language)
4. PLAN: Suggested considerations (NOT prescriptions)

Include a disclaimer that this is AI-generated and requires clinical review.
"""
        
        soap_response = bby.llm_gemini(bby.SYSTEM_PROMPT, soap_prompt)
        
        # Store in session
        session = get_session_context(patient_id)
        session["generated_notes"] = {
            "soap": soap_response,
            "summary": clinical_note,  # BBY agent 5-point format
            "impression": clinical_note
        }
        
        return jsonify({
            "success": True,
            "patient_id": patient_id,
            "notes": {
                "soap_note": {
                    "full_text": soap_response,
                    "subjective_data": subjective,
                    "objective_data": objective
                },
                "patient_summary": clinical_note,  # BBY agent's advanced 5-point format
                "clinical_impression": clinical_note
            },
            "bby_agent": {
                "format": "5-point clinical note",
                "sections": ["Clinical summary", "Risk level", "Questions to ask", "Suggested next step", "Safety note"]
            },
            "context": context,
            "disclaimer": "AI-generated clinical documentation. Must be reviewed, edited, or rejected by a licensed clinician.",
            "label": "AI-Suggested (Review Required)",
            "editable": True,
            "rejectable": True,
            "model": "gemini-2.5-flash-lite via BBY Agent",
            "data_source": "synthetic_dataset",
            "traceable_fields": list(context.keys()),
            "generated_at": datetime.now().isoformat()
        })
        
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "error_type": "PATIENT_NOT_FOUND"
        }), 404
    except RuntimeError as e:
        return jsonify({
            "error": str(e),
            "error_type": "API_ERROR"
        }), 500
    except Exception as e:
        return jsonify({
            "error": str(e),
            "error_type": "PROCESSING_ERROR",
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/chat', methods=['POST'])
def clinical_chat():
    """
    Clinical AI chatbot with patient context.
    Uses BBY agent for AI responses with synthetic dataset.
    """
    if not bby.gemini_model:
        return jsonify({
            "error": "Gemini API not configured. Set GEMINI_API_KEY.",
            "error_type": "API_NOT_CONFIGURED"
        }), 500
    
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({
                "error": "message is required",
                "error_type": "VALIDATION_ERROR"
            }), 400
        
        message = data["message"].strip()
        patient_id = data.get("patient_id")
        
        if not patient_id:
            return jsonify({
                "error": "patient_id is required for context-aware responses",
                "error_type": "VALIDATION_ERROR"
            }), 400
        
        # Validate and get context
        validate_patient_exists(patient_id)
        session = get_session_context(patient_id)
        summary_text = patient_summary_text(patient_id)
        
        # Build chat prompt
        chat_history_text = ""
        if session["chat_history"]:
            for msg in session["chat_history"][-6:]:
                role = "Doctor" if msg["role"] == "user" else "AI"
                chat_history_text += f"{role}: {msg['content']}\n"
        
        chat_prompt = f"""
PATIENT DATA (Use ONLY this data for your response):
{summary_text}

PREVIOUS NOTES GENERATED:
{json.dumps(session.get('generated_notes'), indent=2) if session.get('generated_notes') else 'None yet'}

CHAT HISTORY:
{chat_history_text if chat_history_text else 'No previous messages'}

DOCTOR'S QUESTION:
{message}

INSTRUCTIONS:
- Answer based ONLY on the patient data provided
- If the question asks for information not in the data, say "This information is not available in the current dataset"
- Use professional medical terminology
- Ask clarifying questions if the query is ambiguous
- Be concise and clinically relevant
"""
        
        # Use BBY agent's llm_gemini function
        response = bby.llm_gemini(bby.SYSTEM_PROMPT, chat_prompt)
        
        # Update chat history
        session["chat_history"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        session["chat_history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "response": response,
            "type": "success",
            "patient_id": patient_id,
            "model": "gemini-2.5-flash-lite via BBY Agent",
            "data_source": "synthetic_dataset",
            "context_available": True,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "error_type": "PATIENT_NOT_FOUND"
        }), 404
    except RuntimeError as e:
        return jsonify({
            "error": str(e),
            "error_type": "API_ERROR"
        }), 500
    except Exception as e:
        return jsonify({
            "error": str(e),
            "error_type": "PROCESSING_ERROR"
        }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MedGamma Backend (Synthetic Patient Data + BBY Agent)")
    print("=" * 60)
    print(f"BBY Agent AI: {'[OK]' if bby.gemini_model else '[NOT CONFIGURED]'}")
    print(f"Patients Loaded: {len(SYNTHETIC_PATIENTS)}")
    print(f"Patient IDs: {', '.join(get_patient_ids())}")
    print("=" * 60)
    print("Endpoints:")
    print("  GET  /api/health           - System status")
    print("  GET  /api/patients         - List all patients")
    print("  GET  /api/patient/<id>     - Get patient details")
    print("  POST /api/generate-notes   - Generate SOAP notes")
    print("  POST /api/chat             - Clinical chatbot")
    print("=" * 60)
    
    # use_reloader=False prevents the common Windows "immediate exit" bug
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
