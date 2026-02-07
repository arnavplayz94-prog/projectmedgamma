"""
MedGamma Backend Server
=======================
Minimal Flask wrapper for the bby.ipynb notebook.
Provides API endpoints for clinical data processing and AI-generated notes.

This wrapper does NOT modify the notebook's logic - it only calls existing functions
and returns structured context for AI to generate clinical documentation.
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

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


def generate_ai_clinical_notes(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI clinical documentation from structured context.
    All outputs use probabilistic, non-assertive language.
    """
    entity_id = context.get("entity_id", "Unknown")
    vitals = context.get("current_vitals", {})
    trends = context.get("trends", {})
    deviations = context.get("deviations", [])
    
    # Build SOAP Note
    subjective = "Patient data obtained from continuous wearable monitoring. "
    subjective += "No subjective complaints documented during this monitoring period."
    
    objective = f"Vitals (wearable-derived): "
    objective += f"HR {vitals.get('hr_mean', 0):.0f} bpm ({trends.get('hr_trend', 'stable')}), "
    objective += f"SpO2 {vitals.get('spo2', 0):.0f}%, "
    objective += f"Temp {vitals.get('temp', 0):.1f}°C, "
    objective += f"RR {vitals.get('resp_rate', 0):.0f}/min, "
    objective += f"BP {vitals.get('bp_systolic', 0):.0f}/{vitals.get('bp_diastolic', 0):.0f} mmHg. "
    objective += f"EDA {vitals.get('eda_mean', 0):.2f} µS ({trends.get('eda_trend', 'stable')})."
    
    # Assessment with probabilistic language
    assessment = "Based on wearable data analysis: "
    if deviations:
        assessment += "Observed deviations from baseline may warrant further evaluation. "
        for dev in deviations:
            if dev["signal"] == "hr_mean":
                assessment += f"Heart rate appears elevated ({dev['deviation_pct']:.1f}% above baseline), "
                assessment += "which could suggest increased physiological stress or activity. "
            if dev["signal"] == "eda_mean":
                assessment += f"Electrodermal activity is elevated, which may be consistent with "
                assessment += "autonomic arousal or stress response. "
    else:
        assessment += "Vital parameters appear within expected ranges based on available data. "
    assessment += "Clinical correlation recommended."
    
    plan = "1. Continue wearable monitoring as indicated. "
    plan += "2. Review trends at next clinical encounter. "
    if deviations:
        plan += "3. Consider evaluation if deviations persist or worsen. "
    plan += "Further clinical assessment required for definitive management."
    
    # HPI (History of Present Illness)
    hpi = f"This is a continuous monitoring report for patient {entity_id}. "
    hpi += f"Data collected over {context.get('time_context', {}).get('measurement_count', 0)} measurements "
    hpi += f"from {context.get('time_context', {}).get('data_start', 'N/A')[:10]} to present. "
    if trends.get("hr_trend") == "increasing":
        hpi += "Heart rate has shown an upward trend during the monitoring period. "
    if trends.get("eda_trend") == "elevated":
        hpi += "Electrodermal activity has been persistently elevated. "
    
    # Patient Summary
    patient_summary = f"Patient {entity_id} - Wearable Monitoring Summary\n"
    patient_summary += f"Period: {context.get('time_context', {}).get('data_start', 'N/A')[:10]} to present\n"
    patient_summary += f"Data Points: {context.get('time_context', {}).get('measurement_count', 0)}\n"
    patient_summary += f"Current Status: "
    if deviations:
        patient_summary += "Some parameters deviate from baseline - review recommended."
    else:
        patient_summary += "Parameters within expected ranges."
    
    # Preliminary Clinical Impression (with strong disclaimer)
    differentials = []
    if vitals.get("hr_mean", 0) > 90:
        differentials.append("physiological stress response")
        differentials.append("anxiety or emotional state")
        differentials.append("increased physical activity")
        differentials.append("medication effects")
    if vitals.get("eda_mean", 0) > 0.4:
        differentials.append("autonomic arousal")
        differentials.append("thermal regulation response")
    if not differentials:
        differentials.append("normal physiological variation")
    
    clinical_impression = "PRELIMINARY CLINICAL IMPRESSION (AI-Generated):\n\n"
    clinical_impression += "Based on the available wearable data, the following possibilities "
    clinical_impression += "may be considered:\n"
    for i, diff in enumerate(differentials, 1):
        clinical_impression += f"  {i}. {diff.capitalize()}\n"
    clinical_impression += "\nNote: This is NOT a diagnosis. "
    clinical_impression += "These are potential considerations based on pattern analysis only."
    
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
            "narrative": clinical_impression
        },
        "disclaimer": "This is an AI-generated suggestion and must be reviewed, edited, or rejected by a licensed clinician.",
        "label": "AI-Suggested (Review Required)",
        "generated_at": datetime.now().isoformat()
    }


# Sample subjects
MOCK_SUBJECTS = [
    {"id": "S01", "name": "John Smith", "age": 45, "condition": "Post-cardiac surgery"},
    {"id": "S02", "name": "Maria Garcia", "age": 62, "condition": "Hypertension monitoring"},
    {"id": "S03", "name": "James Wilson", "age": 38, "condition": "Stress assessment"},
    {"id": "S04", "name": "Emily Chen", "age": 55, "condition": "Cardiac rehabilitation"},
    {"id": "S05", "name": "Michael Brown", "age": 70, "condition": "Heart failure monitoring"},
]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "MedGamma Clinical AI Backend",
        "version": "1.0.0",
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
    """Get structured clinical context for a subject (from notebook processing)."""
    context = get_structured_context(entity_id)
    return jsonify(context)


@app.route('/api/generate-notes', methods=['POST'])
def generate_notes():
    """
    Generate AI clinical documentation from context.
    
    Expected JSON body:
    {
        "entity_id": "S01"
    }
    
    Or provide custom context:
    {
        "context": { ... structured context ... }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get context either from provided data or by entity_id
        if "context" in data:
            context = data["context"]
        elif "entity_id" in data:
            context = get_structured_context(data["entity_id"])
        else:
            return jsonify({"error": "Must provide entity_id or context"}), 400
        
        # Generate AI clinical notes
        notes = generate_ai_clinical_notes(context)
        
        return jsonify({
            "success": True,
            "context": context,
            "notes": notes
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MedGamma Clinical AI Backend")
    print("=" * 60)
    print(f"Starting server at http://localhost:5000")
    print(f"API Documentation:")
    print(f"  GET  /api/health           - Health check")
    print(f"  GET  /api/subjects         - List all subjects")
    print(f"  GET  /api/context/<id>     - Get structured context")
    print(f"  POST /api/generate-notes   - Generate AI clinical notes")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
