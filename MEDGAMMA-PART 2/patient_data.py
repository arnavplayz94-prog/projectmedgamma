# patient_data.py
# ================
# Synthetic patient dataset for MedGamma backend.
# All AI outputs MUST be derived from this data only.

from typing import Dict, List, Any, Optional

SYNTHETIC_PATIENTS = {
    "P001": {
        "patient_id": "P001",
        "demographics": {
            "age": 45,
            "sex": "Male",
            "height_cm": 172,
            "weight_kg": 78
        },
        "vitals": {
            "heart_rate_avg": 82,
            "blood_pressure": "138/88",
            "respiratory_rate": 18,
            "body_temperature_c": 37.1,
            "spo2_percent": 96
        },
        "wearable_data": {
            "steps_per_day_avg": 5200,
            "sleep_hours_avg": 6.2,
            "hrv_ms_avg": 42,
            "stress_index": "Moderate"
        },
        "symptoms_reported": [
            "Fatigue",
            "Occasional headache",
            "Poor sleep quality"
        ],
        "medical_history": [
            "Hypertension (diagnosed 3 years ago)"
        ],
        "current_medications": [
            "Amlodipine 5mg once daily"
        ]
    },
    "P002": {
        "patient_id": "P002",
        "demographics": {
            "age": 29,
            "sex": "Female",
            "height_cm": 160,
            "weight_kg": 55
        },
        "vitals": {
            "heart_rate_avg": 74,
            "blood_pressure": "112/72",
            "respiratory_rate": 16,
            "body_temperature_c": 36.8,
            "spo2_percent": 99
        },
        "wearable_data": {
            "steps_per_day_avg": 9100,
            "sleep_hours_avg": 7.8,
            "hrv_ms_avg": 65,
            "stress_index": "Low"
        },
        "symptoms_reported": [
            "Occasional anxiety during work hours"
        ],
        "medical_history": [],
        "current_medications": []
    },
    "P003": {
        "patient_id": "P003",
        "demographics": {
            "age": 61,
            "sex": "Male",
            "height_cm": 168,
            "weight_kg": 82
        },
        "vitals": {
            "heart_rate_avg": 88,
            "blood_pressure": "146/92",
            "respiratory_rate": 20,
            "body_temperature_c": 37.3,
            "spo2_percent": 94
        },
        "wearable_data": {
            "steps_per_day_avg": 3100,
            "sleep_hours_avg": 5.4,
            "hrv_ms_avg": 35,
            "stress_index": "High"
        },
        "symptoms_reported": [
            "Shortness of breath on exertion",
            "Chronic fatigue"
        ],
        "medical_history": [
            "Type 2 Diabetes",
            "Hypertension"
        ],
        "current_medications": [
            "Metformin 500mg twice daily",
            "Losartan 50mg once daily"
        ]
    }
}


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve patient data by ID.
    Returns None if patient not found.
    """
    return SYNTHETIC_PATIENTS.get(patient_id)


def get_all_patients() -> List[Dict[str, Any]]:
    """Get list of all patients."""
    return list(SYNTHETIC_PATIENTS.values())


def get_patient_ids() -> List[str]:
    """Get list of all patient IDs."""
    return list(SYNTHETIC_PATIENTS.keys())


def validate_patient_exists(patient_id: str) -> None:
    """
    Validate that a patient exists.
    Raises ValueError if not found.
    """
    if patient_id not in SYNTHETIC_PATIENTS:
        available = ", ".join(SYNTHETIC_PATIENTS.keys())
        raise ValueError(f"Patient {patient_id} not found. Available patients: {available}")


def build_patient_context(patient_id: str) -> Dict[str, Any]:
    """
    Build structured context from patient data for Gemini.
    All fields are traceable to the dataset.
    """
    validate_patient_exists(patient_id)
    patient = SYNTHETIC_PATIENTS[patient_id]
    
    # Parse blood pressure
    bp = patient["vitals"]["blood_pressure"].split("/")
    bp_systolic = int(bp[0])
    bp_diastolic = int(bp[1])
    
    # Calculate BMI
    height_m = patient["demographics"]["height_cm"] / 100
    bmi = round(patient["demographics"]["weight_kg"] / (height_m ** 2), 1)
    
    # Risk assessment based on data
    risk_factors = []
    risk_level = "low"
    
    # Check vitals for concerns
    if bp_systolic >= 140 or bp_diastolic >= 90:
        risk_factors.append("Elevated blood pressure")
        risk_level = "medium"
    
    if patient["vitals"]["spo2_percent"] < 95:
        risk_factors.append("Low oxygen saturation")
        risk_level = "high"
    
    if patient["wearable_data"]["hrv_ms_avg"] < 40:
        risk_factors.append("Low heart rate variability")
        if risk_level == "low":
            risk_level = "medium"
    
    if patient["wearable_data"]["stress_index"] == "High":
        risk_factors.append("High stress index from wearable")
        if risk_level == "low":
            risk_level = "medium"
    
    if patient["wearable_data"]["sleep_hours_avg"] < 6:
        risk_factors.append("Insufficient sleep")
    
    if len(patient["medical_history"]) >= 2:
        risk_factors.append("Multiple comorbidities")
        risk_level = "high" if risk_level != "high" else risk_level
    
    return {
        "patient_id": patient_id,
        "demographics": patient["demographics"],
        "bmi": bmi,
        "vitals": {
            **patient["vitals"],
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic
        },
        "wearable_data": patient["wearable_data"],
        "symptoms": patient["symptoms_reported"],
        "medical_history": patient["medical_history"],
        "medications": patient["current_medications"],
        "risk_assessment": {
            "level": risk_level,
            "factors": risk_factors
        },
        "data_source": "synthetic_dataset"
    }


def build_subjective_data(patient_id: str) -> str:
    """Build subjective section from patient-reported data."""
    validate_patient_exists(patient_id)
    patient = SYNTHETIC_PATIENTS[patient_id]
    
    symptoms = patient["symptoms_reported"]
    history = patient["medical_history"]
    
    parts = []
    
    if symptoms:
        parts.append(f"Patient reports: {', '.join(symptoms)}.")
    else:
        parts.append("No symptoms currently reported.")
    
    if history:
        parts.append(f"Medical history includes: {', '.join(history)}.")
    
    return " ".join(parts)


def build_objective_data(patient_id: str) -> str:
    """Build objective section from vitals and measurements."""
    validate_patient_exists(patient_id)
    patient = SYNTHETIC_PATIENTS[patient_id]
    
    d = patient["demographics"]
    v = patient["vitals"]
    w = patient["wearable_data"]
    
    height_m = d["height_cm"] / 100
    bmi = round(d["weight_kg"] / (height_m ** 2), 1)
    
    return (
        f"Demographics: {d['age']}-year-old {d['sex']}, "
        f"Height {d['height_cm']} cm, Weight {d['weight_kg']} kg, BMI {bmi}. "
        f"Vitals: HR {v['heart_rate_avg']} bpm, BP {v['blood_pressure']} mmHg, "
        f"RR {v['respiratory_rate']}/min, Temp {v['body_temperature_c']}°C, SpO2 {v['spo2_percent']}%. "
        f"Wearable Data: {w['steps_per_day_avg']} steps/day avg, "
        f"{w['sleep_hours_avg']}h sleep avg, HRV {w['hrv_ms_avg']}ms, Stress: {w['stress_index']}."
    )


def patient_summary_text(patient_id: str) -> str:
    """Generate structured patient summary text for Gemini input."""
    validate_patient_exists(patient_id)
    ctx = build_patient_context(patient_id)
    
    return f"""
PATIENT: {ctx['patient_id']}
Age: {ctx['demographics']['age']} | Sex: {ctx['demographics']['sex']} | BMI: {ctx['bmi']}

VITALS:
- Heart Rate: {ctx['vitals']['heart_rate_avg']} bpm
- Blood Pressure: {ctx['vitals']['blood_pressure']} mmHg
- Respiratory Rate: {ctx['vitals']['respiratory_rate']}/min
- Temperature: {ctx['vitals']['body_temperature_c']}°C
- SpO2: {ctx['vitals']['spo2_percent']}%

WEARABLE DATA:
- Steps/day: {ctx['wearable_data']['steps_per_day_avg']}
- Sleep: {ctx['wearable_data']['sleep_hours_avg']} hours avg
- HRV: {ctx['wearable_data']['hrv_ms_avg']} ms
- Stress Index: {ctx['wearable_data']['stress_index']}

SYMPTOMS: {', '.join(ctx['symptoms']) if ctx['symptoms'] else 'None reported'}

MEDICAL HISTORY: {', '.join(ctx['medical_history']) if ctx['medical_history'] else 'None'}

CURRENT MEDICATIONS: {', '.join(ctx['medications']) if ctx['medications'] else 'None'}

RISK ASSESSMENT: {ctx['risk_assessment']['level'].upper()}
Risk Factors: {', '.join(ctx['risk_assessment']['factors']) if ctx['risk_assessment']['factors'] else 'None identified'}
"""
