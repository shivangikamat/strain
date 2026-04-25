import uuid
from datetime import datetime
from typing import Any

def generate_fhir_bundle(screening_results: dict[str, Any], patient_id: str = "demo-alex") -> dict[str, Any]:
    """
    Format EmotiScan demo screening results into a FHIR R4 Collection Bundle.
    """
    now_str = datetime.utcnow().isoformat() + "Z"
    
    # Extract data securely
    dep_score = screening_results.get("depression_risk", {}).get("score", 0.0)
    anx_score = screening_results.get("anxiety_risk", {}).get("score", 0.0)
    recommendation = screening_results.get("recommendation", "Unknown")
    findings = "\n".join(screening_results.get("key_findings", []))

    # Base Bundle
    bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": now_str,
        "entry": []
    }

    # 1. Diagnostic Report
    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": "final",
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "8646-2", "display": "Electroencephalogram (EEG) study"}]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": now_str,
        "conclusion": f"Recommendation: {recommendation}",
        "conclusionCode": [
            {"text": findings}
        ]
    }
    bundle["entry"].append({"resource": diagnostic_report})

    # 2. Depression Risk Assessment
    depression_risk = {
        "resourceType": "RiskAssessment",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "71358-6", "display": "Depression screening score"}]
        },
        "prediction": [
            {
                "outcome": {
                    "text": "Risk of Depression (Demo proxy based on EEG features)"
                },
                "probabilityDecimal": round(dep_score / 100.0, 3)
            }
        ]
    }
    bundle["entry"].append({"resource": depression_risk})

    # 3. Anxiety Risk Assessment
    anxiety_risk = {
        "resourceType": "RiskAssessment",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "70274-6", "display": "Generalized anxiety disorder screen"}]
        },
        "prediction": [
            {
                "outcome": {
                    "text": "Risk of Anxiety (Demo proxy based on EEG features)"
                },
                "probabilityDecimal": round(anx_score / 100.0, 3)
            }
        ]
    }
    bundle["entry"].append({"resource": anxiety_risk})

    # 4. Observation: Beta/Alpha Ratio Proxy
    # This grabs out one of the key_findings details to show raw EEG mapping
    observation = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [
            {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "exam"}]}
        ],
        "code": {
            "text": "EEG Beta/Alpha Ratio (Demo)"
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": now_str,
        "valueString": findings # In a real clinical flow, use valueQuantity for extracted floats
    }
    bundle["entry"].append({"resource": observation})

    return bundle
