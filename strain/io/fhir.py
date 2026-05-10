import uuid
from datetime import datetime, timezone
from typing import Any

from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.riskassessment import RiskAssessment, RiskAssessmentPrediction
from fhir.resources.observation import Observation
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference

def generate_fhir_bundle(screening_results: dict[str, Any], patient_id: str = "demo-alex") -> dict[str, Any]:
    """
    Format STRAIN demo screening results into a FHIR R4 Collection Bundle using fhir.resources validation.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    # fhir.resources requires compliant date formats; Python's isoformat with timezone handles this well
    
    # Extract data securely
    dep_score = screening_results.get("depression_risk", {}).get("score", 0.0)
    anx_score = screening_results.get("anxiety_risk", {}).get("score", 0.0)
    recommendation = screening_results.get("recommendation", "Unknown")
    findings = "\n".join(screening_results.get("key_findings", []))

    patient_ref = Reference(reference=f"Patient/{patient_id}")

    # 1. Diagnostic Report
    diagnostic_report = DiagnosticReport(
        id=str(uuid.uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="8646-2", display="Electroencephalogram (EEG) study")]
        ),
        subject=patient_ref,
        effectiveDateTime=now_str,
        conclusion=f"Recommendation: {recommendation}",
        conclusionCode=[CodeableConcept(text=findings)]
    )

    # 2. Depression Risk Assessment
    depression_risk = RiskAssessment(
        id=str(uuid.uuid4()),
        status="final",
        subject=patient_ref,
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="71358-6", display="Depression screening score")]
        ),
        prediction=[
            RiskAssessmentPrediction(
                outcome=CodeableConcept(text="Risk of Depression (Demo proxy based on EEG features)"),
                probabilityDecimal=round(dep_score / 100.0, 3)
            )
        ]
    )

    # 3. Anxiety Risk Assessment
    anxiety_risk = RiskAssessment(
        id=str(uuid.uuid4()),
        status="final",
        subject=patient_ref,
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="70274-6", display="Generalized anxiety disorder screen")]
        ),
        prediction=[
            RiskAssessmentPrediction(
                outcome=CodeableConcept(text="Risk of Anxiety (Demo proxy based on EEG features)"),
                probabilityDecimal=round(anx_score / 100.0, 3)
            )
        ]
    )

    # 4. Observation: Beta/Alpha Ratio Proxy
    observation = Observation(
        id=str(uuid.uuid4()),
        status="final",
        category=[
            CodeableConcept(coding=[Coding(system="http://terminology.hl7.org/CodeSystem/observation-category", code="exam")])
        ],
        code=CodeableConcept(text="EEG Beta/Alpha Ratio (Demo)"),
        subject=patient_ref,
        effectiveDateTime=now_str,
        valueString=findings
    )

    # Base Bundle
    bundle = Bundle(
        id=str(uuid.uuid4()),
        type="collection",
        timestamp=now_str,
        entry=[
            BundleEntry(resource=diagnostic_report),
            BundleEntry(resource=depression_risk),
            BundleEntry(resource=anxiety_risk),
            BundleEntry(resource=observation)
        ]
    )

    return bundle.model_dump(exclude_none=True)

