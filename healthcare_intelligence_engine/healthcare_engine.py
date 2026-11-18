"""
Healthcare Intelligence Engine (simple base version)
----------------------------------------------------
Modules:
  1. Patient model
  2. Vitals & BMI analysis
  3. Rule-based risk scoring (cardiac, diabetes, overall)
  4. Symptom -> possible condition matcher
  5. Drug interaction checker
  6. Population analytics (cohort summary)
  7. Report generator

Pure Python 3.8+, no external dependencies.

DISCLAIMER: Educational demo only. Rules are simplified and NOT a
substitute for professional medical advice, diagnosis, or treatment.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from collections import Counter
import json


# ----------------------------------------------------------------------
# 1. Patient model
# ----------------------------------------------------------------------
@dataclass
class Patient:
    patient_id: str
    name: str
    age: int
    sex: str                      # "M" / "F"
    height_cm: float
    weight_kg: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    glucose_mgdl: float           # fasting glucose
    cholesterol_mgdl: float
    smoker: bool = False
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)

    @property
    def bmi(self) -> float:
        h = self.height_cm / 100
        return round(self.weight_kg / (h * h), 1)


# ----------------------------------------------------------------------
# 2. Vitals analysis
# ----------------------------------------------------------------------
def classify_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def classify_bp(sys_bp: int, dia_bp: int) -> str:
    if sys_bp >= 180 or dia_bp >= 120:
        return "Hypertensive Crisis"
    if sys_bp >= 140 or dia_bp >= 90:
        return "Hypertension Stage 2"
    if sys_bp >= 130 or dia_bp >= 80:
        return "Hypertension Stage 1"
    if sys_bp >= 120:
        return "Elevated"
    return "Normal"


def classify_glucose(g: float) -> str:
    if g < 70:
        return "Low (Hypoglycemia)"
    if g < 100:
        return "Normal"
    if g < 126:
        return "Prediabetic range"
    return "Diabetic range"


# ----------------------------------------------------------------------
# 3. Risk scoring (points-based, 0-100 scale per category)
# ----------------------------------------------------------------------
def cardiac_risk(p: Patient) -> int:
    score = 0
    score += 10 if p.age >= 45 else 0
    score += 10 if p.age >= 60 else 0
    score += 5 if p.sex == "M" else 0
    score += 20 if p.systolic_bp >= 140 else (10 if p.systolic_bp >= 130 else 0)
    score += 15 if p.cholesterol_mgdl >= 240 else (8 if p.cholesterol_mgdl >= 200 else 0)
    score += 15 if p.smoker else 0
    score += 10 if p.bmi >= 30 else 0
    score += 10 if p.glucose_mgdl >= 126 else 0
    score += 15 if "heart disease" in [c.lower() for c in p.conditions] else 0
    return min(score, 100)


def diabetes_risk(p: Patient) -> int:
    score = 0
    score += 30 if p.glucose_mgdl >= 126 else (20 if p.glucose_mgdl >= 100 else 0)
    score += 20 if p.bmi >= 30 else (10 if p.bmi >= 25 else 0)
    score += 10 if p.age >= 45 else 0
    score += 10 if p.systolic_bp >= 130 else 0
    score += 20 if "family history diabetes" in [c.lower() for c in p.conditions] else 0
    score += 10 if "diabetes" in [c.lower() for c in p.conditions] else 0
    return min(score, 100)


def risk_level(score: int) -> str:
    if score >= 60:
        return "HIGH"
    if score >= 30:
        return "MODERATE"
    return "LOW"


# ----------------------------------------------------------------------
# 4. Symptom matcher (rule-based knowledge base)
# ----------------------------------------------------------------------
SYMPTOM_KB: Dict[str, List[str]] = {
    "Common Cold": ["cough", "sneezing", "runny nose", "sore throat", "mild fever"],
    "Influenza": ["fever", "body ache", "chills", "fatigue", "cough", "headache"],
    "Type 2 Diabetes": ["increased thirst", "frequent urination", "fatigue", "blurred vision", "slow healing"],
    "Hypertension": ["headache", "dizziness", "blurred vision", "chest pain", "shortness of breath"],
    "Migraine": ["headache", "nausea", "light sensitivity", "blurred vision"],
    "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "stomach pain", "fever"],
    "Anemia": ["fatigue", "pale skin", "dizziness", "shortness of breath", "cold hands"],
    "Asthma": ["wheezing", "shortness of breath", "cough", "chest tightness"],
}

EMERGENCY_SYMPTOMS = {"chest pain", "shortness of breath", "fainting", "severe bleeding",
                      "slurred speech", "sudden weakness"}


def match_symptoms(symptoms: List[str], top_n: int = 3) -> List[Dict]:
    sx = {s.strip().lower() for s in symptoms}
    results = []
    for condition, kb_symptoms in SYMPTOM_KB.items():
        matched = sx.intersection(kb_symptoms)
        if matched:
            confidence = round(len(matched) / len(kb_symptoms) * 100)
            results.append({"condition": condition,
                            "matched": sorted(matched),
                            "confidence_pct": confidence})
    results.sort(key=lambda r: r["confidence_pct"], reverse=True)
    return results[:top_n]


def emergency_flags(symptoms: List[str]) -> List[str]:
    return sorted({s.lower() for s in symptoms}.intersection(EMERGENCY_SYMPTOMS))


# ----------------------------------------------------------------------
# 5. Drug interaction checker (small sample database)
# ----------------------------------------------------------------------
INTERACTIONS = {
    frozenset(["warfarin", "aspirin"]): ("HIGH", "Increased bleeding risk"),
    frozenset(["metformin", "alcohol"]): ("MODERATE", "Risk of lactic acidosis"),
    frozenset(["lisinopril", "ibuprofen"]): ("MODERATE", "Reduced BP control, kidney stress"),
    frozenset(["simvastatin", "clarithromycin"]): ("HIGH", "Risk of muscle damage"),
    frozenset(["sildenafil", "nitroglycerin"]): ("HIGH", "Severe drop in blood pressure"),
}


def check_interactions(meds: List[str]) -> List[Dict]:
    meds = [m.lower() for m in meds]
    found = []
    for i in range(len(meds)):
        for j in range(i + 1, len(meds)):
            key = frozenset([meds[i], meds[j]])
            if key in INTERACTIONS:
                sev, note = INTERACTIONS[key]
                found.append({"drugs": [meds[i], meds[j]], "severity": sev, "note": note})
    return found


# ----------------------------------------------------------------------
# 6. Population analytics
# ----------------------------------------------------------------------
def cohort_summary(patients: List[Patient]) -> Dict:
    n = len(patients)
    if n == 0:
        return {}
    return {
        "total_patients": n,
        "avg_age": round(sum(p.age for p in patients) / n, 1),
        "avg_bmi": round(sum(p.bmi for p in patients) / n, 1),
        "smokers_pct": round(sum(p.smoker for p in patients) / n * 100, 1),
        "bmi_distribution": dict(Counter(classify_bmi(p.bmi) for p in patients)),
        "bp_distribution": dict(Counter(classify_bp(p.systolic_bp, p.diastolic_bp) for p in patients)),
        "high_cardiac_risk": [p.name for p in patients if risk_level(cardiac_risk(p)) == "HIGH"],
        "high_diabetes_risk": [p.name for p in patients if risk_level(diabetes_risk(p)) == "HIGH"],
    }


# ----------------------------------------------------------------------
# 7. Recommendations & report
# ----------------------------------------------------------------------
def recommendations(p: Patient) -> List[str]:
    recs = []
    if p.bmi >= 25:
        recs.append("Adopt a balanced diet and aim for 150 min/week of moderate exercise.")
    if p.smoker:
        recs.append("Enroll in a smoking-cessation program.")
    if p.systolic_bp >= 130 or p.diastolic_bp >= 80:
        recs.append("Reduce sodium intake and monitor blood pressure regularly.")
    if p.glucose_mgdl >= 100:
        recs.append("Schedule an HbA1c test and limit refined sugars.")
    if p.cholesterol_mgdl >= 200:
        recs.append("Request a full lipid panel; reduce saturated fats.")
    if not recs:
        recs.append("Maintain current healthy lifestyle and annual check-ups.")
    return recs


def generate_report(p: Patient) -> str:
    c, d = cardiac_risk(p), diabetes_risk(p)
    lines = [
        "=" * 60,
        f"HEALTH INTELLIGENCE REPORT - {p.name} ({p.patient_id})",
        "=" * 60,
        f"Age/Sex      : {p.age} / {p.sex}",
        f"BMI          : {p.bmi} ({classify_bmi(p.bmi)})",
        f"Blood Pressure: {p.systolic_bp}/{p.diastolic_bp} ({classify_bp(p.systolic_bp, p.diastolic_bp)})",
        f"Glucose      : {p.glucose_mgdl} mg/dL ({classify_glucose(p.glucose_mgdl)})",
        f"Cholesterol  : {p.cholesterol_mgdl} mg/dL",
        "",
        f"Cardiac Risk : {c}/100 -> {risk_level(c)}",
        f"Diabetes Risk: {d}/100 -> {risk_level(d)}",
    ]

    if p.symptoms:
        flags = emergency_flags(p.symptoms)
        if flags:
            lines.append(f"\n!! EMERGENCY SYMPTOMS: {', '.join(flags)} - seek immediate care !!")
        lines.append("\nPossible conditions (from symptoms):")
        for m in match_symptoms(p.symptoms):
            lines.append(f"  - {m['condition']} ({m['confidence_pct']}% match: {', '.join(m['matched'])})")

    interactions = check_interactions(p.medications)
    if interactions:
        lines.append("\nDrug interaction warnings:")
        for i in interactions:
            lines.append(f"  - [{i['severity']}] {' + '.join(i['drugs'])}: {i['note']}")

    lines.append("\nRecommendations:")
    lines += [f"  * {r}" for r in recommendations(p)]
    lines.append("\n(Educational tool only - consult a qualified clinician.)")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
def main():
    patients = [
        Patient("P001", "Ravi Kumar", 58, "M", 172, 92, 145, 95, 82, 134, 248,
                smoker=True, conditions=["Family History Diabetes"],
                medications=["Warfarin", "Aspirin"],
                symptoms=["chest pain", "dizziness", "headache"]),
        Patient("P002", "Anita Sharma", 32, "F", 160, 54, 112, 72, 70, 88, 170,
                symptoms=["cough", "sneezing", "sore throat"]),
        Patient("P003", "Suresh Reddy", 49, "M", 168, 85, 132, 84, 78, 118, 215,
                medications=["Lisinopril", "Ibuprofen"],
                symptoms=["increased thirst", "frequent urination", "fatigue"]),
    ]

    for p in patients:
        print(generate_report(p))
        print()

    print("COHORT SUMMARY")
    print(json.dumps(cohort_summary(patients), indent=2))


if __name__ == "__main__":
    main()
