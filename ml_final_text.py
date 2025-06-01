#!pip install transformers torch sentencepiece

import re
import numpy as np
from transformers import pipeline

_zero_shot_pipeline=None

labels_dict = {
    "gender": ["남성", "여성"],
    "alcohol": ["전혀 안 마심", "가끔 마심", "주 3회", "매일 마심"],
    "smoking": ["흡연자", "비흡연자"],
    "genetic": ["유전력 없음", "유전력 약간 있음", "유전력 강함"],
    "activity": ["운동 안 함", "가끔 운동", "주말 운동", "매일 1시간 이상 운동"],
    "diabetes": ["당뇨 있음", "당뇨 없음"],
    "hypertension": ["고혈압 있음", "고혈압 없음"]
}

def preprocess(text):
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        _zero_shot_pipeline=pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        framework="pt",  
        use_fast=False
        )
    
    pred_labels = {}
    for key, candidate_labels in labels_dict.items():
        out = _zero_shot_pipeline(text, candidate_labels)
        pred_labels[key] = out["labels"][0]

    match_age = re.search(r"(\d{1,3})\s*(세|살)|age\s*[:\-]?\s*(\d{1,3})", text, re.IGNORECASE)
    if match_age:
        age_val = int(match_age.group(1) or match_age.group(3))
    else:
        age_labels = ["20대", "30대", "40대", "50대", "60대", "70대 이상"]
        out_age = _zero_shot_pipeline(text, age_labels)
        label_age = out_age["labels"][0]
        age_val = {
            "20대": 25,
            "30대": 35,
            "40대": 45,
            "50대": 55,
            "60대": 65,
            "70대 이상": 75
        }.get(label_age, 50)

    match_weight = re.search(r"(몸무게|체중|weight)[^\d]{0,5}(\d{2,3})", text, re.IGNORECASE)
    match_height = re.search(r"(키|신장|height)[^\d]{0,5}(\d{3})", text, re.IGNORECASE)
    if match_weight and match_height:
        weight = float(match_weight.group(2))
        height_cm = float(match_height.group(2))
        height_m = height_cm / 100.0
        bmi_val = round(weight / (height_m ** 2), 1)
    else:
        bmi_val = 25.0

    match_liver = re.search(r"(간[^\d]{0,5}|liver)[^\d]{0,5}(\d{2,3})", text, re.IGNORECASE)
    if match_liver:
        lf = int(match_liver.group(2))
        liver_val = min(max(lf, 20), 100)
    else:
        liver_val = 60

    gender_val = 0 if pred_labels["gender"] == "남성" else 1

    alcohol_map = {
        "전혀 안 마심": 0.0,
        "가끔 마심": 2.0,
        "주 3회": 6.0,
        "매일 마심": 14.0
    }
    alcohol_val = alcohol_map.get(pred_labels["alcohol"], 6.0)

    smoking_val = 1 if pred_labels["smoking"] == "흡연자" else 0

    genetic_map = {
        "유전력 없음": 0,
        "유전력 약간 있음": 1,
        "유전력 강함": 2
    }
    genetic_val = genetic_map.get(pred_labels["genetic"], 1)

    activity_map = {
        "운동 안 함": 0.0,
        "가끔 운동": 1.0,
        "주말 운동": 2.5,
        "매일 1시간 이상 운동": 7.0
    }
    activity_val = activity_map.get(pred_labels["activity"], 3.0)

    diabetes_val = 1 if pred_labels["diabetes"] == "당뇨 있음" else 0
    hypertension_val = 1 if pred_labels["hypertension"] == "고혈압 있음" else 0

    vector = [
        age_val,
        gender_val,
        bmi_val,
        alcohol_val,
        smoking_val,
        genetic_val,
        activity_val,
        diabetes_val,
        hypertension_val,
        liver_val
    ]

    return np.array(vector).reshape(1, -1)
