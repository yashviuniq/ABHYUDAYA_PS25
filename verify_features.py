import pandas as pd
import joblib
import numpy as np
from ocr_engine import extract_ehr_data
from PIL import Image

def verify_weather_impact():
    print("Verifying Weather Impact...")
    model = joblib.load("models/rf_pipeline.joblib")
    
    # Base patient
    patient = {
        "lead_time": 10,
        "distance_km": 5.0,
        "past_no_shows": 0,
        "age": 30,
        "gender": "M",
        "appointment_type": "PrimaryCare",
        "weather": "Sunny"
    }
    
    df_sunny = pd.DataFrame([patient])
    prob_sunny = model.predict_proba(df_sunny)[:, 1][0]
    
    patient["weather"] = "Stormy"
    df_stormy = pd.DataFrame([patient])
    prob_stormy = model.predict_proba(df_stormy)[:, 1][0]
    
    print(f"Prob Sunny: {prob_sunny:.4f}")
    print(f"Prob Stormy: {prob_stormy:.4f}")
    
    if prob_stormy > prob_sunny:
        print("PASS: Stormy weather increases risk.")
    else:
        print("FAIL: Stormy weather did not increase risk.")

def verify_ocr_robustness():
    print("\nVerifying OCR Robustness...")
    # Create a dummy image
    img = Image.new('RGB', (100, 30), color = (255, 255, 255))
    
    try:
        # This calls pytesseract. If binary missing, should hit fallback
        res = extract_ehr_data(img)
        print("OCR Result:", res)
        if isinstance(res, dict) and "Age" in res:
            print("PASS: OCR returned valid dict (likely fallback or real).")
        else:
            print("FAIL: OCR returned unexpected format.")
    except Exception as e:
        print(f"FAIL: OCR crashed with {e}")

if __name__ == "__main__":
    verify_weather_impact()
    verify_ocr_robustness()
