"""Gemini AI integration for Hospital Strategic Insights."""
import requests
import json
from typing import Dict, Any

def get_gemini_analysis(summary_stats: str, api_key: str) -> str:
    """
    Get strategic insights from Google Gemini based on patient data summary.
    
    Args:
        summary_stats (str): Text description of the current patient risk state.
        api_key (str): Google Gemini API Key.
        
    Returns:
        str: MARKDOWN formatted response from Gemini with reasoning and predictions.
    """
    if not api_key:
        return "⚠️ API Key is missing. Cannot generate insights."

    # Reverting to reliable model and increasing timeout
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    You are a Senior Hospital Strategy Advisor and Data Scientist.
    
    Analyze the following summary of patient risk data for the upcoming week:
    {summary_stats}
    
    Provide a concise strategic report with TWO sections:
    
    ### 1. Root Cause Analysis (Constraints)
    Explain the primary constraints causing patients to be 'High Risk' or unable to visit. Focus on patterns like Weather, Distance, or previous history.
    
    ### 2. Future Strategic Prediction
    Predict the operational impact on the hospital and suggest specific actions. 
    (e.g., "Due to expected storms, expect 20% drop in physical visits. Increase Telehealth staffing on Tuesday.")
    
    Keep the tone professional, insightful, and actionable. Use Markdown formatting.
    """
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        # Extract text: candidates[0].content.parts[0].text
        try:
            insight_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return insight_text
        except (KeyError, IndexError):
            return "⚠️ Unclear response from Gemini API. Please try again."
            
    except Exception as e:
        return f"⚠️ Analysis Failed: {str(e)}"
