import google.generativeai as genai
from loguru import logger

genai.configure(api_key="") 

def generate_recommendations(diabetes_prob: float, heart_prob: float, kidney_prob: float) -> str:
    prompt = f"""
    Patient Risk Profile:
    - Diabetes likelihood: {diabetes_prob:.1%}
    - Heart disease likelihood: {heart_prob:.1%}
    - Kidney disease likelihood: {kidney_prob:.1%}

    Provide **concise, actionable** lifestyle and medical recommendations.
    Prioritize high-risk areas. Include diet, exercise, monitoring, and when to see a doctor.
    Keep under 500 words.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "Warning: Consult a healthcare provider immediately for personalized advice."