import os
import json
from groq import Groq
from backend.core.security import validate_llm_output, sanitize_input
from backend.agents.tools import predict_tool, explain_tool, memory_tool

def call_agent(features: dict) -> dict:
    # 1. Prediction Tool
    from backend.models.schemas import PatientFeatures
    pf = PatientFeatures(**features)
    pred = predict_tool(pf)
    risk_score = pred['risk_probability']
    label = pred['label']
    scaled = pred['scaled_features']

    # 2. Explanation Tool
    top_factors = explain_tool(scaled)

    # 3. Memory Tool
    record_id = memory_tool(risk_score, top_factors)

    # 4. LLM Recommendation
    # Security: input natively constructed with numeric payload
    llm_input = {
        "risk": risk_score,
        "top_factors": top_factors,
        "patient_data": features
    }

    sys_prompt = """You are a cardiovascular risk analyst agent.
RULES:
1. Provide personalized lifestyle recommendations based strictly on "top_factors".
2. DO NOT provide a diagnosis.
3. DO NOT suggest medications.
4. Explanations must reference the top contributing features.
5. Your output must be a valid JSON with keys: "explanation" (string), "recommendation" (string).
"""
    user_prompt = json.dumps(llm_input)

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return {
            "prediction": {"score": risk_score, "label": label},
            "top_factors": top_factors,
            "explanation": "Groq API Key not configured.",
            "recommendation": "Configure GROQ_API_KEY in .env to enable Agentic Recommendations.",
            "history_record_id": record_id
        }

    try:
        client = Groq(api_key=groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content
        if not validate_llm_output(response_text):
            response_text = '{"explanation": "Content blocked due to safety guidelines.", "recommendation": "Consult a local physician."}'
            
        out_json = json.loads(response_text)
    except Exception as e:
        out_json = {"explanation": "Error communicating with Groq API.", "recommendation": str(e)}

    return {
        "prediction": {"score": risk_score, "label": label},
        "top_factors": top_factors,
        "explanation": out_json.get("explanation", ""),
        "recommendation": out_json.get("recommendation", ""),
        "history_record_id": record_id
    }
