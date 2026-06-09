import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Initialize Vertex AI
# It will automatically look for GOOGLE_APPLICATION_CREDENTIALS in the environment
project_id = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
location = os.getenv("GCP_LOCATION", "us-central1")
vertexai.init(project=project_id, location=location)

# Using the Vertex AI Gemini model
model = GenerativeModel("gemini-1.5-flash-001")

def structure_prescription_text(raw_ocr_text: str) -> List[Dict]:
    """
    Takes raw, messy OCR text extracted by EasyOCR and uses Vertex AI Gemini 
    as an intelligent middleman to extract and return perfectly structured JSON.
    """
    prompt = f"""
    You are an intelligent pharmacy assistant. I will provide you with raw, messy text
    extracted from a doctor's prescription via an OCR deep learning model.
    
    Your job is to read this messy text, figure out what medicines the doctor actually prescribed,
    and output a strictly formatted JSON array representing the order. 
    Correct any obvious OCR typos (e.g. "Paracetamal" -> "Paracetamol").
    
    Raw OCR Text:
    "{raw_ocr_text}"
    
    Output strictly in this JSON format (no markdown, no backticks, just the raw JSON array):
    [
      {{
        "name": "string",
        "dosage": "string",
        "quantity": number
      }}
    ]
    """
    
    try:
        # Generate the structured response
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Parse the JSON string returned by Vertex AI
        structured_data = json.loads(response.text)
        return structured_data
        
    except json.JSONDecodeError:
        return [{"name": "Error Parsing Prescription", "dosage": "N/A", "quantity": 0, "raw_text": raw_ocr_text}]
    except Exception as e:
        print(f"Vertex AI Error: {e}")
        return [{"name": "AI Connection Error", "dosage": "N/A", "quantity": 0, "raw_text": raw_ocr_text}]
