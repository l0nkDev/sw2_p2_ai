import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Initialize Vertex AI
# Explicitly targeting Iowa (us-central1) for your new deployment
vertexai.init(location="us-central1")

# Using the generic Vertex AI Gemini model alias with strict system instructions
model = GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="You are a highly precise OCR data extraction API for a pharmacy. You only output valid JSON arrays. You never output markdown, conversation, or explanations."
)

def analyze_prescription(image_bytes: bytes) -> List[Dict]:
    """
    Passes raw image bytes natively to Gemini 1.5 Flash Vision to completely bypass local OCR.
    """
    prompt = f"""
    Analyze the provided image of a medical prescription and extract the medications ordered.
    
    Extract the medicines into a strictly formatted JSON array. 
    Do NOT wrap the output in markdown (like ```json). Just the raw array.
    
    Expected format:
    [
      {{
        "name": "Ibuprofen",
        "dosage": "200mg",
        "quantity": 30
      }}
    ]
    """
    
    try:
        # Construct the multimodal request
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
        
        # Generate the structured response with temperature 0.0 for deterministic extraction
        response = model.generate_content(
            [image_part, prompt],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
        )
        
        # Parse the JSON string returned by Vertex AI
        structured_data = json.loads(response.text)
        return structured_data
        
    except json.JSONDecodeError:
        return [{"name": "Error Parsing Prescription", "dosage": "N/A", "quantity": 0, "raw_text": "N/A"}]
    except Exception as e:
        print(f"Vertex AI Error: {e}")
        return [{"name": "AI Connection Error", "dosage": "N/A", "quantity": 0, "raw_text": "N/A"}]
