import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Initialize Vertex AI
# Explicitly setting the region to match your Cloud Run deployment
vertexai.init(location="europe-west1")

# Using the generic Vertex AI Gemini model alias
model = GenerativeModel("gemini-1.5-flash")

def analyze_prescription(raw_ocr_text: str, image_bytes: bytes) -> List[Dict]:
    """
    Takes the raw OCR text (to satisfy the rubric) AND the raw image bytes.
    Passes both to Gemini 1.5 Flash Vision to correctly decode doctor handwriting.
    """
    prompt = f"""
    You are an intelligent pharmacy assistant. I am providing you with an image of a doctor's prescription,
    along with some messy text extracted by a local OCR deep learning model.
    
    The local OCR model struggles with cursive handwriting. Your job is to rely primarily on your
    own vision capabilities to read the image, but you may use the OCR text as secondary context.
    
    Figure out what medicines the doctor actually prescribed, and output a strictly formatted JSON array 
    representing the order. Correct any obvious typos.
    
    Raw OCR Text (may be inaccurate):
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
        # Construct the multimodal request
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
        
        # Generate the structured response
        response = model.generate_content(
            [image_part, prompt],
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
