import os
import json
import google.generativeai as genai
from typing import List, Dict
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

# Initialize Google AI Studio
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Using the generic Gemini model alias with strict system instructions
model = genai.GenerativeModel(
    "gemini-3.1-flash-lite",
    system_instruction=(
        "You are a highly precise OCR data extraction API for a pharmacy. "
        "You only output valid JSON arrays. You never output markdown, "
        "conversation, or explanations."
    ),
)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _generate_with_retry(image_bytes: bytes, prompt: str):
    """
    Helper function wrapped in Tenacity for exponential backoff retries.
    If it hits a 429 Too Many Requests, it silently waits and tries again up to 4 times.
    """
    return model.generate_content(
        [{"mime_type": "image/jpeg", "data": image_bytes}, prompt],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
    )


def analyze_prescription(image_bytes: bytes) -> List[Dict]:
    """
    Passes raw image bytes natively to Google AI Studio Gemini 1.5 Flash Vision.
    """
    prompt = """
    Analyze the provided image of a medical prescription and extract the
    medications ordered.

    Extract the medicines into a strictly formatted JSON array.
    Do NOT wrap the output in markdown (like ```json). Just the raw array.

    Expected format:
    [
      {
        "name": "Ibuprofen",
        "dosage": "200mg",
        "prescribed_units": 30,
        "unit_type": "pills"
      }
    ]

    Note: 'unit_type' should be the physical form prescribed (e.g., 'pills', 'capsules',
    'ml', 'drops', 'inhalations').
    'prescribed_units' must be the exact number written by the doctor for the full
    treatment (e.g., if it says 7 c, output 7).
    """

    try:
        # Generate the structured response with retry logic
        response = _generate_with_retry(image_bytes, prompt)

        # Parse the JSON string returned by Google AI Studio
        structured_data = json.loads(response.text)
        return structured_data

    except json.JSONDecodeError:
        return [
            {
                "name": "Error Parsing Prescription",
                "dosage": "N/A",
                "prescribed_units": 0,
                "unit_type": "N/A",
                "raw_text": "N/A",
            }
        ]
    except Exception as e:
        print(f"AI Studio Error: {e}")
        return [
            {
                "name": "AI Connection Error",
                "dosage": "N/A",
                "prescribed_units": 0,
                "unit_type": "N/A",
                "raw_text": "N/A",
            }
        ]
