import easyocr
import io
import numpy as np
from PIL import Image

# Initialize the easyocr reader
# This automatically downloads the pre-trained weights for the deep learning models
# We disable GPU by default so it runs smoothly on CPU without CUDA setup errors
reader = easyocr.Reader(['en', 'es'], gpu=False)

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Takes raw image bytes, converts it to a numpy array for EasyOCR,
    and returns a single string of all extracted text.
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Ensure image is RGB (EasyOCR handles standard formats well)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image_np = np.array(image)
    
    # Predict text using EasyOCR
    # Readtext returns a list of tuples: (bbox, text, confidence)
    results = reader.readtext(image_np)
    
    # Extract just the text strings
    extracted_words = [text for bbox, text, conf in results]
    
    # Join into a single string
    full_text = " ".join(extracted_words)
    return full_text
