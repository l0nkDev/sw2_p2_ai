from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_jwt
from vision_service import extract_text_from_image
from nlp_service import structure_prescription_text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FarmaFICCT AI Scanner Service")

# Configure CORS so the mobile app or frontend can access it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "sw2_p2_ai Deep Learning EasyOCR"}

@app.post("/api/ai/scan-prescription")
async def scan_prescription(
    file: UploadFile = File(...),
    user_id: int = Depends(verify_jwt)
):
    """
    Scans an uploaded prescription image.
    Requires a valid JWT Bearer token.
    Uses Keras-OCR (TensorFlow) to extract text, then structures it using NLP.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not an image."
        )
        
    try:
        image_bytes = await file.read()
        logger.info(f"User {user_id} requested prescription scan. Size: {len(image_bytes)} bytes.")
        
        # 1. Run Local Deep Learning Vision Model (EasyOCR / PyTorch)
        logger.info("Running EasyOCR...")
        raw_text = extract_text_from_image(image_bytes)
        logger.info(f"Extracted Raw Text: {raw_text}")
        
        # 2. Run NLP Structuring Model (SpaCy/Regex)
        logger.info("Structuring text...")
        structured_order = structure_prescription_text(raw_text)
        
        return {
            "success": True,
            "user_id": user_id,
            "raw_ocr_text": raw_text,
            "order": structured_order
        }
        
    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal AI processing error: {str(e)}"
        )
