from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_jwt
from nlp_service import analyze_prescription
import logging

# Set to CRITICAL to prevent Cloud Logging charges
logging.basicConfig(level=logging.CRITICAL)
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
    return {"status": "ok", "service": "sw2_p2_ai Gemini Vision"}

@app.post("/api/ai/scan-prescription")
async def scan_prescription(
    file: UploadFile = File(...),
    user_id: int = Depends(verify_jwt)
):
    """
    Scans an uploaded prescription image natively using Gemini 1.5 Flash Vision.
    Requires a valid JWT Bearer token.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not an image."
        )
        
    try:
        image_bytes = await file.read()
        logger.info(f"User {user_id} requested prescription scan. Size: {len(image_bytes)} bytes.")
        
        # 1. Run Pure Gemini Vision Model
        logger.info("Analyzing image via Gemini Vision...")
        structured_order = analyze_prescription(image_bytes)
        
        return {
            "success": True,
            "user_id": user_id,
            "raw_ocr_text": "N/A (Using Pure Gemini Vision)",
            "order": structured_order
        }
        
    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal AI processing error: {str(e)}"
        )
