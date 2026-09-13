from pydantic import BaseModel

class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float