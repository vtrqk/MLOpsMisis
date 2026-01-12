from fastapi import APIRouter, HTTPException

from app.core.model import SentimentModel
from app.schemas.sentiment import SentimentRequest, SentimentResponse

router = APIRouter()
model = SentimentModel()


@router.post("/predict", response_model=SentimentResponse)
async def predict(payload: SentimentRequest):
    try:
        label, probs = model.predict(payload.text)
        return SentimentResponse(label=label, probabilities=probs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
