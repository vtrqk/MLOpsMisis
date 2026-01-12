from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.v1.sentiment import router as sentiment_router

app = FastAPI(title="Sentiment Service")

app.include_router(sentiment_router, prefix="/api/v1", tags=["sentiment"])

templates = Jinja2Templates(directory="app/web/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
