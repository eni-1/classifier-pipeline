import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Settings(BaseSettings):
    model_path: str = "./final_model_artifacts"
    max_length: int = 256
    model_version: str = "1.0.1"
    host: str = "0.0.0.0"
    port: int = 8000
    class Config:
        env_file = ".env"
settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)
ml_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up API")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ml_assets["device"] = device
    logger.info(f"Using: {device}")
    
    logger.info(f"Loading model from: {settings.model_path}")
    ml_assets["tokenizer"] = AutoTokenizer.from_pretrained(settings.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
    model.to(device)
    model.eval()
    ml_assets["model"] = model
    
    logger.info("Loading complete")
    yield
    logger.info("Shutting down API")
    ml_assets.clear()

app = FastAPI(
    title="arXiv Research Paper Classifier API",
    description="API for classifying arXiv paper abstracts.",
    version="1.0.0",
    lifespan=lifespan,
)

class Paper(BaseModel):
    abstract: str = Field(..., min_length=50, max_length=5000)
    
    @field_validator('abstract')
    def validate_abstract(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Abstract cannot be empty or contain only whitespace.")
        return v.strip()

class Prediction(BaseModel):
    category: str
    confidence: float
    all_probabilities: dict[str, float]
    model_version: str

@app.get("/health", tags=["General"])
def health_check() -> dict[str, str]:
    if "model" in ml_assets and "tokenizer" in ml_assets:
        return {"status": "ok", "model_loaded": "true"}
    raise HTTPException(status_code=503, detail="Model not loaded yet")

@app.post("/v1/predict", response_model=Prediction, tags=["Prediction"])
def predict_category(paper: Paper) -> Prediction:
    logger.info(f"Received prediction request for abstract of length {len(paper.abstract)}.")
    
    try:
        tokenizer = ml_assets.get("tokenizer")
        model = ml_assets.get("model")
        device = ml_assets.get("device")
        if not all([tokenizer, model, device]):
            raise HTTPException(status_code=503, detail="Model is not available.")
        inputs = tokenizer(
            paper.abstract,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_length,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_id = probabilities.argmax().item()
        confidence = probabilities[predicted_id].item()
        category = model.config.id2label[predicted_id]
        all_probs = {model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities)}
        logger.info(f"Prediction successful: {category} (Confidence: {confidence:.4f})")
        return Prediction(
            category=category,
            confidence=confidence,
            all_probabilities=all_probs,
            model_version=settings.model_version
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)