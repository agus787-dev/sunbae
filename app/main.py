"""FastAPI service for Sunbae.

Run from the project root:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs to test it in the browser.
"""

import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.generator import Generator

# Frontend addresses that are allowed to call this API (comma separated in the env variable)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load the model into memory
    app.state.generator = Generator()
    yield


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Allow only requests that send the correct X-API-Key header."""
    if not INTERNAL_API_KEY:
        return  # key not configured: open mode, for local development only
    if x_api_key is None or not secrets.compare_digest(x_api_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


class GenerateRequest(BaseModel):
    """Body of POST /generate. FastAPI returns 422 if a value is out of range."""

    prompt: str = Field(min_length=1, max_length=1000)
    max_tokens: int = Field(default=100, ge=1, le=300)
    temperature: float = Field(default=0.8, ge=0, le=2)   # 0 = greedy
    top_k: int = Field(default=50, ge=0, le=500)          # 0 = off
    top_p: float = Field(default=0.95, ge=0, le=1)        # 1 = off


app = FastAPI(title="Sunbae API", lifespan=lifespan)

# Without CORS the browser blocks requests from the frontend to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Simple check that the server is running and which model is loaded."""
    generator = app.state.generator
    return {
        "status": "ok",
        "parameters": generator.num_parameters(),
        "block_size": generator.config.block_size,
        "step": generator.step,
        "val_loss": round(generator.val_loss, 4),
    }


# A normal "def" (not "async def"): generation is heavy work,
# so FastAPI runs it in a thread pool and the server stays responsive.
@app.post("/generate", dependencies=[Depends(verify_api_key)])
def generate(request: GenerateRequest):
    """Continue the prompt and return the text, timing and tokens."""
    return app.state.generator.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
