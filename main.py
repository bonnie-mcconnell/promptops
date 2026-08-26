from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from openai import APIStatusError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import time
import os

from database import init_db, log_request, get_db, get_stats
from service import optimize_prompt, optimize_prompt_mock 
from resilience import CircuitOpenError
from cache import create_vector_index, cache_key
from eval import judge_output


USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "false").lower() == "true"
optimizer = optimize_prompt_mock if USE_MOCK_LLM else optimize_prompt


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    create_vector_index()
    yield

app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str = Field(
        description="The original prompt to optimize",
        min_length=1,
        max_length=4000,
    )
    goal: str = Field(
        description="What the prompt should accomplish",
        min_length=1,
        max_length=4000,
    )
    model_config = {"extra": "forbid"}


class PromptResponse(BaseModel):
    original_prompt: str = Field(
        description="The original prompt that was submitted"
    )
    optimized_prompt: str = Field(
        description="The improved version of the prompt"
    )
    changes: str = Field(
        description="Explanation of what was improved and why"
    )


class CompareRequest(BaseModel):
    prompt: str
    goal: str
    candidate_a: str = Field(
        description="First candidate prompt to compare",
        min_length=1,
        max_length=4000,
    )
    candidate_b: str = Field(
        description="Second candidate prompt to compare",
        min_length=1,
        max_length=4000,
    )
    model_config = {"extra": "forbid"}


class CompareResponse(BaseModel):
    score_a: float
    score_b: float
    winner: str
    reasoning: str


class StatsResponse(BaseModel):
    total: int
    exact_hits: int
    semantic_hits: int
    cache_misses: int
    errors: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/optimize", response_model=PromptResponse)
def optimize_prompt_endpoint(request: PromptRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()
    key = cache_key(request.prompt, request.goal)

    status = "ok"
    cache_type = "none"
    error_detail = None
    http_error = None
    result = None

    try:
        result, cache_type = optimizer(request.prompt, request.goal)
    except APIStatusError as e:
        status, error_detail = "error", str(e)
        http_error = HTTPException(status_code=502, detail="Upstream LLM provider error. Please retry.")
    except ValueError as e:
        status, error_detail = "error", str(e)
        http_error = HTTPException(status_code=502, detail="Invalid upstream LLM response. Please retry.")
    except CircuitOpenError as e:
        status, error_detail = "error", str(e)
        http_error = HTTPException(status_code=503, detail="Service temporarily unavailable, please retry shortly.")
    except Exception as e:
        status, error_detail = "error", str(e)
        http_error = HTTPException(status_code=500, detail="Internal server error. Please try again.")
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(db, request.prompt, request.goal, key, cache_type, latency_ms, status, error_detail)

    if http_error:
        raise http_error

    assert result is not None
    return PromptResponse(**result)

@app.post("/compare", response_model=CompareResponse)
def compare_prompts(request: CompareRequest):
    try:
        result = judge_output(request.prompt, request.goal, request.candidate_a, request.candidate_b)
    except Exception:
        raise HTTPException(status_code=502, detail="Judge evaluation failed. Please retry.")

    winner = "a" if result["score_a"] > result["score_b"] else "b" if result["score_b"] > result["score_a"] else "tie"

    return CompareResponse(
        score_a=result["score_a"],
        score_b=result["score_b"],
        winner=winner,
        reasoning=result["reasoning"]
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats_endpoint(db: Session = Depends(get_db)):
    return get_stats(db)