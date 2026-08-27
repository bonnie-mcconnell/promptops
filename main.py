from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from openai import APIStatusError
from sqlalchemy.orm import Session
import time
import os

from models import PromptRequest, PromptResponse, CompareRequest, CompareResponse, StatsResponse
from database import init_db, log_request, get_db, get_stats
from service import optimize_prompt, optimize_prompt_mock 
from resilience import CircuitOpenError
from cache import create_vector_index, cache_key
from eval import judge_output
from auth import verify_api_key


logger = logging.getLogger(__name__)

USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "false").lower() == "true"
optimizer = optimize_prompt_mock if USE_MOCK_LLM else optimize_prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    create_vector_index()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/optimize", response_model=PromptResponse)
def optimize_prompt_endpoint(request: PromptRequest, _: None = Depends(verify_api_key), db: Session = Depends(get_db)):
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
def compare_prompts(request: CompareRequest, _: None = Depends(verify_api_key)):
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
def get_stats_endpoint(_: None = Depends(verify_api_key), db: Session = Depends(get_db)):
    return get_stats(db)