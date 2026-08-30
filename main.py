from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import APIStatusError
from sqlalchemy.orm import Session
from uuid import uuid4
import time
import os

from models import PromptRequest, PromptResponse, CompareRequest, CompareResponse, StatsResponse
from database import init_db, log_request, get_db, get_stats
from service import optimize_prompt, optimize_prompt_mock 
from resilience import CircuitOpenError
from cache import create_vector_index, cache_key
from eval import judge_output, judge_output_mock
from auth import verify_api_key
from ratelimit import check_rate_limit

logger = logging.getLogger(__name__)

USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "false").lower() == "true"
optimizer = optimize_prompt_mock if USE_MOCK_LLM else optimize_prompt
judge = judge_output_mock if USE_MOCK_LLM else judge_output


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    create_vector_index()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "https://promptops-frontend.onrender.com"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/optimize", response_model=PromptResponse)
def optimize_prompt_endpoint(request: PromptRequest, response: Response, _: None = Depends(verify_api_key), __: None = Depends(check_rate_limit), db: Session = Depends(get_db)):
    start = time.perf_counter()

    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id  

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
        http_error = HTTPException(status_code=502, detail="Upstream LLM provider error. Please retry.", headers={"X-Request-ID": request_id})
    except ValueError as e:
        status, error_detail = "error", str(e)
        http_error = HTTPException(status_code=502, detail="Invalid upstream LLM response. Please retry.", headers={"X-Request-ID": request_id})
    except CircuitOpenError as e:
        status, error_detail = "error", str(e)
        http_error = HTTPException(status_code=503, detail="Service temporarily unavailable, please retry shortly.", headers={"X-Request-ID": request_id})
    except Exception as e:
        status, error_detail = "error", str(e)
        http_error = HTTPException(status_code=500, detail="Internal server error. Please try again.", headers={"X-Request-ID": request_id})
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(db, request_id, request.prompt, request.goal, key, cache_type, latency_ms, status, error_detail)
        logger.info(f"request_id={request_id} status={status} cache_type={cache_type} latency_ms={latency_ms}")

    if http_error:
        raise http_error

    assert result is not None
    return PromptResponse(**result)

@app.post("/compare", response_model=CompareResponse)
def compare_prompts(request: CompareRequest, response: Response, _: None = Depends(verify_api_key), __: None = Depends(check_rate_limit)):
    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id
        
    try:
        result = judge(request.prompt, request.goal, request.candidate_a, request.candidate_b)
    except Exception:
        raise HTTPException(status_code=502, detail="Judge evaluation failed. Please retry.", headers={"X-Request-ID": request_id})

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