import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from openai import APIStatusError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import time
import os

from database import init_db, log_request, get_db
from service import optimize_prompt, optimize_prompt_mock, _cache_key, redis_client, CircuitOpenError


USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "false").lower() == "true"
optimizer = optimize_prompt_mock if USE_MOCK_LLM else optimize_prompt


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str = Field(
        description="The original prompt to optimize"
    )
    goal: str = Field(
        description="What the prompt should accomplish"
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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/optimize", response_model=PromptResponse)
def optimize_prompt_endpoint(request: PromptRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()
    key = _cache_key(request.prompt, request.goal)
    was_cached = redis_client.get(key) is not None

    status = "ok"
    error_detail = None
    http_error = None
    result = None

    try:
        result = optimizer(request.prompt, request.goal)
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
        log_request(db, request.prompt, request.goal, key, was_cached, latency_ms, status, error_detail)

    if http_error:
        raise http_error

    assert result is not None
    return PromptResponse(**result)