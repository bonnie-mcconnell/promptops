from pydantic import BaseModel, Field


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
