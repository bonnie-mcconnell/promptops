# PromptOps

A prompt-optimization API that started as a FastAPI tutorial exercise and grew into a small exploration of how LLM-backed services behave in production: caching, retries, circuit breakers, observability, and a stastical A/B testing framework for deciding whether a change to a system message is actually an improvement, or just noise.

Given a prompt and a goal, this service returns an optimized version of that prompt, an exploration of what changed, caches aggressively (with both exact match and semantic caching), degrades gracefully when the upstream LLM is unreliable, and logs enough about every request to answer 'what happened, and why' after the fact.

This project is interested in the problems that start after 'call the API, format the output'. The upstream call is unreliable and not cheap, a naive cache only catches literal duplicate traffic, and "we changed the prompt and it feels better" isn't an engineering claim but a guess. This project is an attempt to actually address those three things, not just the happy path.

## Architecture

```
Client
  │
  ▼
FastAPI (/optimize)
  │
  ├─► Exact-match cache (Redis, SHA-256 key)
  │
  ├─► Semantic cache (Redis Stack + RediSearch, KNN cosine similarity
  │     over OpenAI embeddings which catches the near-duplicate prompts an
  │     exact-match cache misses)
  │
  ├─► LLM call (OpenAI), wrapped in:
  │     - exponential backoff retries (transient failure)
  │     - a circuit breaker (sustained failure, fails fast instead
  │       of retrying against a dead upstream)
  │
  └─► Every request (hit, miss, or error) logged to Postgres:
        prompt, goal, cache_type, latency, status, error detail
```

A seperate evaluation harness (`eval.py`)~runs a fixed, hand built prompt set through two system message variants, scores each output with an LLM judge, and compares the two with a paired statistical test.

**Stack:** FastAPI · Redis Stack (cache + vector search) · PostgreSQL (trace log) ·
Docker Compose · pytest · scipy

## Running it

```bash
cp .env.example .env   # fill in POSTGRES_PASSWORD, OPENAI_API_KEY optional
docker compose up --build
```

The API comes up at `http://localhost:8000`. Interactive docs at `/docs`.

```bash
docker compose exec api pytest -v
```

### Mock mode

`USE_MOCK_LLM=true` (the default in `.env.example`) swaps the real OpenAI call for
a deterministic mock that adds persona framing without hitting any external API.
The entire request path (caching, retries, the circuit breaker's error handling,
Postgres logging) runs identically either way, since the switch happens at the
one call site (`main.py`), not throughout the codebase. This means the whole
system is runnable and testable with zero API cost. Flipping to a real key only
changes which function actually talks to OpenAI.
 
Set `USE_MOCK_LLM=false` and provide a real `OPENAI_API_KEY` to use the real
model. The evaluation harness (`scripts/run_eval.py`) always calls the real API
directly, regardless of this flag, since there's nothing meaningful to evaluate
against a mock.

## Design Decisions

**Three circuit breakers** Chat completions, embeddings and LLM judging are
three seperate OpenAI calls to different models that can each fail independently. A single shared
breaker would mean an embeddings outage incorrectly blocks chat completions
too, or that a judge outage incorrectly blocks the main optimization path, even though the two aren't related failures. Each upstream dependency gets its own breaker.

**FLAT vector index, not HNSW.** RediSearch supports both. HNSW is faster at
scale but approximate, meaning it can miss the true nearest neighbor in exchange for
speed. At this project's actual data volume (a cache, not a corpus), FLAT's
exact brute-force search costs single-digit milliseconds. Using HNSW would mean trading correctness for a speedup that would never be realized.

**Wilcoxon signed-rank test, not a paired t-test**, for comparing prompt
variants. A t-test assumes the paired differences are roughly normally
distributed, which is reasonable for large samples of continuous data, but less reliable for a bounded 1–10 LLM-judge scale with a modest sample size as in this project. Wilcoxon works on the ranks of the differences instead of their raw magnitudes, which is more robust to that shape of data. The harness
reports both a p-value and a median-difference effect size, since a statistically detectable difference and a practically meaningful one aren't the same.

**The LLM judge is a different model than the one being evaluated**
(`gpt-4o` judging `gpt-4o-mini` output), is done specifically to reduce
self-preference bias (the documented tendency of LLM judges to rate their
own model family's output more favorably). Which variant is shown to the judge
first is also randomized per comparison, to control for judge position bias
(judges favouring whichever option
appears first, independent of quality). It would be a stronger control to have a fully independent judge from a different provider,  than just a different model from the same provider.

**Semantic caching is a second, parallel cache, not a replacement for
exact-match.** Exact-match is cheaper (a plain key lookup, no embedding call)
and exact, so it's checked first. Semantic search only runs on an exact-match
miss.

## Evaluation harness
 
`eval.py` takes two system-message variants, currently set to the shipped
persona-framing prompt against a bare-minimum control, comparing them across a 16-prompt
hand-built set deliberately spanning creative, informational, technical,
business, and persuasive tasks, and both vague and tightly-specified goals.
For each prompt, both variants are run, an LLM judge scores each output
1–10 against an anchored rubric (5 = no better than doing nothing), and the
paired scores are compared with `scipy.stats.wilcoxon`.
 
Run it directly (this uses API credits, is not gated by `USE_MOCK_LLM`):
 
```bash
docker compose exec api python scripts/run_eval.py
```

### Comparing your own candidate prompts

`judge_output` - the same LLM-judge primitive the batch harness uses internally
- is also exposed directly as `POST /compare`: submit two candidate prompts and
a goal to get back which one the judge scored higher and why. A single
comparison doesn't warrant a statistical test (that's what the batch harness
above is for), but it's a direct, practical use of the same judging logic for
anyone who wants to A/B test their own prompt wording rather than trust
either variant blindly.

## Testing
 
Unit tests for the circuit breaker in isolation, the LLM call and embedding
call with mocked OpenAI responses, the semantic cache's similarity threshold
using deliberately constructed identical and orthogonal test vectors (so
correctness doesn't depend on a real embedding model's actual judgment), and
the evaluation harness's full flow, partial-failure handling, and
statistical-validity guard.
 
```bash
docker compose exec api pytest -v
```

## What I'd add next
 
- Basic API-key auth and per-key rate limiting
- Distributed circuit breaker state (currently in-process, wouldn't
  coordinate correctly across multiple replicas)
- A second, independent judge provider for a stronger self-preference-bias
  control
- A minimal frontend for trying `/optimize` and `/compare` without curl or `/docs`