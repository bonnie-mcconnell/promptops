import os
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
import secrets


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(provided_key: str = Depends(api_key_header)):
    expected_key = os.environ.get("API_KEY")
    if not expected_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
