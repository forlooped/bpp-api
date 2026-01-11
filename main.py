"""
BackPocket Power API - Should Escalate Endpoint
Production-ready FastAPI application
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import openai
import os
import sqlite3
from datetime import datetime
import secrets

app = FastAPI(
    title="BackPocket Power API",
    description="AI-powered escalation detection for customer messages",
    version="1.0.0"
)

# CORS - allows browsers to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set your OpenAI API key here (or use environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-key-here")

# Database for tracking usage
DB_PATH = "api_usage.db"

# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_db():
    """Initialize SQLite database for API key tracking"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # API keys table
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
        key TEXT PRIMARY KEY,
        customer_email TEXT,
        calls_limit INTEGER,
        calls_used INTEGER DEFAULT 0,
        created_at TEXT,
        active INTEGER DEFAULT 1
    )''')

    # Usage log table
    c.execute('''CREATE TABLE IF NOT EXISTS usage_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT,
        timestamp TEXT,
        endpoint TEXT,
        success INTEGER
    )''')

    conn.commit()
    conn.close()

init_db()

# ============================================================================
# MODELS (Request/Response Schemas)
# ============================================================================

class EscalationRequest(BaseModel):
    message: str = Field(..., description="The customer message to analyze")
    context: Optional[str] = Field(None, description="Optional context about the conversation")

class EscalationResponse(BaseModel):
    should_escalate: bool = Field(..., description="Whether this message needs human attention")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    reason: str = Field(..., description="Why this decision was made")

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# ============================================================================
# API KEY VALIDATION
# ============================================================================

def validate_api_key(authorization: str = Header(None)):
    """Validate API key and check usage limits"""

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Extract key from "Bearer <key>" format
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Use: Bearer <your-key>")

    api_key = authorization.replace("Bearer ", "")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if key exists and is active
    c.execute("SELECT calls_limit, calls_used, active FROM api_keys WHERE key = ?", (api_key,))
    result = c.fetchone()

    if not result:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid API key")

    calls_limit, calls_used, active = result

    if not active:
        conn.close()
        raise HTTPException(status_code=403, detail="API key has been deactivated")

    if calls_used >= calls_limit:
        conn.close()
        raise HTTPException(
            status_code=402,
            detail=f"Usage limit reached ({calls_used}/{calls_limit} calls). Please upgrade."
        )

    conn.close()
    return api_key

# ============================================================================
# CORE API LOGIC
# ============================================================================

def analyze_message(message: str, context: Optional[str] = None) -> dict:
    """Call OpenAI to analyze if message should escalate"""

    system_prompt = """You are an expert at detecting when customer service messages need human escalation.

Analyze the message and return JSON with:
- should_escalate (boolean): true if needs human attention
- confidence (float 0-1): how certain you are
- reason (string): brief explanation

Escalate if message shows:
- High frustration or anger
- Legal threats or compliance issues
- Safety concerns
- Complex problems needing human judgment
- Explicit requests for human help
- Payment disputes or refund demands

Do NOT escalate routine questions, simple requests, or friendly messages."""

    user_message = f"Message: {message}"
    if context:
        user_message += f"\n\nContext: {context}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Fast and cheap for this use case
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        import json
        result = json.loads(response.choices[0].message.content)

        # Ensure all required fields exist
        return {
            "should_escalate": result.get("should_escalate", False),
            "confidence": float(result.get("confidence", 0.5)),
            "reason": result.get("reason", "Analysis completed")
        }

    except Exception as e:
        # Fallback to safe default
        return {
            "should_escalate": True,
            "confidence": 0.5,
            "reason": f"Error in analysis: {str(e)}"
        }

def log_usage(api_key: str, endpoint: str, success: bool):
    """Log API usage and increment counter"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Log the call
    c.execute(
        "INSERT INTO usage_log (api_key, timestamp, endpoint, success) VALUES (?, ?, ?, ?)",
        (api_key, datetime.utcnow().isoformat(), endpoint, 1 if success else 0)
    )

    # Increment usage counter
    c.execute("UPDATE api_keys SET calls_used = calls_used + 1 WHERE key = ?", (api_key,))

    conn.commit()
    conn.close()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "BackPocket Power API",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/should-escalate", response_model=EscalationResponse)
def should_escalate(
    request: EscalationRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Analyze a customer message to determine if it needs human escalation.

    **Authentication:** Bearer token in Authorization header

    **Rate limiting:** Based on your plan (1000/3000/10000 calls)

    **Returns 402** when usage limit is reached
    """

    try:
        # Analyze the message
        result = analyze_message(request.message, request.context)

        # Log successful usage
        log_usage(api_key, "/should-escalate", True)

        return EscalationResponse(**result)

    except Exception as e:
        # Log failed attempt (still counts against quota)
        log_usage(api_key, "/should-escalate", False)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usage")
def check_usage(api_key: str = Depends(validate_api_key)):
    """Check your current API usage"""

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT calls_limit, calls_used FROM api_keys WHERE key = ?",
        (api_key,)
    )
    result = c.fetchone()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="API key not found")

    calls_limit, calls_used = result

    return {
        "calls_used": calls_used,
        "calls_limit": calls_limit,
        "calls_remaining": calls_limit - calls_used,
        "percentage_used": round((calls_used / calls_limit) * 100, 2)
    }

# ============================================================================
# ADMIN FUNCTIONS (for you to create API keys)
# ============================================================================

def create_api_key(customer_email: str, calls_limit: int = 1000) -> str:
    """
    Create a new API key for a customer

    Run this function manually when someone pays:

    from main import create_api_key
    key = create_api_key("customer@example.com", 1000)
    print(f"API Key: {key}")
    """

    api_key = f"sk_{secrets.token_urlsafe(32)}"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "INSERT INTO api_keys (key, customer_email, calls_limit, created_at) VALUES (?, ?, ?, ?)",
        (api_key, customer_email, calls_limit, datetime.utcnow().isoformat())
    )

    conn.commit()
    conn.close()

    return api_key

# To manually create a key, run in Python:
# from main import create_api_key
# print(create_api_key("test@example.com", 1000))
