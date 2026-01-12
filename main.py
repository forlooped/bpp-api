"""
BackPocket Power API - Should Escalate Endpoint
Production-ready FastAPI application
"""

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import sqlite3
from datetime import datetime
import secrets

# OpenAI v1-style client
from openai import OpenAI

# Stripe
import stripe

app = FastAPI(
    title="BackPocket Power API",
    description="AI-powered escalation detection for customer messages",
    version="1.1.0"
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

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")
client = OpenAI(api_key=OPENAI_API_KEY)

# Stripe config
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "your-stripe-secret-here")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "your-webhook-secret-here")

# Your own free/admin identity
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "you@backpocketpower.com")
ADMIN_KEY_PREFIX = os.getenv("ADMIN_KEY_PREFIX", "sk_admin_")

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
# HELPERS FOR ADMIN / FREE KEY
# ============================================================================

def is_free_key_record(row: tuple) -> bool:
    """
    row = (key, customer_email, calls_limit, calls_used, created_at, active)
    Treat keys as free/unlimited if:
    - email == ADMIN_EMAIL, or
    - key starts with ADMIN_KEY_PREFIX
    """
    if not row:
        return False
    key, customer_email, *_rest = row
    if customer_email == ADMIN_EMAIL:
        return True
    if key.startswith(ADMIN_KEY_PREFIX):
        return True
    return False

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

    # Fetch full row so we can detect free/admin keys
    c.execute("SELECT key, customer_email, calls_limit, calls_used, created_at, active FROM api_keys WHERE key = ?", (api_key,))
    row = c.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid API key")

    key, customer_email, calls_limit, calls_used, created_at, active = row

    if not active:
        conn.close()
        raise HTTPException(status_code=403, detail="API key has been deactivated")

    # Free/admin keys bypass quota checks
    if not is_free_key_record(row):
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
        import json

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

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

    # Increment usage counter, unless it's a free/admin key
    c.execute("SELECT key, customer_email, calls_limit, calls_used, created_at, active FROM api_keys WHERE key = ?", (api_key,))
    row = c.fetchone()

    if row and not is_free_key_record(row):
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
        "version": "1.1.0",
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

    **Rate limiting:** Based on your plan (1000/3000/10000 calls).
    Free/admin keys bypass quota but still log usage.
    """

    try:
        # Analyze the message
        result = analyze_message(request.message, request.context)

        # Log usage
        log_usage(api_key, "/should-escalate", True)

        return EscalationResponse(**result)

    except Exception as e:
        # Log failed attempt
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
        "percentage_used": round((calls_used / calls_limit) * 100, 2) if calls_limit > 0 else None
    }

# ============================================================================
# ADMIN FUNCTIONS (for you to create API keys)
# ============================================================================

def create_api_key(customer_email: str, calls_limit: int = 1000, prefix: Optional[str] = "sk_") -> str:
    """
    Create a new API key for a customer
    """

    api_key = f"{prefix}{secrets.token_urlsafe(32)}"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "INSERT INTO api_keys (key, customer_email, calls_limit, created_at) VALUES (?, ?, ?, ?)",
        (api_key, customer_email, calls_limit, datetime.utcnow().isoformat())
    )

    conn.commit()
    conn.close()

    return api_key

# ============================================================================
# ADMIN ENDPOINT (for manual / bootstrap)
# ============================================================================

@app.post("/admin/create-key")
def admin_create_key(
    email: str,
    calls_limit: int = 1000,
    admin_secret: str = Header(None, alias="X-Admin-Secret"),
    admin_prefix: Optional[str] = None,
):
    """
    Admin endpoint to create API keys
    Requires X-Admin-Secret header for security.
    If admin_prefix is provided, that prefix is used for the key (e.g. "sk_admin_").
    """
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change-me-in-production")

    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    prefix = admin_prefix if admin_prefix else "sk_"
    api_key = create_api_key(email, calls_limit, prefix=prefix)

    return {
        "api_key": api_key,
        "email": email,
        "calls_limit": calls_limit,
        "created_at": datetime.utcnow().isoformat()
    }

# ============================================================================
# STRIPE WEBHOOK (auto-create keys after payment)
# ============================================================================

def map_price_to_quota(price_id: Optional[str], session: dict) -> Optional[int]:
    small = os.getenv("PRICE_ID_SMALL")
    medium = os.getenv("PRICE_ID_MEDIUM")
    large = os.getenv("PRICE_ID_LARGE")

    price_map = {}
    if small:
        price_map[small] = 1000
    if medium:
        price_map[medium] = 3000
    if large:
        price_map[large] = 10000

    if price_id and price_id in price_map:
        return price_map[price_id]

    amount_cents = session.get("amount_total")
    if amount_cents:
        return int(amount_cents)

    return None

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe webhook endpoint.
    Creates an API key automatically when a checkout.session.completed event fires.
    """
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        customer_email = session.get("customer_details", {}).get("email")
        # If you set metadata on the Checkout Session with price_id
        price_id = None
        if session.get("metadata"):
            price_id = session["metadata"].get("price_id")

        calls_limit = map_price_to_quota(price_id, session)

        if customer_email and calls_limit:
            # Normal user keys use "sk_" prefix
            api_key = create_api_key(customer_email, calls_limit, prefix="sk_")
            # TODO: send email to customer_email with api_key
            # For now you could log it, or later integrate with a mail provider.

    return {"received": True}

class CheckoutRequest(BaseModel):
    price_id: str
    success_url: Optional[str] = "https://backpocketpower.com/success"
    cancel_url: Optional[str] = "https://backpocketpower.com/cancel"

@app.post("/create-checkout")
def create_checkout_session(request: CheckoutRequest):
    """
    Create a Stripe Checkout session for testing
    Returns a URL you can visit to complete payment
    """
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": request.price_id,
                "quantity": 1,
            }],
            mode="payment",
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            customer_email=None,  # Customer enters their email
            metadata={
                "price_id": request.price_id  # Pass price_id to webhook
            }
        )
        
        return {
            "checkout_url": session.url,
            "session_id": session.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/list-keys")
def admin_list_keys(
    admin_secret: str = Header(None, alias="X-Admin-Secret"),
    limit: int = 10
):
    """List recent API keys (admin only)"""
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change-me-in-production")
    
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT key, customer_email, calls_limit, calls_used, created_at FROM api_keys ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    
    return {
        "keys": [
            {
                "key": row[0],
                "email": row[1],
                "calls_limit": row[2],
                "calls_used": row[3],
                "created_at": row[4]
            }
            for row in rows
        ]
    }
