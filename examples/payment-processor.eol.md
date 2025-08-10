---
name: payment-processor
version: 1.0.0
phase: hybrid
tags: [payments, stripe, core]
tests: payment-processor.test.eol.md
dependencies:
  features:
    - path: auth/authentication.eol.md
      version: "^2.0.0"
      phase: all
      inject:
        - authenticate_user
        - validate_api_key
    - path: core/rate-limiting.eol.md
      version: "~1.5.0"
      phase: all
      config:
        max_requests: 100
        window_seconds: 60
  
  mcp_servers:
    - name: redis-mcp
      version: ">=1.0.0"
      transport: stdio
      config:
        url: ${REDIS_URL:-redis://localhost:6379}
      fallback: redis-direct
      phase: prototyping
  
  services:
    - name: stripe-api
      type: rest
      url: ${STRIPE_API_URL:-https://api.stripe.com}
      version: "2024-01-01"
      auth:
        type: bearer
        token: ${STRIPE_API_KEY}
      circuit_breaker:
        failure_threshold: 5
        timeout: 60
      phase: implementation
  
  packages:
    - name: stripe
      version: "^7.0.0"
      phase: implementation
    - name: redis[vector]
      version: ">=5.0.0"
      phase: all
  
  containers:
    - name: redis
      image: redis/redis-stack:latest
      ports:
        - "6379:6379"
      phase: all
  
  models:
    - name: claude-3-opus
      provider: anthropic
      version: "20240229"
      purpose: fraud-detection
      config:
        temperature: 0.3
        max_tokens: 2048
      phase: prototyping
    - name: gpt-4-turbo
      provider: openai
      purpose: customer-communication
      config:
        temperature: 0.7
      fallback: gpt-3.5-turbo
      phase: all
---

# Payment Processor

## Description
Advanced payment processing system with fraud detection, rate limiting, and customer communication capabilities. Supports both prototyping with natural language and production implementation with Stripe API.

## Requirements
- Process payments securely using Stripe API
- Detect fraudulent transactions using AI
- Rate limit payment attempts per user
- Cache payment methods for returning customers
- Generate customer receipts and notifications
- Support refunds and partial refunds
- Maintain PCI compliance

## Context
- @context/patterns/payment-processing.md
- @context/security/pci-compliance.md
- @examples/stripe-integration.py

## Prototyping
```natural
When a payment is requested:
  1. Authenticate the user using the injected auth feature
  2. Check rate limits for the user
  3. Retrieve cached payment method from Redis if available
  4. Analyze transaction for fraud using Claude-3-Opus:
     - Check unusual patterns
     - Verify billing/shipping address match
     - Score risk level (0-100)
  5. If risk score > 70:
     - Flag for manual review
     - Send notification to admin
  6. Process payment through appropriate gateway
  7. Store transaction details in Redis with 30-day TTL
  8. Generate and send receipt using GPT-4-Turbo
  
Handle payment errors:
  - Log detailed error information
  - Return user-friendly error message
  - Implement exponential backoff for retries
```

## Implementation
```python
import stripe
from typing import Dict, Optional, List
from decimal import Decimal
import asyncio
from redis import asyncio as aioredis

class PaymentProcessor:
    def __init__(self, redis_client: aioredis.Redis, auth_service, rate_limiter):
        self.redis = redis_client
        self.auth = auth_service
        self.rate_limiter = rate_limiter
        stripe.api_key = os.getenv('STRIPE_API_KEY')
    
    async def process_payment(
        self,
        user_id: str,
        amount: Decimal,
        currency: str = "usd",
        payment_method_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Process a payment transaction"""
        
        # Authenticate user (injected from auth feature)
        user = await self.auth.authenticate_user(user_id)
        if not user:
            raise AuthenticationError("Invalid user")
        
        # Check rate limits (injected from rate-limiting feature)
        if not await self.rate_limiter.check_limit(user_id, "payment"):
            raise RateLimitError("Too many payment attempts")
        
        # Check for cached payment method
        if not payment_method_id:
            cached_method = await self.redis.get(f"payment_method:{user_id}")
            if cached_method:
                payment_method_id = cached_method.decode()
        
        # Create payment intent
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency,
                payment_method=payment_method_id,
                customer=user.stripe_customer_id,
                metadata=metadata or {},
                confirm=True
            )
            
            # Cache successful payment method
            if intent.status == "succeeded":
                await self.redis.setex(
                    f"payment_method:{user_id}",
                    86400 * 30,  # 30 days
                    payment_method_id
                )
                
                # Store transaction
                await self.store_transaction(user_id, intent)
            
            return {
                "status": intent.status,
                "transaction_id": intent.id,
                "amount": amount,
                "currency": currency
            }
            
        except stripe.error.StripeError as e:
            await self.handle_stripe_error(e, user_id)
            raise
    
    async def detect_fraud(self, transaction: Dict) -> Dict:
        """Analyze transaction for fraud (implementation phase)"""
        
        # In implementation phase, use rule-based detection
        risk_score = 0
        flags = []
        
        # Check amount threshold
        if transaction['amount'] > 10000:
            risk_score += 30
            flags.append("high_amount")
        
        # Check velocity
        recent_count = await self.redis.get(f"tx_count:{transaction['user_id']}:1h")
        if recent_count and int(recent_count) > 5:
            risk_score += 40
            flags.append("high_velocity")
        
        # Check country mismatch
        if transaction.get('billing_country') != transaction.get('shipping_country'):
            risk_score += 20
            flags.append("country_mismatch")
        
        return {
            "risk_score": min(risk_score, 100),
            "flags": flags,
            "require_review": risk_score > 70
        }
    
    async def store_transaction(self, user_id: str, intent):
        """Store transaction details in Redis"""
        
        transaction_key = f"transaction:{intent.id}"
        transaction_data = {
            "user_id": user_id,
            "amount": intent.amount,
            "currency": intent.currency,
            "status": intent.status,
            "created_at": intent.created,
            "payment_method": intent.payment_method
        }
        
        # Store with 30-day TTL
        await self.redis.hset(transaction_key, mapping=transaction_data)
        await self.redis.expire(transaction_key, 86400 * 30)
        
        # Update user transaction list
        await self.redis.lpush(f"user_transactions:{user_id}", intent.id)
        await self.redis.ltrim(f"user_transactions:{user_id}", 0, 99)  # Keep last 100
        
        # Update velocity counter
        await self.redis.incr(f"tx_count:{user_id}:1h")
        await self.redis.expire(f"tx_count:{user_id}:1h", 3600)
    
    async def refund_payment(
        self,
        transaction_id: str,
        amount: Optional[Decimal] = None,
        reason: Optional[str] = None
    ) -> Dict:
        """Process a refund for a transaction"""
        
        try:
            refund_params = {
                "payment_intent": transaction_id,
                "reason": reason or "requested_by_customer"
            }
            
            if amount:
                refund_params["amount"] = int(amount * 100)
            
            refund = stripe.Refund.create(**refund_params)
            
            # Update transaction status in Redis
            await self.redis.hset(
                f"transaction:{transaction_id}",
                "refund_status",
                refund.status
            )
            
            return {
                "refund_id": refund.id,
                "status": refund.status,
                "amount": refund.amount / 100
            }
            
        except stripe.error.StripeError as e:
            raise RefundError(f"Refund failed: {str(e)}")
```

## Operations
```yaml
operations:
  - name: process_payment
    phase: implementation
    function: process_payment
    input:
      - user_id: string
      - amount: decimal
      - currency: string
      - payment_method_id: string
    output:
      type: object
      properties:
        status: string
        transaction_id: string
    
  - name: detect_fraud
    phase: hybrid
    prototyping:
      description: "Analyze transaction with Claude-3-Opus for fraud patterns"
      model: claude-3-opus
    implementation:
      function: detect_fraud
    input:
      - transaction: object
    output:
      type: object
      properties:
        risk_score: integer
        flags: array
        require_review: boolean
  
  - name: generate_receipt
    phase: prototyping
    description: "Generate customer receipt using GPT-4-Turbo"
    model: gpt-4-turbo
    input:
      - transaction: object
      - customer: object
    output:
      type: string
      format: html
  
  - name: refund_payment
    phase: implementation
    function: refund_payment
    input:
      - transaction_id: string
      - amount: decimal
      - reason: string
    output:
      type: object
      properties:
        refund_id: string
        status: string
```

## Configuration
```yaml
stripe:
  api_version: "2024-01-01"
  webhook_secret: ${STRIPE_WEBHOOK_SECRET}
  retry_policy:
    max_attempts: 3
    backoff_factor: 2

redis:
  host: ${REDIS_HOST:-localhost}
  port: ${REDIS_PORT:-6379}
  db: ${REDIS_DB:-0}
  pool_size: 10

fraud_detection:
  risk_threshold: 70
  auto_block_threshold: 90
  review_queue: "fraud_review_queue"

rate_limiting:
  payment_attempts:
    max_requests: 10
    window_seconds: 3600
  refund_attempts:
    max_requests: 5
    window_seconds: 86400
```

## Monitoring
```yaml
metrics:
  - name: payment_processing_time
    type: histogram
    labels: [status, currency]
    buckets: [0.1, 0.5, 1, 2, 5, 10]
  
  - name: payment_amount
    type: histogram
    labels: [currency]
    buckets: [10, 50, 100, 500, 1000, 5000, 10000]
  
  - name: fraud_detections
    type: counter
    labels: [risk_level, action_taken]
  
  - name: refund_rate
    type: gauge
    labels: [reason]

alerts:
  - name: high_failure_rate
    condition: "failure_rate > 0.05"
    severity: critical
    notification: pagerduty
  
  - name: fraud_spike
    condition: "fraud_detection_rate > 0.1"
    severity: warning
    notification: slack
```

## Examples
```python
# Process a payment
result = await payment_processor.process_payment(
    user_id="user_123",
    amount=Decimal("99.99"),
    currency="usd",
    payment_method_id="pm_1234567890"
)

# Check fraud risk
risk = await payment_processor.detect_fraud({
    "user_id": "user_123",
    "amount": 99.99,
    "billing_country": "US",
    "shipping_country": "US"
})

# Process refund
refund = await payment_processor.refund_payment(
    transaction_id="pi_1234567890",
    amount=Decimal("50.00"),
    reason="partial_refund"
)
```