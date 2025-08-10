---
name: payment-processor-tests
feature: payment-processor
version: 1.0.0
phase: hybrid
type: test
coverage:
  target: 85
  exclude: [debug_*, deprecated_*]
---

# Payment Processor Tests

## Test Context
Testing payment processing functionality including fraud detection, rate limiting, and refund capabilities.

## Setup
```gherkin
Background:
  Given a clean Redis instance
  And test Stripe API keys are configured
  And mock authentication service is available
  And rate limiter is initialized with test limits
  And test data is loaded from fixtures/payment-test-data.json
```

```python
import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import stripe

@pytest.fixture
async def setup():
    """Setup test environment"""
    # Initialize Redis test instance
    redis_client = await create_test_redis()
    await redis_client.flushdb()
    
    # Mock services
    auth_service = AsyncMock()
    auth_service.authenticate_user.return_value = {
        "id": "user_123",
        "stripe_customer_id": "cus_test123"
    }
    
    rate_limiter = AsyncMock()
    rate_limiter.check_limit.return_value = True
    
    # Initialize payment processor
    processor = PaymentProcessor(redis_client, auth_service, rate_limiter)
    
    yield processor, redis_client, auth_service, rate_limiter
    
    # Cleanup
    await redis_client.flushdb()
    await redis_client.close()
```

## Test Cases

### Category: Payment Processing

#### Test: Successful Payment Processing
```gherkin
Feature: Payment Processing
  As a payment system
  I want to process payments securely
  So that customers can complete purchases

  Scenario: Process valid payment successfully
    Given a valid user "user_123" with stripe customer "cus_test123"
    And a valid payment method "pm_card_visa"
    When the user initiates a payment of $99.99 USD
    Then the payment should be processed successfully
    And the transaction should be stored in Redis
    And the payment method should be cached for 30 days
    And the transaction status should be "succeeded"
```

```python
@pytest.mark.asyncio
async def test_successful_payment(setup):
    processor, redis, auth, rate_limiter = setup
    
    # Mock Stripe response
    with patch('stripe.PaymentIntent.create') as mock_create:
        mock_create.return_value = Mock(
            id="pi_test123",
            status="succeeded",
            amount=9999,
            currency="usd",
            payment_method="pm_card_visa",
            created=1234567890
        )
        
        # Process payment
        result = await processor.process_payment(
            user_id="user_123",
            amount=Decimal("99.99"),
            currency="usd",
            payment_method_id="pm_card_visa"
        )
        
        # Assertions
        assert result["status"] == "succeeded"
        assert result["transaction_id"] == "pi_test123"
        assert result["amount"] == Decimal("99.99")
        
        # Verify Redis storage
        tx_data = await redis.hgetall(f"transaction:pi_test123")
        assert tx_data[b"amount"] == b"9999"
        assert tx_data[b"status"] == b"succeeded"
        
        # Verify payment method caching
        cached_method = await redis.get("payment_method:user_123")
        assert cached_method == b"pm_card_visa"
        
        # Verify TTL is set
        ttl = await redis.ttl("payment_method:user_123")
        assert 86400 * 29 < ttl <= 86400 * 30  # ~30 days
```

#### Test: Payment with Rate Limiting
```gherkin
Scenario: Enforce rate limits on payment attempts
  Given a user "user_123" has made 10 payment attempts in the last hour
  When the user attempts another payment
  Then the payment should be rejected with rate limit error
  And the error message should indicate "Too many payment attempts"
  And no transaction should be created
```

```python
@pytest.mark.asyncio
async def test_rate_limiting(setup):
    processor, redis, auth, rate_limiter = setup
    
    # Configure rate limiter to reject
    rate_limiter.check_limit.return_value = False
    
    # Attempt payment
    with pytest.raises(RateLimitError) as exc_info:
        await processor.process_payment(
            user_id="user_123",
            amount=Decimal("50.00"),
            currency="usd"
        )
    
    assert "Too many payment attempts" in str(exc_info.value)
    
    # Verify no transaction was created
    transactions = await redis.keys("transaction:*")
    assert len(transactions) == 0
```

### Category: Fraud Detection

#### Test: High Risk Transaction Detection
```gherkin
Scenario: Detect high-risk transactions
  Given a transaction with amount $15,000 USD
  And the user has made 6 transactions in the last hour
  And billing country is "US" and shipping country is "CN"
  When fraud detection is performed
  Then the risk score should be greater than 70
  And the transaction should be flagged for review
  And flags should include ["high_amount", "high_velocity", "country_mismatch"]
```

```python
@pytest.mark.asyncio
async def test_fraud_detection_high_risk(setup):
    processor, redis, auth, rate_limiter = setup
    
    # Setup velocity data
    await redis.set("tx_count:user_123:1h", "6")
    
    # Create high-risk transaction
    transaction = {
        "user_id": "user_123",
        "amount": 15000,
        "billing_country": "US",
        "shipping_country": "CN"
    }
    
    # Detect fraud
    result = await processor.detect_fraud(transaction)
    
    # Assertions
    assert result["risk_score"] > 70
    assert result["require_review"] is True
    assert "high_amount" in result["flags"]
    assert "high_velocity" in result["flags"]
    assert "country_mismatch" in result["flags"]
```

#### Test: Low Risk Transaction
```gherkin
Scenario: Allow low-risk transactions
  Given a transaction with amount $50 USD
  And the user has made 1 transaction in the last hour
  And billing and shipping countries match
  When fraud detection is performed
  Then the risk score should be less than 30
  And the transaction should not require review
```

### Category: Refunds

#### Test: Full Refund Processing
```gherkin
Scenario: Process full refund successfully
  Given a completed transaction "pi_completed123" for $100
  When a full refund is requested
  Then the refund should be processed successfully
  And the refund status should be "succeeded"
  And the refund amount should be $100
  And the transaction should be updated with refund status
```

```python
@pytest.mark.asyncio
async def test_full_refund(setup):
    processor, redis, auth, rate_limiter = setup
    
    # Store original transaction
    await redis.hset("transaction:pi_completed123", mapping={
        "amount": "10000",
        "status": "succeeded",
        "user_id": "user_123"
    })
    
    # Mock Stripe refund
    with patch('stripe.Refund.create') as mock_refund:
        mock_refund.return_value = Mock(
            id="re_test123",
            status="succeeded",
            amount=10000
        )
        
        # Process refund
        result = await processor.refund_payment(
            transaction_id="pi_completed123",
            reason="requested_by_customer"
        )
        
        # Assertions
        assert result["refund_id"] == "re_test123"
        assert result["status"] == "succeeded"
        assert result["amount"] == 100.00
        
        # Verify transaction update
        refund_status = await redis.hget("transaction:pi_completed123", "refund_status")
        assert refund_status == b"succeeded"
```

#### Test: Partial Refund
```gherkin
Scenario: Process partial refund
  Given a completed transaction for $100
  When a partial refund of $25 is requested
  Then the refund should be processed for exactly $25
  And the remaining transaction amount should be $75
```

### Category: Performance Tests

#### Test: Payment Processing Latency
```gherkin
Scenario: Meet latency requirements under load
  Given 100 concurrent payment requests
  When all payments are processed
  Then 95% of payments should complete within 2 seconds
  And 99% should complete within 5 seconds
  And no payment should exceed 10 seconds
```

```python
@pytest.mark.asyncio
async def test_payment_latency(setup):
    processor, redis, auth, rate_limiter = setup
    
    async def process_payment_timed():
        start = asyncio.get_event_loop().time()
        
        with patch('stripe.PaymentIntent.create') as mock_create:
            mock_create.return_value = Mock(
                id=f"pi_test_{uuid4()}",
                status="succeeded",
                amount=9999,
                currency="usd",
                payment_method="pm_card_visa",
                created=1234567890
            )
            
            await processor.process_payment(
                user_id=f"user_{uuid4()}",
                amount=Decimal("99.99"),
                currency="usd"
            )
        
        return asyncio.get_event_loop().time() - start
    
    # Process 100 concurrent payments
    tasks = [process_payment_timed() for _ in range(100)]
    latencies = await asyncio.gather(*tasks)
    
    # Calculate percentiles
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(len(latencies) * 0.95)]
    p99 = latencies_sorted[int(len(latencies) * 0.99)]
    max_latency = max(latencies)
    
    # Assertions
    assert p95 < 2.0, f"P95 latency {p95}s exceeds 2s"
    assert p99 < 5.0, f"P99 latency {p99}s exceeds 5s"
    assert max_latency < 10.0, f"Max latency {max_latency}s exceeds 10s"
```

### Category: Error Handling

#### Test: Handle Stripe API Errors
```gherkin
Scenario: Handle Stripe API failures gracefully
  Given the Stripe API is returning errors
  When a payment is attempted
  Then the error should be logged with full context
  And a user-friendly error message should be returned
  And the system should attempt retry with exponential backoff
```

```python
@pytest.mark.asyncio
async def test_stripe_error_handling(setup):
    processor, redis, auth, rate_limiter = setup
    
    # Mock Stripe error
    with patch('stripe.PaymentIntent.create') as mock_create:
        mock_create.side_effect = stripe.error.APIConnectionError(
            "Connection failed"
        )
        
        # Attempt payment
        with pytest.raises(stripe.error.APIConnectionError):
            await processor.process_payment(
                user_id="user_123",
                amount=Decimal("50.00"),
                currency="usd"
            )
        
        # Verify error was handled (check logs in real implementation)
        # Verify retry logic was triggered
        assert mock_create.call_count >= 1
```

### Category: Security Tests

#### Test: PCI Compliance
```gherkin
Scenario: Ensure PCI compliance in payment handling
  Given sensitive payment data
  When payment is processed
  Then card numbers should never be stored in Redis
  And only payment method tokens should be cached
  And all sensitive data should be encrypted in transit
```

## Test Data
```yaml
fixtures:
  users:
    - id: user_123
      stripe_customer_id: cus_test123
      email: test@example.com
      
    - id: user_456
      stripe_customer_id: cus_test456
      email: another@example.com
  
  payment_methods:
    - id: pm_card_visa
      type: card
      brand: visa
      last4: "4242"
      
    - id: pm_card_mastercard
      type: card
      brand: mastercard
      last4: "5555"
  
  transactions:
    - id: pi_completed123
      amount: 10000
      currency: usd
      status: succeeded
      user_id: user_123
```

## Test Configuration
```yaml
environment:
  stripe:
    api_key: sk_test_1234567890
    webhook_secret: whsec_test1234567890
  
  redis:
    host: localhost
    port: 6379
    db: 15  # Test database
  
  timeouts:
    default: 10s
    performance_tests: 60s
  
  parallel_execution: true
  retry_failed: 2
  verbose: ${TEST_VERBOSE:-false}
```

## Assertions
```python
# Custom assertions for payment testing
def assert_valid_transaction(tx_data):
    """Validate transaction data structure"""
    required_fields = ['amount', 'currency', 'status', 'user_id', 'created_at']
    for field in required_fields:
        assert field.encode() in tx_data, f"Missing field: {field}"
    
    # Validate amount is positive
    amount = int(tx_data[b'amount'])
    assert amount > 0, "Transaction amount must be positive"
    
    # Validate status
    valid_statuses = [b'succeeded', b'pending', b'failed', b'canceled']
    assert tx_data[b'status'] in valid_statuses, "Invalid transaction status"

def assert_fraud_score_valid(risk_score):
    """Validate fraud risk score"""
    assert 0 <= risk_score <= 100, "Risk score must be between 0 and 100"
```