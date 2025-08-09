# EOL File Format Specification

## Overview
EOL uses markdown-based files with embedded code to specify AI application features. The format supports both prototyping (natural language) and implementation (deterministic code) phases.

## File Types

### 1. Feature Files (.eol.md)
Main feature specifications that define functionality, requirements, and implementation. Uses `.eol.md` extension for native GitHub/IDE markdown preview support.

### 2. Test Files (.test.eol.md)
Test specifications using natural language and/or Python code for validation. Uses `.test.eol.md` extension for native markdown rendering.

## Feature File Format (.eol.md)

### Structure
```yaml
# feature-name.eol.md

---
# YAML Frontmatter (required)
name: feature-identifier
version: 1.0.0
phase: prototyping | implementation | hybrid
tags: [tag1, tag2, tag3]
tests: feature-name.test.eol.md  # Optional link to test file
dependencies:
  - redis-mcp
  - another-feature
---

# Feature Name (H1 required)

## Description (required)
Natural language description of the feature for LLM understanding.

## Requirements (required)
- Business requirement 1
- Technical requirement 2
- Performance requirement 3

## Context (optional)
References to relevant documentation and examples:
- @context/patterns/relevant-pattern.md
- @examples/similar-feature.py
- @knowledge/domain-info.md

## Prototyping (required for prototyping/hybrid phase)
```natural
Natural language specification for LLM execution:
  When event occurs:
    Perform action A
    Store result in Redis
    Return response
    
  Handle errors:
    Log error details
    Return graceful error response
```

## Implementation (required for implementation/hybrid phase)
```python
# Python implementation
import redis
from typing import List, Dict

async def feature_function(param1: str, param2: List[str]) -> Dict:
    """Deterministic implementation of feature"""
    # Implementation code
    result = await process_data(param1, param2)
    await redis.set(f"feature:{param1}", result)
    return result
```

```javascript
// Alternative language implementations
function featureFunction(param1, param2) {
    // JavaScript implementation
}
```

## Configuration (optional)
```yaml
# Environment-specific settings
redis:
  host: ${REDIS_HOST:-localhost}
  port: ${REDIS_PORT:-6379}
  db: ${REDIS_DB:-0}

feature:
  timeout: 30
  retry_count: 3
  cache_ttl: 3600
```

## Operations (required)
Mapping of operations to execution phases:

```yaml
operations:
  - name: create_session
    phase: implementation
    function: create_session
    input:
      - user_id: string
      - metadata: object
    output: session_token
    
  - name: validate_session
    phase: prototyping
    description: "Check if session is valid and not expired"
    input:
      - token: string
    output: validation_result
```

## Monitoring (optional)
```yaml
metrics:
  - name: operation_latency
    type: histogram
    labels: [operation, status]
  
  - name: error_rate
    type: counter
    labels: [error_type]

alerts:
  - name: high_latency
    condition: "p95_latency > 100ms"
    severity: warning
```

## Examples (optional)
```python
# Example usage
result = await feature_function("user123", ["role1", "role2"])
assert result["status"] == "success"
```
```

## Test File Format (.test.eol.md)

### Structure
```yaml
# feature-name.test.eol.md

---
# YAML Frontmatter (required)
name: feature-name-tests
feature: feature-name  # Reference to feature file (without extension)
version: 1.0.0
phase: prototyping | implementation | hybrid
type: test
coverage:
  target: 80
  exclude: [debug_*, deprecated_*]
---

# Feature Name Tests (H1 required)

## Test Context (optional)
Setup and teardown requirements for all tests.

## Setup (optional)
```gherkin
# Gherkin setup specification
Background:
  Given a clean Redis instance
  And test data is loaded from fixtures/test-data.json
  And mocked external services are configured
```

```python
# Python setup
@pytest.fixture
async def setup():
    redis_client = await create_test_redis()
    await load_fixtures("fixtures/test-data.json")
    yield redis_client
    await redis_client.flushdb()
```

## Test Cases (required)

### Category: Functional Tests

#### Test: Happy Path Scenario
```gherkin
# Gherkin test specification (prototyping phase)
Feature: Session Management
  As a system
  I want to manage user sessions
  So that users can authenticate and access resources

  Scenario: Create new session for valid user
    Given a valid user "test-user-123" with roles ["admin", "user"]
    When the user creates a new session
    Then a session token should be returned
    And the token should be 32 characters long
    And the session should be stored in Redis with correct data
    And the session TTL should be 1800 seconds
```

```python
# Python implementation (implementation phase)
async def test_happy_path_scenario(setup):
    # Arrange
    user_id = "test-user-123"
    roles = ["admin", "user"]
    
    # Act
    token = await create_session(user_id, roles)
    
    # Assert
    assert token is not None
    assert len(token) == 32
    
    session_data = await redis.hget(f"session:{token}")
    assert session_data["user_id"] == user_id
    assert json.loads(session_data["roles"]) == roles
    
    ttl = await redis.ttl(f"session:{token}")
    assert 1790 < ttl <= 1800
```

#### Test: Error Handling
```gherkin
Scenario: Handle Redis connection failure
  Given Redis connection is unavailable
  When attempting to create a session
  Then the operation should retry 3 times with exponential backoff
  And return an error with code "REDIS_UNAVAILABLE"
  And log the error with appropriate context
```

### Category: Performance Tests

#### Test: Latency Requirements
```gherkin
Scenario: Meet latency requirements under load
  When creating 1000 sessions concurrently
  Then 95% of operations should complete within 50ms
  And 99% should complete within 100ms
  And no operation should exceed 200ms
```

```python
async def test_latency_requirements(setup):
    latencies = []
    
    async def create_timed():
        start = time.time()
        await create_session(f"user-{uuid4()}", ["user"])
        return (time.time() - start) * 1000
    
    tasks = [create_timed() for _ in range(1000)]
    latencies = await asyncio.gather(*tasks)
    
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    assert p95 < 50, f"P95 latency {p95}ms exceeds 50ms"
    assert p99 < 100, f"P99 latency {p99}ms exceeds 100ms"
    assert max(latencies) < 200, f"Max latency {max(latencies)}ms"
```

### Category: Security Tests

#### Test: Token Uniqueness
```gherkin
Scenario: Ensure token uniqueness and security
  When generating 10000 session tokens
  Then all tokens should be unique
  And tokens should use cryptographically secure randomness
  And token entropy should be at least 128 bits
```

### Category: Edge Cases

#### Test: Concurrent Session Limit
```gherkin
Scenario: Enforce maximum sessions per user
  Given the maximum sessions per user is 5
  And a user already has 5 active sessions
  When creating a 6th session
  Then the oldest session should be evicted
  And the new session should be created successfully
  And exactly 5 sessions should remain for the user
```

## Test Data (optional)
```yaml
fixtures:
  users:
    - id: test-user-123
      name: Test User
      roles: [admin, user]
      
    - id: test-user-456
      name: Another User
      roles: [user]
  
  sessions:
    - token: existing-token-123
      user_id: test-user-123
      created_at: 2024-01-01T00:00:00Z
```

## Test Configuration (optional)
```yaml
environment:
  redis:
    host: localhost
    port: 6379
    db: 15  # Separate test database
  
  timeouts:
    default: 5s
    performance_tests: 30s
  
  parallel_execution: true
  retry_failed: 2
  verbose: ${TEST_VERBOSE:-false}
```

## Assertions (optional)
Custom assertion helpers:

```python
# Custom assertions for reuse across tests
def assert_valid_session(session_data):
    assert "user_id" in session_data
    assert "roles" in session_data
    assert "created_at" in session_data
    assert datetime.fromisoformat(session_data["created_at"])
```
```

## File Format Features

### 1. File Extensions
- **`.eol.md`**: Feature specification files
- **`.test.eol.md`**: Test specification files
- **Benefits**: Native GitHub/IDE markdown preview, syntax highlighting, better tooling support

### 2. Frontmatter Metadata
- **Required fields**: name, version, phase
- **Optional fields**: tags, dependencies, tests
- **Extensible**: Add custom fields as needed

### 3. Phase Support
- **prototyping**: Natural language only, executed via LLM
- **implementation**: Deterministic code only
- **hybrid**: Mix of both, operations specify which to use

### 4. Code Block Languages
```
```natural    - Natural language specifications
```gherkin    - BDD test specifications (Given/When/Then)
```python     - Python implementation
```javascript - JavaScript implementation
```yaml       - Configuration
```json       - Data structures
```sql        - Database queries
```

### 5. Reference Syntax
- `@context/` - Reference context documentation
- `@examples/` - Reference example implementations
- `@knowledge/` - Reference domain knowledge
- `@features/` - Reference other features

### 6. Variable Substitution
- `${VAR_NAME}` - Environment variable
- `${VAR_NAME:-default}` - With default value

## Parser Implementation

### EOL Parser
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import markdown

@dataclass
class EOLFeature:
    name: str
    version: str
    phase: str
    tags: List[str]
    description: str
    requirements: List[str]
    prototyping: Optional[Dict]
    implementation: Optional[Dict]
    operations: List[Dict]
    tests: Optional[str]
    
class EOLParser:
    def parse(self, file_path: str) -> EOLFeature:
        """Parse .eol.md file"""
        # Validate file extension
        if not file_path.endswith('.eol.md'):
            raise ValueError(f"Invalid file extension. Expected .eol.md, got {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> EOLFeature:
        # Split frontmatter and content
        parts = content.split('---')
        frontmatter = yaml.safe_load(parts[1])
        markdown_content = '---'.join(parts[2:])
        
        # Parse markdown sections
        sections = self.parse_markdown_sections(markdown_content)
        
        # Extract code blocks by language
        code_blocks = self.extract_code_blocks(sections)
        
        # Build feature model
        feature = EOLFeature(
            name=frontmatter['name'],
            version=frontmatter['version'],
            phase=frontmatter['phase'],
            tags=frontmatter.get('tags', []),
            description=sections.get('description', ''),
            requirements=self.parse_requirements(sections),
            prototyping=code_blocks.get('natural'),
            implementation=code_blocks.get('python'),
            operations=self.parse_operations(sections),
            tests=frontmatter.get('tests')
        )
        
        # Validate
        self.validate_feature(feature)
        
        return feature
    
    def parse_markdown_sections(self, content: str) -> Dict:
        """Parse markdown into sections by headers"""
        md = markdown.Markdown(extensions=['meta', 'fenced_code'])
        # Implementation details
        return sections
    
    def extract_code_blocks(self, sections: Dict) -> Dict:
        """Extract code blocks by language identifier"""
        code_blocks = {}
        # Extract ```language blocks
        return code_blocks
    
    def validate_feature(self, feature: EOLFeature):
        """Validate feature specification"""
        if feature.phase in ['prototyping', 'hybrid']:
            assert feature.prototyping, "Prototyping section required"
        if feature.phase in ['implementation', 'hybrid']:
            assert feature.implementation, "Implementation section required"
```

### Test Parser
```python
@dataclass
class EOLTest:
    name: str
    description: str
    given: List[str]  # Setup conditions
    when: List[str]   # Actions
    then: List[str]   # Assertions
    implementation: Optional[str]  # Python code
    
class EOLTestParser:
    def parse_test_file(self, file_path: str) -> List[EOLTest]:
        """Parse .test.eol.md file"""
        # Validate file extension
        if not file_path.endswith('.test.eol.md'):
            raise ValueError(f"Invalid file extension. Expected .test.eol.md, got {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return self.parse_test_content(content)
    
    def parse_test_content(self, content: str) -> List[EOLTest]:
        # Parse frontmatter
        # Extract test cases
        # Parse natural language tests
        # Extract Python implementations
        # Map natural language to code
        return tests
    
    def generate_test_code(self, test: EOLTest) -> str:
        """Generate Python test from Gherkin specification"""
        if test.implementation:
            return test.implementation
            
        # Use LLM to generate from Gherkin
        prompt = f"""
        Generate Python test for:
        Given: {test.given}
        When: {test.when}
        Then: {test.then}
        """
        
        return generated_code
    
    def parse_gherkin(self, gherkin_text: str) -> List[EOLTest]:
        """Parse Gherkin syntax into test cases"""
        from gherkin.parser import Parser
        from gherkin.pickles import Compiler
        
        parser = Parser()
        gherkin_doc = parser.parse(gherkin_text)
        pickles = Compiler().compile(gherkin_doc)
        
        tests = []
        for pickle in pickles:
            test = EOLTest(
                name=pickle['name'],
                description=pickle.get('description', ''),
                given=[step['text'] for step in pickle['steps'] 
                       if step['type'] == 'Context'],
                when=[step['text'] for step in pickle['steps'] 
                      if step['type'] == 'Action'],
                then=[step['text'] for step in pickle['steps'] 
                      if step['type'] == 'Outcome'],
                implementation=None
            )
            tests.append(test)
        
        return tests
```

## Usage Examples

### Creating a Feature
```bash
# Create feature file
cat > payment-processor.eol.md << EOF
---
name: payment-processor
version: 1.0.0
phase: hybrid
tags: [payments, stripe]
tests: payment-processor.test.eol.md
---

# Payment Processor

## Description
Process payments using Stripe API with Redis caching.

## Prototyping
\`\`\`natural
When processing a payment:
  Validate payment details
  Check Redis cache for customer info
  Process payment via Stripe
  Store transaction in Redis
  Return confirmation
\`\`\`
EOF
```

### Running Tests
```bash
# Run natural language tests
eol test payment-processor.test.eol.md --phase prototyping

# Run implemented tests
eol test payment-processor.test.eol.md --phase implementation

# Generate test implementations from natural language
eol generate-tests payment-processor.test.eol.md
```

### Phase Switching
```bash
# Switch feature to implementation phase
eol switch payment-processor.eol.md --to implementation

# Run in hybrid mode
eol run payment-processor.eol.md --phase hybrid
```

## Best Practices

1. **Start with Natural Language**: Write features and tests in natural language first
2. **Progressive Implementation**: Convert to code incrementally as understanding solidifies
3. **Keep Tests in Sync**: Update test files when features change
4. **Use References**: Link to context documents for complex features
5. **Version Everything**: Track feature and test evolution
6. **Validate Early**: Run natural language tests during prototyping
7. **Document Assumptions**: Include context and requirements clearly

## EOL Integration Benefits

1. **Rapid Prototyping**: Natural language to working prototype quickly
2. **Test-Driven**: Tests guide implementation
3. **Documentation**: Files serve as living documentation
4. **LLM-Friendly**: Optimized for AI understanding and generation
5. **Flexible Evolution**: Smooth transition from prototype to production
6. **Traceable**: Clear mapping between requirements and implementation