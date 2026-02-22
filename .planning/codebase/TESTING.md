# Testing Patterns

**Analysis Date:** 2026-02-21

## Test Framework

**Runner:**
- pytest (version 8.3.0+)
- Config: Not yet created; no `pytest.ini`, `conftest.py`, or pytest configuration in `pyproject.toml`
- Can be configured via `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml`

**Assertion Library:**
- Not yet configured; pytest's built-in assertion introspection is available
- Can add `pytest>=8.3.0` with built-in assertions or `pytest-assertions` if needed

**Run Commands:**
```bash
pytest                          # Run all tests (when tests exist)
pytest -v                       # Verbose mode with test names
pytest -k "test_name"           # Run specific test by name pattern
pytest --cov=app                # Run with coverage (requires pytest-cov)
pytest -x                       # Exit on first failure
pytest -s                       # Show print statements and logging
```

## Test File Organization

**Location:**
- **Current state:** No test files exist in repository
- **Recommended pattern:** Co-located with source files
  - Tests for `app/services/firebase.py` → `app/services/test_firebase.py`
  - Tests for `app/utils/image.py` → `app/utils/test_image.py`
  - Alternatively, parallel directory structure: `tests/services/`, `tests/utils/`, `tests/routers/`

**Naming:**
- Test files: `test_*.py` or `*_test.py` (pytest discovers both)
- Test functions: `test_<function_or_behavior>()`
- Test classes: `Test<ComponentName>` (pytest discovers test classes with `Test` prefix)

**Structure:**
```
app/
├── services/
│   ├── firebase.py
│   ├── test_firebase.py          # Tests for firebase service
│   ├── flux_inpaint.py
│   ├── test_flux_inpaint.py      # Tests for flux_inpaint service
│   └── pipeline.py
│       test_pipeline.py          # Tests for pipeline orchestration
├── utils/
│   ├── image.py
│   └── test_image.py             # Tests for image utilities
├── routers/
│   ├── inpaint.py
│   └── test_inpaint.py           # Tests for API endpoints
└── test_main.py                  # Tests for FastAPI app setup
```

## Test Structure

**Suite Organization:**
- No existing tests, but pattern to follow:
```python
import pytest
from app.services.firebase import download_blob, upload_blob

class TestFirebaseStorage:
    """Tests for Firebase Cloud Storage operations."""

    def test_download_blob_returns_bytes(self):
        """Downloading a blob should return bytes."""
        # Setup
        # Execute
        # Assert

    def test_download_blob_logs_size(self, caplog):
        """Download operation should log byte count."""
        # Setup
        # Execute
        # Assert with caplog

class TestFirebaseFirestore:
    """Tests for Firebase Firestore operations."""

    def test_session_exists_returns_bool(self):
        """session_exists should return boolean value."""
        pass
```

**Patterns:**
- **Setup:** Arrange test data, mock dependencies, set up fixtures
- **Teardown:** Clean up mocks, reset state (handled by pytest fixtures with `yield`)
- **Assertion:** Use pytest's assertion syntax with clear messages

## Mocking

**Framework:** `unittest.mock` (built-in Python library)

**Patterns:**
- Use `unittest.mock.patch()` to mock Firebase clients and services
- Use `unittest.mock.MagicMock()` for creating mock objects
- For async code (e.g., FastAPI endpoints), use `pytest-asyncio` and `unittest.mock.AsyncMock()`

**Example mocking pattern for Firebase:**
```python
from unittest.mock import patch, MagicMock
import pytest

@patch('app.services.firebase.get_storage_bucket')
def test_download_blob(mock_bucket):
    """Test download_blob with mocked bucket."""
    # Setup mock
    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = b"image_data"
    mock_bucket.return_value.blob.return_value = mock_blob

    # Execute
    result = download_blob("path/to/file.jpg")

    # Assert
    assert result == b"image_data"
    mock_bucket.return_value.blob.assert_called_once_with("path/to/file.jpg")
```

**Async mocking pattern for FastAPI:**
```python
@pytest.mark.asyncio
@patch('app.services.flux_inpaint.is_ready')
async def test_handle_inpaint_when_not_ready(mock_is_ready):
    """Test POST /inpaint returns 503 when model not ready."""
    mock_is_ready.return_value = False

    # Execute with test client (use httpx.AsyncClient or FastAPI TestClient)
    # Assert response.status_code == 503
```

**What to Mock:**
- External service clients: Firebase Admin SDK, diffusers pipelines, torch models
- Environment-dependent operations: file I/O, API calls, model loading
- Expensive operations: model inference, image processing on real data
- Non-deterministic operations: time.time() if testing timing logic

**What NOT to Mock:**
- Pure utility functions: `resize_for_flux()`, image encoding/decoding logic
- Small functions with no dependencies: `is_ready()` (just checks state)
- Application logic flow: the pipeline orchestration (test with mocked I/O instead)
- Standard library functions unless necessary (e.g., `logging` fixtures)

## Fixtures and Factories

**Test Data:**
- No test fixtures currently defined
- Should create fixtures for common test data:

```python
import pytest
from PIL import Image
import io

@pytest.fixture
def sample_image() -> bytes:
    """Create a small test image as JPEG bytes."""
    img = Image.new('RGB', (256, 256), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

@pytest.fixture
def sample_mask() -> bytes:
    """Create a test mask image as PNG bytes."""
    img = Image.new('L', (256, 256), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

@pytest.fixture
def inpaint_payload():
    """Create sample Pub/Sub message payload."""
    return {
        "userId": "test-user-123",
        "sessionId": "test-session-456"
    }
```

**Location:**
- `conftest.py` in root of `tests/` directory (or `app/` if using co-located pattern)
- Shared fixtures accessible to all tests
- Module-specific fixtures in `conftest.py` files within subdirectories

## Coverage

**Requirements:** Not enforced
- No minimum coverage threshold set
- No coverage reporting configured in `pyproject.toml`

**View Coverage:**
```bash
# Requires pytest-cov plugin (add to dev dependencies)
pytest --cov=app --cov-report=html
# Opens coverage report in htmlcov/index.html

pytest --cov=app --cov-report=term-missing
# Shows covered lines and missing line numbers
```

**Recommended coverage targets:**
- Utilities (`app/utils/`) - aim for 95%+ (pure functions)
- Services (`app/services/`) - aim for 80%+ (external dependencies mocked)
- Routers (`app/routers/`) - aim for 70%+ (integration testing)
- Main app setup - 60%+ (harder to test, less critical)

## Test Types

**Unit Tests:**
- **Scope:** Individual functions in isolation with dependencies mocked
- **Approach:**
  - Test each pure utility function: `resize_for_flux()`, `encode_jpeg()`, `create_preview()`
  - Test service functions with mocked external calls: `download_blob()` with mocked storage bucket
  - Test logic branches: error cases in `update_session()`, `session_exists()`
  - Use mocks to test error handling: `NotFound` exception from Firestore
- **Location:** `app/*/test_*.py` co-located with source

**Integration Tests:**
- **Scope:** Test interactions between multiple modules with minimal mocking
- **Approach:**
  - Test `run_inpaint()` pipeline with mocked Firebase I/O but real image processing
  - Test API endpoints with mocked services using FastAPI `TestClient`
  - Test message parsing and routing through `handle_inpaint()` endpoint
  - Verify state updates flow correctly from router → pipeline → firebase
- **Example:**
```python
def test_inpaint_endpoint_updates_firestore():
    """Full flow: receive message → process → update Firestore."""
    # Mock firebase calls, patch inpaint service
    # Send POST request to /inpaint endpoint
    # Assert session marked as 'completed'
    # Assert paths updated correctly
```

**E2E Tests:**
- **Framework:** Not used
- **Rationale:** Full end-to-end testing would require real Firebase credentials, model loading, and actual inference - better suited for manual testing or staging environment

## Common Patterns

**Async Testing:**
```python
import pytest

# For testing async FastAPI endpoints
@pytest.mark.asyncio
async def test_health_endpoint():
    """Health endpoint should return status ok."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Or using httpx.AsyncClient for true async testing
@pytest.mark.asyncio
async def test_ready_endpoint_when_model_loaded():
    """Ready endpoint should return 503 while model loading."""
    from httpx import AsyncClient
    from app.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/ready")
        # Assert based on mock state
```

**Error Testing:**
```python
def test_handle_inpaint_with_missing_field():
    """Missing required field should return 200 (ack bad message)."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post("/inpaint",
        json={"message": {"data": base64.b64encode(b'{"sessionId": "123"}')}}
    )
    assert response.status_code == 200  # Acked despite error

def test_handle_inpaint_retries_on_transient_error():
    """Transient errors should nack (status 500) to trigger retry."""
    with patch('app.services.pipeline.run_inpaint', side_effect=Exception("Network error")):
        response = client.post("/inpaint", json=valid_payload)
        assert response.status_code == 500  # Nack for retry

def test_handle_inpaint_acks_after_max_retries():
    """After 5 attempts, should ack to prevent infinite retries."""
    payload_with_attempt = {
        "message": {...},
        "deliveryAttempt": 6
    }
    with patch('app.services.pipeline.run_inpaint', side_effect=Exception("Error")):
        response = client.post("/inpaint", json=payload_with_attempt)
        assert response.status_code == 200  # Acked despite error
```

---

*Testing analysis: 2026-02-21*
