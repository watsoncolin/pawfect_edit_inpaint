# Coding Conventions

**Analysis Date:** 2026-02-21

## Naming Patterns

**Files:**
- Snake case: `flux_inpaint.py`, `image.py`, `firebase.py`, `pipeline.py`
- Directory names use snake case: `routers/`, `services/`, `utils/`
- Router modules named by feature: `inpaint.py`
- Service modules named by integration or function: `flux_inpaint.py`, `firebase.py`
- Utility modules named by domain: `image.py`

**Functions:**
- Snake case for all functions: `load_model()`, `is_ready()`, `inpaint()`, `run_inpaint()`, `decode_image()`, `resize_for_flux()`
- Private functions prefixed with underscore: `_init()`, `_load_model_background()`
- Descriptive, action-oriented names: `upload_blob()`, `session_exists()`, `update_session()`

**Variables:**
- Snake case for all variables: `user_id`, `session_id`, `image_bytes`, `delivery_attempt`, `doc_ref`
- Module-level constants in UPPER_SNAKE_CASE: `PROMPT`, `NUM_STEPS`, `GUIDANCE_SCALE`, `GGUF_URL`, `COST_PER_INPAINT`
- Global state variables prefixed with underscore: `_pipe`, `_initialized`
- Meaningful names that indicate type or purpose: `image_resized`, `mask_bytes`, `edited_path`

**Types:**
- No explicit type aliases defined; uses built-in types and library types
- PIL types used directly: `Image.Image`, `Image.LANCZOS`
- FastAPI types used directly: `Request`, `Response`, `APIRouter`

## Code Style

**Formatting:**
- Black (version 24.10.0+) - enforced via tool configuration
- Line length: 100 characters
- Target Python version: 3.10+

**Linting:**
- Ruff (version 0.8.0+) - enforced via tool configuration
- Line length: 100 characters
- Tool configuration in `pyproject.toml`:
  ```toml
  [tool.black]
  line-length = 100
  target-version = ['py310']

  [tool.ruff]
  line-length = 100
  target-version = "py310"
  ```

## Import Organization

**Order:**
1. Standard library imports: `import logging`, `import threading`, `import base64`, `import json`, `import os`, `import time`, `import io`
2. Third-party library imports: `import torch`, `import fastapi`, `import firebase_admin`, `from PIL import Image`, `from diffusers import ...`
3. Local application imports: `from app.services import firebase`, `from app.services.flux_inpaint import inpaint`

**Path Aliases:**
- Uses direct relative imports from package root: `from app.services.flux_inpaint import is_ready`
- Module imports for indirect service access: `from app.services import firebase` (allows `firebase.session_exists()`)

**Handling of late imports:**
- Router imports deferred to end of file in `main.py` to avoid circular dependencies, with `# noqa: E402` comment on import line 47

## Error Handling

**Patterns:**
- Generic exception catching with context logging: `except Exception as e:` followed by `logger.exception()` to log full traceback
- Specific exception types caught where needed: `except KeyError as e:` for missing message fields, `except NotFound` for deleted Firestore documents
- Graceful degradation: Return appropriate HTTP status codes (503 for not ready, 200 to ack messages)
- Error context preservation: Include relevant identifiers in error messages: `logger.error(f"Missing required field in message: {e}")`
- Transient vs. permanent error distinction: Check delivery attempt count to limit retries (max 5 attempts in `inpaint.py`)
- Two-stage error handling in pipeline: Log exception, then attempt to update Firestore with error status, with fallback logging if that fails

**Error recovery:**
```python
# From routers/inpaint.py - delivery attempt checking
delivery_attempt = envelope.get("deliveryAttempt", 1)
if delivery_attempt >= 5:
    logger.error(f"Max retries reached (attempt {delivery_attempt}), acking: {e}")
    return Response(status_code=200)  # Ack to prevent infinite retry
```

## Logging

**Framework:** Python `logging` module (not third-party)

**Setup:**
- Configured in `main.py` via `logging.basicConfig(level=logging.INFO)`
- Module-level loggers obtained via `logger = logging.getLogger(__name__)`

**Patterns:**
- **Informational logging:** Track major operations: `logger.info(f"Starting inpaint: userId={user_id}, sessionId={session_id}")`
- **Warning logging:** Log degraded conditions or recoverable errors: `logger.warning(f"Model not ready yet, nacking for retry")`
- **Error logging:** Log exceptions with context and identifiers: `logger.error(f"Missing required field in message: {e}")`
- **Exception logging:** Use `logger.exception()` to automatically include full traceback: `logger.exception(f"Error during inpaint: {e}")`
- **Include identifying data in all log messages:** user IDs, session IDs, timings, sizes, attempt counts
- **Log timing information:** `logger.info(f"Inference completed in {elapsed:.1f}s")`

## Comments

**When to Comment:**
- Docstrings on all public functions describing purpose and parameters
- Comment technical decisions that may be non-obvious: "Composite: only use FLUX output in the masked area, keep original pixels elsewhere. This prevents color shifts in unmasked regions."
- Comment non-standard behaviors: "Ensure mask matches image dimensions" before resizing logic
- Comment business logic: "Mark as processing" before state update
- Do not comment obvious code: function names and docstrings usually suffice

**JSDoc/TSDoc:**
- Uses Python docstrings (not JSDoc)
- Style: triple-quoted strings with description on first line
- Simple single-line docstrings for straightforward functions:
  ```python
  def is_ready() -> bool:
      return _pipe is not None
  ```
- More detailed docstrings for complex operations:
  ```python
  def load_model():
      """Load FLUX.1-Fill-dev pipeline with Q4 GGUF quantized transformer."""
  ```
- Docstring describes what the function does, not implementation details

## Function Design

**Size:**
- Functions are concise and focused on single responsibility
- Typical range: 5-30 lines
- Longest function is `run_inpaint()` at 75 lines (but it's the full pipeline orchestration)
- Utility functions are very small: `is_ready()` is 2 lines, image transform functions 5-8 lines

**Parameters:**
- Explicit type hints on all parameters: `def download_blob(path: str) -> bytes:`
- Use positional parameters for required arguments
- Use keyword arguments with defaults for optional configuration: `max_size: int = 1024`, `quality: int = 95`
- Limit parameters to 3-4 for readability
- Pass dictionaries for related configuration groups: `firebase.update_session(user_id, session_id, updates: dict)`

**Return Values:**
- Always include return type hints: `-> bool`, `-> bytes`, `-> Image.Image`, `-> None` (implicit)
- Return meaningful values that indicate success/failure or results
- None is used only for operations that modify state externally (e.g., `firebase.update_session()`)
- Use early returns for error conditions: check `if not firebase.session_exists()` and `return` before proceeding

## Module Design

**Exports:**
- Modules export only public functions and constants
- Private implementation functions prefixed with underscore: `_init()`, `_load_model_background()`
- Services export facade functions: `firebase.download_blob()`, `firebase.upload_blob()`
- Utilities export pure functions: `image.decode_image()`, `image.resize_for_flux()`

**Barrel Files:**
- Not used; `__init__.py` files are empty
- Direct imports from specific modules: `from app.services.firebase import ...` or `from app.services import firebase`

**Module structure:**
- Imports at top
- Constants next
- Private module-state variables (`_pipe`, `_initialized`)
- Public functions in logical order
- One responsibility per module: flux_inpaint handles model, firebase handles Cloud Firestore/Storage, pipeline orchestrates the full flow

---

*Convention analysis: 2026-02-21*
