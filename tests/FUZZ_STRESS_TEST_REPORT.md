# PrometheusFuse Stress Test & Edge Case Report

**Test File:** `tests/test_fuse_stress_edge_cases.py`  
**Date:** 2026-07-06  
**Total Tests:** 83  
**Status:** ✅ All tests passing

---

## Test Coverage Summary

### 1. Empty/None Inputs (8 tests) ✅
- Empty descriptions, payloads, agent names, action types
- Whitespace-only descriptions
- None values in payloads
- Empty session/trace IDs

**Findings:** Fuse handles all empty/None inputs gracefully. No crashes.

---

### 2. Malformed LLM Responses (13 tests) ✅
- Invalid JSON (garbage text)
- Empty JSON objects `{}`
- Partial fields (missing required keys)
- Wrong types (strings instead of ints)
- Extra unknown fields
- Null values
- Arrays instead of objects
- Negative scores (clamped to 0)
- Scores > 10 (clamped to 10)
- Float scores (truncated to int)
- Remediation failures (invalid JSON, missing fields, `impossible: true`)

**Findings:** 
- LLM scoring failures fall back to heuristic baseline ✅
- Remediation failures return `None` → action blocked ✅
- Score clamping works correctly (0-10 range) ✅
- Extra fields in LLM response cause `TypeError` → fallback ✅

---

### 3. Boundary Values (12 tests) ✅
- Score exactly at thresholds: 10 (GREEN), 11 (YELLOW), 25 (YELLOW), 26 (ORANGE), 40 (ORANGE), 41 (RED), 55 (RED), 56 (BLACK)
- Score 0 (GREEN) and 60 (BLACK)
- LLM scoring band boundaries (15-45)

**Findings:**
- All boundary values map to correct `FuseLevel` ✅
- Score 15 triggers LLM re-scoring ✅
- Score 14 does NOT trigger LLM ✅

---

### 4. Unicode/Encoding (11 tests) ✅
- Emoji in descriptions and payloads
- Non-ASCII characters (French, Chinese, Arabic)
- Mixed scripts
- Keywords hidden in Unicode (e.g., `🔥kill🔥`)
- Very long descriptions (10,000+ chars)
- Special characters (`!@#$%^&*()...`)
- Null bytes (`\x00`)
- Newlines and tabs

**Findings:**
- Unicode handling is robust ✅
- Keywords embedded in Unicode still detected ✅
- No encoding crashes ✅

---

### 5. Payload Injection (8 tests) ✅
- Keywords in payload values
- Keywords in payload keys
- Nested payload objects (3+ levels deep)
- List values in payloads
- Permission-based risky combos
- Boolean, numeric, and mixed-type values

**Findings:**
- Payload is serialized via `json.dumps()` and scanned ✅
- Nested objects are flattened into the search text ✅
- Risky permission combos detected correctly ✅

---

### 6. Redis Failures (6 tests) ✅
- Connection errors during audit
- Timeouts during hard_kill
- Data corruption (garbage bytes in Redis)
- ltrim failures
- Unknown agent lock checks

**Findings:**
- **CRITICAL GAP:** Redis failures propagate as unhandled exceptions ⚠️
  - `ConnectionError` during `lpush` (audit) → crashes `evaluate()`
  - `TimeoutError` during `set` (hard_kill) → crashes `evaluate()`
  - `ConnectionError` during `ltrim` → crashes `evaluate()`
- **Recommendation:** Wrap Redis operations in try/except to prevent fuse bypass on Redis failures

---

### 7. LLM Failures (7 tests) ✅
- Timeouts during scoring
- Network errors during scoring
- Timeouts during remediation
- Network errors during remediation
- Empty string responses
- None content (AttributeError)
- Generic exceptions

**Findings:**
- All LLM failures are caught and handled gracefully ✅
- Scoring failures fall back to heuristic baseline ✅
- Remediation failures return `None` → action blocked ✅
- Broad `except Exception` catches unexpected errors ✅

---

### 8. Webhook Failures (6 tests) ✅
- Invalid URLs
- Connection refused (localhost:19999)
- Timeouts (non-routable IP)
- Empty webhook string (no call)
- Webhook called for RED and BLACK decisions

**Findings:**
- Webhook failures are caught and logged ✅
- No crashes on webhook errors ✅
- Empty webhook string skips HTTP call ✅
- `httpx.HTTPError` and generic `Exception` both caught ✅

---

### 9. Extreme Values (12 tests) ✅
- Very large payloads (10,000 keys)
- Deeply nested payloads (50 levels)
- Very long agent names (10,000 chars)
- Very long action types (10,000 chars)
- Repeated keywords (1,000× "kill")
- Many permissions (500+)
- Concurrent evaluations (20 parallel)
- Rapid sequential evaluations (50 in a row)
- Unicode keys in payloads
- All signal keywords in one description

**Findings:**
- No performance degradation or crashes ✅
- Concurrent evaluations work correctly ✅
- Audit log grows correctly (50 entries) ✅
- Axis capping works (repeated keywords don't overflow) ✅
- All 6 axes max out at 60 total ✅

---

## Error Handling Gaps Found

### ⚠️ CRITICAL: Redis Failures Not Handled

**Issue:** Redis operations in `_audit()`, `_hard_kill()`, and `is_locked()` can raise unhandled exceptions that crash the fuse.

**Affected Code:**
```python
# prometheus/fuse.py:291-303
async def _audit(self, dec: FuseDecision) -> None:
    rec = {...}
    await self.redis.lpush("prometheus:audit", json.dumps(rec))  # Can raise
    await self.redis.ltrim("prometheus:audit", 0, AUDIT_CAP - 1)  # Can raise

# prometheus/fuse.py:326-331
async def _hard_kill(self, name: str) -> None:
    await self.redis.set(...)  # Can raise
    log.critical(...)

# prometheus/fuse.py:333-336
@staticmethod
async def is_locked(redis_client: Any, name: str) -> bool:
    return bool(await redis_client.exists(...))  # Can raise
```

**Impact:**
- If Redis is down, the fuse crashes instead of failing safe
- BLACK-tier threats may not be locked if Redis times out
- Audit log may be lost on Redis failures

**Recommendation:**
Wrap Redis operations in try/except blocks:
```python
async def _audit(self, dec: FuseDecision) -> None:
    try:
        rec = {...}
        await self.redis.lpush("prometheus:audit", json.dumps(rec))
        await self.redis.ltrim("prometheus:audit", 0, AUDIT_CAP - 1)
    except Exception as e:
        log.error(f"[FUSE] Audit failed: {e}")

async def _hard_kill(self, name: str) -> None:
    try:
        await self.redis.set(...)
        log.critical(...)
    except Exception as e:
        log.error(f"[FUSE] Hard kill failed: {e}")
```

---

## Inputs That Broke the Fuse

**None.** The fuse handled all 83 edge cases without crashes (except for the Redis gap above, which is a design issue, not an input issue).

**Notable behaviors:**
- Repeated keywords only match once (e.g., "kill kill kill" → +9, not +27)
- LLM responses with extra fields cause `TypeError` → fallback to baseline
- Payload injection via nested objects is detected (json.dumps flattens structure)
- Unicode keywords are detected (e.g., "🔥kill🔥" matches "kill")

---

## Test Execution

```bash
$ python3 -m pytest tests/test_fuse_stress_edge_cases.py -v
============================== 83 passed in 5.93s ==============================
```

**Breakdown by category:**
- Empty/None inputs: 8 passed
- Malformed LLM responses: 13 passed
- Boundary values: 12 passed
- Unicode/encoding: 11 passed
- Payload injection: 8 passed
- Redis failures: 6 passed
- LLM failures: 7 passed
- Webhook failures: 6 passed
- Extreme values: 12 passed

---

## Recommendations

1. **Fix Redis error handling** (CRITICAL)
   - Wrap `_audit()`, `_hard_kill()`, and `is_locked()` in try/except
   - Log errors but don't crash the fuse
   - Consider fallback behavior (e.g., in-memory lock if Redis is down)

2. **Add rate limiting** (LOW)
   - No protection against rapid-fire evaluations
   - Could be abused to spam LLM API

3. **Add payload size limits** (LOW)
   - Very large payloads (10,000+ keys) are handled but may cause performance issues
   - Consider rejecting payloads > 1MB

4. **Document LLM fallback behavior** (INFO)
   - LLM scoring failures fall back to heuristic baseline
   - This is safe but may miss nuanced threats

---

## Conclusion

The PrometheusFuse is **robust against malformed inputs, LLM failures, and extreme values**. The only critical gap is **unhandled Redis exceptions**, which could cause the fuse to crash instead of failing safe. This should be fixed before production deployment.

All other error paths are handled gracefully with appropriate fallbacks.
