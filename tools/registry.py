
"""
tools/registry.py
======================
Dynamic Tool Registry with Schema Validation and Fallback Chains.
Ensures that agent-tool interactions are resilient to API changes and failures.
"""
import json, logging, time, asyncio, hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from jsonschema import validate, ValidationError

log = logging.getLogger("registry")

@dataclass
class ToolSchema:
    name: str
    endpoint: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    fallback_chain: List[Dict[str, Any]] = field(default_factory=list)
    timeout: int = 30
    retry_limit: int = 3

class ToolRegistry:
    def __init__(self, redis_client=None):
        self.tools: Dict[str, ToolSchema] = {}
        self.redis = redis_client
        self._load_defaults()

    def _load_defaults(self):
        # Pre-seed with essential system tools
        pass

    def register_tool(self, schema: ToolSchema):
        """Register or update a tool definition."""
        self.tools[schema.name] = schema
        log.info(f"[REGISTRY] Tool '{schema.name}' registered/updated.")

    def get_tool(self, name: str) -> ToolSchema:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered in the Adonis Registry.")
        return self.tools[name]

    def validate_input(self, tool_name: str, data: Dict[str, Any]) -> bool:
        tool = self.get_tool(tool_name)
        if not tool.input_schema:
            return True
        try:
            validate(instance=data, schema=tool.input_schema)
            return True
        except ValidationError as e:
            log.error(f"[REGISTRY] Input validation failed for '{tool_name}': {e.message}")
            return False

    def validate_output(self, tool_name: str, data: Dict[str, Any]) -> bool:
        tool = self.get_tool(tool_name)
        if not tool.response_schema:
            return True
        try:
            validate(instance=data, schema=tool.response_schema)
            return True
        except ValidationError as e:
            log.error(f"[REGISTRY] Output validation failed for '{tool_name}': {e.message}")
            return False

class ToolProxy:
    """
    The execution wrapper that handles the Fallback Chain.
    Sequence: Primary -> Retry -> Fallback 1 -> Fallback 2 -> LLM Estimate.
    """
    def __init__(self, registry: ToolRegistry, execution_map: Dict[str, Callable]):
        self.registry = registry
        self.execution_map = execution_map

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.registry.get_tool(tool_name)
        
        # 1. Input Validation
        if not self.registry.validate_input(tool_name, args):
            return {"status": "error", "reason": "Invalid input schema", "agent": "registry"}

        # 2. Primary Attempt with Retries
        for attempt in range(tool.retry_limit + 1):
            try:
                result = await self._execute(tool_name, args)
                if self.registry.validate_output(tool_name, result):
                    return result
                log.warn(f"[PROXY] {tool_name} passed but failed response schema. Attempt {attempt+1}")
            except Exception as e:
                log.warn(f"[PROXY] {tool_name} failed: {e}. Attempt {attempt+1}/{tool.retry_limit+1}")
                if attempt == tool.retry_limit:
                    break
                await asyncio.sleep(2 ** attempt) # Exponential backoff

        # 3. Fallback Chain Execution
        for fallback in tool.fallback_chain:
            try:
                fallback_type = fallback.get("type")
                if fallback_type == "backup_api":
                    backup_name = fallback.get("name")
                    log.info(f"[PROXY] Switching to backup API: {backup_name}")
                    return await self.call(backup_name, args)
                
                elif fallback_type == "cached_data":
                    max_age = fallback.get("max_age", "1h")
                    log.info(f"[PROXY] Attempting cached retrieval (max_age={max_age})")
                    # Logic for redis cached fetch would go here
                    return {"status": "cached", "data": "...", "stale": True}
                
                elif fallback_type == "llm_estimate":
                    log.info(f"[PROXY] Triggering LLM heuristic estimate")
                    return {"status": "estimated", "data": "approximate value", "source": "llm_heuristic"}
            except Exception as e:
                log.error(f"[PROXY] Fallback {fallback} failed: {e}")

        return {"status": "critical_failure", "reason": "All fallback chains exhausted", "agent": "registry"}

    async def _execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Internal executor that maps tool name to the actual function."""
        fn = self.execution_map.get(tool_name)
        if not fn:
            raise NotImplementedError(f"No execution function mapped for tool '{tool_name}'")
        return await fn(args)
