
"""
observability/tracer.py
======================
Adonis Trace System:Decision-level instrumentation, trace persistence, and replay capabilities.
"""
import uuid, time, json, logging, asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("tracer")

@dataclass
class TraceEvent:
    event_id: str
    timestamp: float
    agent: str
    event_type: str  # "input", "tool_call", "memory_access", "output", "decision"
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdonisTracer:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.sample_rate = 0.1  # Default: 10% of traces
        self.current_trace_id = None

    def start_trace(self, trace_id: Optional[str] = None):
        self.current_trace_id = trace_id or str(uuid.uuid4())
        return self.current_trace_id

    async def record(self, agent: str, event_type: str, payload: Any, metadata: Dict[str, Any] = None):
        if not self.current_trace_id:
            return

        # Sampling logic: unless it's a critical error or specifically requested
        # we only record based on sample_rate.
        # For a real system, we might use a deterministic hash of the session_id for sampling.
        
        event = TraceEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            agent=agent,
            event_type=event_type,
            payload=payload,
            metadata=metadata or {}
        )
        
        event_json = json.dumps(event.__dict__, default=str)
        
        # Store in Redis list for the trace duration
        if self.redis:
            key = f"adonis:trace:{self.current_trace_id}"
            await self.redis.lpush(key, event_json)
            await self.redis.expire(key, 86400) # 24h TTL

    def end_trace(self):
        self.current_trace_id = None

# Singleton for easy import across agents
tracer = None

def get_tracer(redis_client=None):
    global tracer
    if tracer is None:
        tracer = AdonisTracer(redis_client)
    return tracer
