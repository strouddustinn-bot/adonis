
"""
observability/analyzer.py
======================
Adonis Trace Analyzer: Replay engine, Drift detection, and Automated Root-Cause Analysis.
"""
import json, logging, asyncio, time
from typing import Any, Dict, List, Tuple, Optional
from observability.tracer import get_tracer

log = logging.getLogger("analyzer")

class TraceAnalyzer:
    def __init__(self, llm_client, redis_client=None):
        self.llm = llm_client
        self.redis = redis_client

    async def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        \"\"\"Fetch all events for a specific trace from Redis.\"\"\"
        if not self.redis: return []
        raw = await self.redis.lrange(f"adonis:trace:{trace_id}", 0, -1)
        return [json.loads(e) for e in reversed(raw)]

    async def replay_and_compare(self, trace_id: str, agent_instance, input_data: Any) -> Dict[str, Any]:
        \"\"\"
        Execution Replay:
        1. Loads the original trace to find the la-baseline.
        2. Executes the agent with the SAME input.
        3. Compares the la-new trace and output to the baseline.
        \"\"\"
        baseline_events = await self.get_trace(trace_id)
        if not baseline_events:
            return {"status": "error", "reason": "Trace not found"}

        # Execute agent again (this would normally require state restoration)
        # Here we simulate the execution and capture the outcome
        start_time = time.monotonic()
        new_output = await agent_instance.handle({"goal": input_data}, "replay_session")
        latency = time.monotonic() - start_time

        baseline_output = next((e['payload'] for e in baseline_events if e['event_type'] == 'output'), None)
        
        # Drift Detection
        drift = self._calculate_drift(baseline_output, new_output)
        
        return {
            "status": "success",
            "drift_score": drift,
            "baseline": baseline_output,
            "current": new_output,
            "latency": latency
        }

    def _calculate_drift(self, old: Any, new: Any) -> float:
        if not old or not new: return 1.0
        # Simplified drift: string distance or token overlap.
        # In production, this would use a semantic similarity model.
        if isinstance(old, str) and isinstance(new, str):
            return 1.0 - (len(set(old.split()) & set(new.split())) / max(len(old.split()), 1))
        return 0.0 if old == new else 1.0

    async def perform_rca(self, trace_id: str) -> str:
        \"\"\"
        Automated Root-Cause Analysis:
        Feeds the entire trace to the LLM to find the logical failure point.
        \"\"\"
        events = await self.get_trace(trace_id)
        trace_text = json.dumps([e.__dict__ if hasattr(e, '__dict__') else e for e in events], indent=2)
        
        prompt = (
            f"Analyze the following AI agent trace for failures or inefficiencies:\n\n"
            f"{trace_text}\n\n"
            f"Identify the EXACT point of failure (event_id) and explain WHY it happened. "
            f"Suggest a specific fix (e.g., update tool schema, adjust soul prompt, or add a checkpoint)."
        )
        
        resp = await self.llm.messages.create(
            model="claude-sonnet-4-6", 
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()

    async def check_drift_alert(self, agent_name: str, metric_key: str = "success_rate") -> bool:
        \"\"\"
        Statistical Drift Detection:
        Compares the last 10 runs with the baseline (last 100).
        Siren if deviation > 15%.
        \"\"\"
        if not self.redis: return False
        
        # Fetch historical metrics from the AutoEval pipeline
        history = await self.redis.lrange(f"adonis:metrics:skills:{agent_name}", 0, 100)
        if not history: return False
        
        data = [json.loads(h) for h in history]
        current_window = data[:10]
        baseline_window = data[10:]
        
        def avg(window): 
            return sum(d.get(metric_key, 0) for d in window) / (len(window) or 1)
        
        current_avg = avg(current_window)
        baseline_avg = avg(baseline_window)
        
        drift = abs(current_avg - baseline_avg) / (baseline_avg or 1)
        if drift > 0.15:
            log.warn(f"[DRIFT] Agent {agent_name} {metric_key} drifted by {drift:.2%}")
            return True
        return False
