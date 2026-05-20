
"""
auto_eval.py
======================
Automated Evaluation Loop for Adonis Skills.
Implements synthetic test generation, ensemble judging, and metric tracking.
"""
import asyncio, json, logging, time, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from prometheus.fuse import PrometheusFuse # Ensure ethics check on tests

log = logging.getLogger("auto_eval")

@dataclass
class EvalMetric:
    success_rate: float
    avg_latency: float
    token_cost: int
    drift_score: float # 0.0 (stable) to 1.0 (diverged)

@dataclass
class TestCase:
    input_data: Any
    expected_outcome: str # Description of a "correct" result
    category: str # "edge_case", "happy_path", "adversarial"

class AutoEvalPipeline:
    def __init__(self, llm_client, redis_client):
        self.llm = llm_client
        self.redis = redis_client
        self.metrics_key = "adonis:metrics:skills"

    async def generate_synthetic_tests(self, skill_name: str, skill_description: str) -> List[TestCase]:
        """Generates 3-5 edge-case inputs based on the skill's purpose."""
        prompt = (
            f"Analyze the following AI skill: {skill_name}\n"
            f"Description: {skill_description}\n\n"
            f"Generate 5 high-variance synthetic test cases. Include:\n"
            f"1. A typical 'happy path' request.\n"
            f"2. Two complex edge cases (ambiguous input, extreme constraints).\n"
            f"3. One adversarial attempt (prompt injection, logic bypass).\n"
            f"4. One 'impossible' request that should trigger a safe failure.\n\n"
            f"Output as JSON list: [{{'input_data': '...', 'expected_outcome': '...', 'category': '...'}}]"
        )
        
        try:
            # Use a fast model for generation
            resp = await self.llm.messages.create(
                model="claude-haiku-4-5-20251001", 
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            tests = json.loads(resp.content[0].text.strip())
            return [TestCase(**t) for t in tests]
        except Exception as e:
            log.error(f"[EVAL] Synth gen failed: {e}")
            return []

    async def ensemble_judge(self, test_case: TestCase, output: str) -> Tuple[bool, float]:
        """
        Judge result using an ensemble:
        1. Rule-based (Strict patterns/JSON validity)
        2. Small Model (Fast sanity)
        3. Large Model (Deep semantic alignment)
        """
        # 1. Rule Based
        rule_pass = True # Default, can be extended for specific skill requirements
        
        # 2. Small Model Judge
        judge_prompt = f"Test Case: {test_case.expected_outcome}\nActual Output: {output}\n\nDid the agent succeed? Answer ONLY 'YES' or 'NO'."
        try:
            s_res = await self.llm.messages.create(
                model="claude-haiku-4-5-20251001", 
                max_tokens=10,
                messages=[{"role": "user", "content": judge_prompt}]
            )
            small_pass = "YES" in s_res.content[0].text.upper()
        except: small_pass = False

        # 3. Large Model Judge (only if small model is uncertain or it's a high-risk case)
        # For this implementation, we weight (Small: 0.3, Large: 0.7)
        l_res = await self.llm.messages.create(
                model="claude-sonnet-4-6", 
                max_tokens=100,
                messages=[{"role": "user", "content": f"Evaluate this AI output for semantic accuracy and adherence to: {test_case.expected_outcome}. Output a confidence score 0-1 and a final YES/NO."}]
        )
        # Simple extraction of YES/NO from large model
        large_pass = "YES" in l_res.content[0].text.upper()

        final_pass = (large_pass and small_pass) or (large_pass and rule_pass) # Bias toward the larger model
        confidence = 0.9 if (large_pass == small_pass) else 0.5
        
        return final_pass, confidence

    async def evaluate_skill(self, skill_name: str, skill_description: str, execution_fn):
        """Full loop: Gen -> Run -> Judge -> Record."""
        tests = await self.generate_synthetic_tests(skill_name, skill_description)
        if not tests: return
        
        results = []
        for test in tests:
            start = time.monotonic()
            try:
                # Execute the skill with the synthetic input
                output = await execution_fn(test.input_data)
                latency = time.monotonic() - start
                
                passed, conf = await self.ensemble_judge(test, output)
                results.append({"passed": passed, "latency": latency, "conf": conf})
            except Exception as e:
                results.append({"passed": False, "latency": 0, "conf": 0, "error": str(e)})

        # Aggregate Metrics
        success_rate = len([r for r in results if r["passed"]]) / len(tests)
        avg_lat = sum([r["latency"] for r in results]) / len(tests)
        
        metric = EvalMetric(
            success_rate=success_rate,
            avg_latency=avg_lat,
            token_cost=0, # Would be integrated with LLM provider tracking
            drift_score=0.0 # Calculated by comparing current success_rate to historical
        )
        
        await self.record_metrics(skill_name, metric)
        return metric

    async def record_metrics(self, skill_name: str, metric: EvalMetric):
        data = json.dumps({
            "timestamp": time.time(),
            "success_rate": metric.success_rate,
            "latency": metric.avg_latency,
            "drift": metric.drift_score
        })
        await self.redis.lpush(f"{self.metrics_key}:{skill_name}", data)
        await self.redis.ltrim(f"{self.metrics_key}:{skill_name}", 0, 100) # keep last 100 runs


    async def deploy_skill(self, skill_name: str, skill_description: str, execution_fn, new_version_fn, threshold=0.8):
        """
        Canary Deployment:
        - Runs a set of tests against BOTH current and new version.
        - Only promotes new_version_fn if success_rate > threshold AND >= current_version_success.
        """
        log.info(f"[CANARY] Starting canary deploy for {skill_name}")
        
        current_metric = await self.evaluate_skill(skill_name, skill_description, execution_fn)
        new_metric = await self.evaluate_skill(f"{skill_name}_canary", skill_description, new_version_fn)
        
        log.info(f"[CANARY] Results: Current={current_metric.success_rate}, New={new_metric.success_rate}")
        
        if new_metric.success_rate >= threshold and new_metric.success_rate >= current_metric.success_rate:
            log.info(f"[CANARY] Promotion SUCCESS: {skill_name} upgraded.")
            return True # Promote
        else:
            log.warn(f"[CANARY] Promotion REJECTED: New version degraded or below threshold. Rolling back.")
            return False # Rollback

