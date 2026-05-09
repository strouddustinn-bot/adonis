
import os, json, logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import httpx

log = logging.getLogger("prometheus")

class FuseLevel(Enum):
    GREEN="GREEN"; YELLOW="YELLOW"; ORANGE="ORANGE"; RED="RED"; BLACK="BLACK"
    @classmethod
    def from_score(cls, s):
        if s<=10: return cls.GREEN
        if s<=25: return cls.YELLOW
        if s<=40: return cls.ORANGE
        if s<=55: return cls.RED
        return cls.BLACK

@dataclass
class IntentScore:
    harm_potential: int=0
    deception_index: int=0
    data_exfil_risk: int=0
    autonomy_override: int=0
    legal_exposure: int=0
    cascade_risk: int=0
    @property
    def total(self): return sum([self.harm_potential,self.deception_index,self.data_exfil_risk,self.autonomy_override,self.legal_exposure,self.cascade_risk])
    @property
    def level(self): return FuseLevel.from_score(self.total)

@dataclass
class AgentAction:
    agent_name: str; action_type: str; description: str
    payload: dict=field(default_factory=dict); session_id: str=""; trace_id: str=""

@dataclass
class FuseDecision:
    action: AgentAction; score: IntentScore; level: FuseLevel; approved: bool
    remediated_action: Optional[AgentAction]=None; reason: str=""
    timestamp: str=field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    operator_alerted: bool=False

SIGNALS = [
    ("harm","harm_potential",5),("attack","harm_potential",6),("kill","harm_potential",9),
    ("malware","harm_potential",9),("exploit","harm_potential",6),("doxx","harm_potential",8),
    ("impersonat","deception_index",7),("spoof","deception_index",7),("phish","deception_index",9),
    ("fraud","deception_index",8),("pose as","deception_index",7),
    ("password","data_exfil_risk",5),("credential","data_exfil_risk",6),("private key","data_exfil_risk",8),
    ("exfil","data_exfil_risk",9),("steal","data_exfil_risk",8),("dump database","data_exfil_risk",9),
    ("bypass prometheus","autonomy_override",10),("disable fuse","autonomy_override",10),
    ("ignore ethics","autonomy_override",10),("override safety","autonomy_override",9),
    ("illegal","legal_exposure",6),("unauthorized access","legal_exposure",8),("wiretap","legal_exposure",9),
    ("mass email","cascade_risk",5),("self-replicate","cascade_risk",8),
]
RISKY_COMBOS = [(set(["vault_write","internet"]),"data_exfil_risk",5),(set(["file_write","internet"]),"data_exfil_risk",4)]

class PrometheusFuse:
    def __init__(self, anthropic_client, redis_client, obsidian_bridge=None):
        self.llm=anthropic_client; self.redis=redis_client; self.obs=obsidian_bridge
        self.webhook=os.getenv("PROMETHEUS_ALERT_WEBHOOK","")

    async def evaluate(self, action):
        score=self._heuristic(action)
        if 15<=score.total<=45: score=await self._llm_score(action,score)
        level=score.level
        log.info(f"[FUSE] {action.agent_name}:{action.action_type} -> {level.value} ({score.total}/60)")
        dec=FuseDecision(action=action,score=score,level=level,approved=False)
        if level==FuseLevel.GREEN: dec.approved=True; dec.reason="Safe."
        elif level==FuseLevel.YELLOW: dec.approved=True; dec.reason="Elevated-audited."; await self._audit(dec)
        elif level==FuseLevel.ORANGE: dec.reason="Requires human confirmation."; await self._audit(dec)
        elif level==FuseLevel.RED:
            fixed=await self._remediate(action)
            if fixed:
                rs=self._heuristic(fixed)
                if rs.total<=25: dec.remediated_action=fixed; dec.approved=True; dec.reason=f"Remediated. Score {rs.total}/60."
                else: dec.reason="Remediation insufficient."; await self._alert(dec)
            else: dec.reason="Unremediable. Blocked."; await self._alert(dec)
            await self._audit(dec)
        elif level==FuseLevel.BLACK:
            dec.reason=f"HARD KILL-{action.agent_name} locked."; await self._hard_kill(action.agent_name)
            await self._alert(dec); await self._audit(dec)
        return dec

    def _heuristic(self, action):
        s=IntentScore(); text=f"{action.description} {json.dumps(action.payload)}".lower()
        perms=set(action.payload.get("permissions",[]))
        for kw,axis,inc in SIGNALS:
            if kw in text: setattr(s,axis,min(10,getattr(s,axis)+inc))
        for combo,axis,inc in RISKY_COMBOS:
            if combo.issubset(perms): setattr(s,axis,min(10,getattr(s,axis)+inc))
        return s

    async def _llm_score(self, action, baseline):
        try:
            r=await self.llm.messages.create(model="claude-haiku-4-5-20251001",max_tokens=100,
                messages=[{"role":"user","content":f"Score ethically 0-10 each axis, JSON only: harm_potential deception_index data_exfil_risk autonomy_override legal_exposure cascade_risk. Action: {action.description}"}])
            d=json.loads(r.content[0].text.strip())
            return IntentScore(**{k:max(0,min(10,int(v))) for k,v in d.items()})
        except: return baseline

    async def _remediate(self, action):
        try:
            r=await self.llm.messages.create(model="claude-haiku-4-5-20251001",max_tokens=400,
                messages=[{"role":"user","content":f"Rewrite this action safely, JSON only: action_type, description, payload. If impossible: {{impossible:true}}. Action: {action.description}"}])
            d=json.loads(r.content[0].text.strip())
            if d.get("impossible"): return None
            return AgentAction(agent_name=action.agent_name,action_type=d["action_type"],description=d["description"],payload=d.get("payload",{}),session_id=action.session_id,trace_id=action.trace_id+"_remediated")
        except: return None

    async def _audit(self, dec):
        rec={"ts":dec.timestamp,"agent":dec.action.agent_name,"action":dec.action.action_type,"score":dec.score.total,"level":dec.level.value,"approved":dec.approved,"reason":dec.reason}
        await self.redis.lpush("prometheus:audit",json.dumps(rec))
        await self.redis.ltrim("prometheus:audit",0,999)

    async def _alert(self, dec):
        if not self.webhook: return
        try:
            async with httpx.AsyncClient() as c:
                await c.post(self.webhook,json={"level":dec.level.value,"agent":dec.action.agent_name,"score":dec.score.total,"reason":dec.reason},timeout=5.0)
        except: pass

    async def _hard_kill(self, name):
        await self.redis.set(f"prometheus:locked:{name}",datetime.now(timezone.utc).isoformat())
        log.critical(f"[FUSE] HARD KILL - {name} locked.")

    @staticmethod
    def is_locked(redis_client, name): return bool(redis_client.exists(f"prometheus:locked:{name}"))
