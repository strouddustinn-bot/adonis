"""
hermes/telegram.py
===================
Telegram chat bridge — long polling, no public URL required.

Connects a Telegram bot to the same Glasswing/Atlas pipeline that
serves the web UI. One Adonis session per Telegram user (`tg_<user_id>`)
so memory carries across messages.

Configure in .env:
  TELEGRAM_BOT_TOKEN=<from @BotFather>
  TELEGRAM_ALLOWED_USER_IDS=<your numeric id>[,<another>...]

Find your user id by messaging @userinfobot (Telegram). Anyone who
DM's the bot whose id is NOT in the allowlist gets a one-line rejection
plus a hint with their id — copy-paste, add to .env, restart.

Telegram commands handled:
  /start    introduction
  /help     command list
  /ask msg  force ask mode
  /task msg force task mode (multi-agent via Atlas)
  /health   probe runtime services
  /ges      Glasswing efficiency scores per agent
  /facts    last 5 active facts
  /reset    new session id

Anything else is sent through `/ask` so users can just type freely.
"""
import asyncio
import json
import logging
import os
import uuid
from typing import Optional

import httpx

log = logging.getLogger("telegram")

API     = "https://api.telegram.org/bot{token}/{method}"
POLL_S  = 30                  # long-poll timeout (server side)
LIMIT   = 4096                # Telegram per-message char limit
ASK_TO  = 90                  # seconds to wait for an /ask response
TASK_TO = 180                 # seconds to wait for a /task synthesis


class TelegramBridge:
    def __init__(
        self,
        *,
        token: str,
        allowed_user_ids: list[int],
        governor,
        llm,
        model: str,
        redis,
    ):
        self.token    = token
        self.allowed  = set(int(x) for x in allowed_user_ids)
        self.governor = governor
        self.llm      = llm
        self.model    = model
        self.redis    = redis
        self.offset   = 0
        self.sessions: dict[int, str] = {}
        self._stopping = False

    # ── lifecycle ────────────────────────────────────────────────────────

    async def run(self) -> None:
        info = await self._api("getMe", method="GET")
        if not info or not info.get("ok"):
            log.error("Telegram getMe failed: %s", info)
            return
        bot = info["result"]
        log.info("Telegram bridge online as @%s (id=%s)", bot.get("username"), bot.get("id"))
        if not self.allowed:
            log.warning("TELEGRAM_ALLOWED_USER_IDS is empty — bot will reject all messages until you allowlist a user id.")

        backoff = 1.0
        while not self._stopping:
            try:
                updates = await self._poll()
                backoff = 1.0
                for upd in updates:
                    asyncio.create_task(self._handle(upd))
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Telegram poll error: %s — backing off %.1fs", e, backoff)
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(30.0, backoff * 2)

    def stop(self) -> None:
        self._stopping = True

    # ── transport ────────────────────────────────────────────────────────

    async def _api(self, method: str, *, params=None, data=None, method_http="POST", method_get=False) -> Optional[dict]:
        url = API.format(token=self.token, method=method)
        try:
            async with httpx.AsyncClient(timeout=POLL_S + 10) as c:
                if method_get or method_http == "GET":
                    r = await c.get(url, params=params)
                else:
                    r = await c.post(url, data=data or params or {})
            return r.json()
        except Exception as e:
            log.debug("telegram %s failed: %s", method, e)
            return None

    async def _poll(self) -> list[dict]:
        resp = await self._api("getUpdates",
                               params={"offset": self.offset, "timeout": POLL_S},
                               method_http="GET", method_get=True)
        if not resp or not resp.get("ok"):
            return []
        ups = resp.get("result", []) or []
        if ups:
            self.offset = max(u["update_id"] for u in ups) + 1
        return ups

    async def _send(self, chat_id: int, text: str) -> None:
        if not text: text = "(empty)"
        for i in range(0, len(text), LIMIT):
            chunk = text[i:i + LIMIT]
            await self._api("sendMessage", data={"chat_id": chat_id, "text": chunk})

    async def _typing(self, chat_id: int) -> None:
        await self._api("sendChatAction", data={"chat_id": chat_id, "action": "typing"})

    # ── handling ─────────────────────────────────────────────────────────

    def _session(self, user_id: int) -> str:
        sid = self.sessions.get(user_id)
        if not sid:
            sid = f"tg_{user_id}"
            self.sessions[user_id] = sid
        return sid

    async def _handle(self, upd: dict) -> None:
        msg = upd.get("message") or upd.get("edited_message")
        if not msg: return
        chat_id = (msg.get("chat") or {}).get("id")
        user    = msg.get("from") or {}
        uid     = user.get("id")
        text    = (msg.get("text") or "").strip()
        if not chat_id or not uid or not text: return

        if not self.allowed:
            await self._send(chat_id,
                f"This Adonis instance has no allowed users configured.\n"
                f"Your Telegram user id is {uid}.\n"
                f"Add it to TELEGRAM_ALLOWED_USER_IDS in .env and restart.")
            return
        if uid not in self.allowed:
            log.warning("Rejected message from unauthorized user %s (@%s)", uid, user.get("username", "?"))
            await self._send(chat_id, "You're not authorized to talk to this Adonis instance.")
            return

        if text.startswith("/"):
            await self._command(chat_id, uid, text)
        else:
            await self._ask(chat_id, uid, text)

    async def _command(self, chat_id: int, uid: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        # Strip @botname suffix Telegram adds in group chats.
        cmd = parts[0].split("@", 1)[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/start":
            return await self._send(chat_id,
                "Adonis online.\n\n"
                "Type anything for a single-turn answer.\n"
                "Use /task <goal> for multi-agent orchestration.\n"
                "/help for the full command list.")
        if cmd == "/help":
            return await self._send(chat_id,
                "Commands:\n"
                "  /ask <msg>   single-turn (default — just type)\n"
                "  /task <goal> multi-agent via Atlas\n"
                "  /health      services status\n"
                "  /ges         agent efficiency scores\n"
                "  /facts       last 5 active facts\n"
                "  /reset       new session id")
        if cmd == "/ask":
            if not arg: return await self._send(chat_id, "Usage: /ask <message>")
            return await self._ask(chat_id, uid, arg)
        if cmd == "/task":
            if not arg: return await self._send(chat_id, "Usage: /task <goal>")
            return await self._task(chat_id, uid, arg)
        if cmd == "/health":
            return await self._cmd_health(chat_id)
        if cmd == "/ges":
            return await self._cmd_ges(chat_id)
        if cmd == "/facts":
            return await self._cmd_facts(chat_id)
        if cmd == "/reset":
            self.sessions.pop(uid, None)
            return await self._send(chat_id, "New session started.")
        return await self._send(chat_id, f"Unknown command: {cmd} — try /help")

    # ── /ask path ────────────────────────────────────────────────────────
    async def _ask(self, chat_id: int, uid: int, text: str) -> None:
        await self._typing(chat_id)
        session_id = self._session(uid)
        ctx = self.governor.context
        try:
            await ctx.append_turn(session_id, "user", text)
            prep = await self.governor.prepare(session_id, text)
            if prep.cache_hit:
                await ctx.append_turn(session_id, "assistant", prep.cached_result)
                return await self._send(chat_id, prep.cached_result)
            r = await self.llm.messages.create(
                model=self.model, max_tokens=1024,
                system=prep.system, messages=prep.messages,
            )
            answer = r.content[0].text if r.content else "(empty)"
            tokens = getattr(r, "usage", None)
            tokens_used = (tokens.input_tokens + tokens.output_tokens) if tokens else 1
            await self.governor.cache_result(text, answer)
            await ctx.append_turn(session_id, "assistant", answer)
            self.governor.record_ges(
                session_id, prep.active_agents,
                tokens_used=max(1, tokens_used), tasks_done=1,
                cache_hit=False, speed_ms=prep.efficiency.get("prep_ms", 0),
            )
            await self._send(chat_id, answer)
        except Exception as e:
            log.warning("telegram /ask failed for uid=%s: %s", uid, e)
            await self._send(chat_id, f"Error: {e}")

    # ── /task path ───────────────────────────────────────────────────────
    async def _task(self, chat_id: int, uid: int, goal: str) -> None:
        await self._typing(chat_id)
        session_id = self._session(uid)
        trace_id   = uuid.uuid4().hex[:10]
        result_key = f"tg:result:{trace_id}"
        payload = {
            "type":       "task",
            "goal":       goal,
            "session_id": session_id,
            "trace_id":   trace_id,
            "result_key": result_key,
        }
        await self.redis.publish("adonis:agent:atlas", json.dumps(payload))

        deadline = asyncio.get_event_loop().time() + TASK_TO
        last_typing = 0.0
        while asyncio.get_event_loop().time() < deadline:
            raw = await self.redis.get(result_key)
            if raw:
                await self.redis.delete(result_key)
                try:
                    r = json.loads(raw)
                except Exception:
                    return await self._send(chat_id, raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw))
                synth = (r.get("synthesis")
                         or (r.get("result") if isinstance(r.get("result"), str) else None)
                         or json.dumps(r, indent=2)[:LIMIT])
                return await self._send(chat_id, synth)
            now = asyncio.get_event_loop().time()
            if now - last_typing >= 4.5:
                await self._typing(chat_id)  # Telegram clears typing every 5s
                last_typing = now
            await asyncio.sleep(0.5)
        await self._send(chat_id, f"Atlas did not synthesise within {TASK_TO}s.")

    # ── small command helpers ────────────────────────────────────────────
    async def _cmd_health(self, chat_id: int) -> None:
        try:
            ok = await self.redis.ping()
        except Exception as e:
            ok = False
        ctx = self.governor.context
        chroma_ok = bool(getattr(ctx, "chroma", None))
        obs_ok    = bool(getattr(ctx, "obs", None))
        await self._send(chat_id,
            f"redis    {'ok' if ok else 'down'}\n"
            f"chroma   {'ok' if chroma_ok else 'off'}\n"
            f"obsidian {'ok' if obs_ok else 'off'}")

    async def _cmd_ges(self, chat_id: int) -> None:
        try:
            report = await self.governor.get_ges_report()
            lines = [f"{k:<10} {v if v is None else round(v,2)}" for k, v in sorted(report.items())]
            await self._send(chat_id, "\n".join(lines) or "no data yet")
        except Exception as e:
            await self._send(chat_id, f"ges error: {e}")

    async def _cmd_facts(self, chat_id: int) -> None:
        fg = getattr(self.governor, "fact_graph", None)
        if not fg:
            return await self._send(chat_id, "fact graph not initialised")
        try:
            facts = await fg.recent(n=5)
            if not facts: return await self._send(chat_id, "no facts yet")
            lines = [f"{f['entity']}.{f['attribute']} = {str(f['value'])[:80]}  (conf {round(f['confidence'],2)})" for f in facts]
            await self._send(chat_id, "\n".join(lines))
        except Exception as e:
            await self._send(chat_id, f"facts error: {e}")
