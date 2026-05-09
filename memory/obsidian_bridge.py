"""
memory/obsidian_bridge.py
==========================
HTTP bridge to Obsidian vault via obsidian-local-rest-api plugin.
Install: https://github.com/coddingtonbear/obsidian-local-rest-api

Configure in .env:
  OBSIDIAN_API=http://localhost:27123
  OBSIDIAN_TOKEN=your_token_here
"""
import os, logging
import httpx

log = logging.getLogger("obsidian")

class ObsidianBridge:
    def __init__(self):
        self.base  = os.getenv("OBSIDIAN_API","http://localhost:27123")
        self.token = os.getenv("OBSIDIAN_TOKEN","")
        self._headers = {"Authorization":f"Bearer {self.token}","Content-Type":"text/markdown"}

    async def read_note(self, path: str) -> str | None:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{self.base}/vault/{path}", headers=self._headers, timeout=5.0)
                return r.text if r.status_code == 200 else None
        except Exception as e:
            log.warning(f"[OBS] read_note failed ({path}): {e}"); return None

    async def write_note(self, path: str, content: str) -> bool:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.put(f"{self.base}/vault/{path}", content=content.encode(),
                                headers=self._headers, timeout=5.0)
                return r.status_code in (200,204)
        except Exception as e:
            log.warning(f"[OBS] write_note failed ({path}): {e}"); return False

    async def search(self, query: str) -> list[dict]:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{self.base}/search/simple/",
                                 params={"query":query,"contextLength":100},
                                 headers=self._headers, timeout=5.0)
                return r.json() if r.status_code == 200 else []
        except Exception as e:
            log.warning(f"[OBS] search failed: {e}"); return []

    async def append_note(self, path: str, content: str) -> bool:
        existing = await self.read_note(path) or ""
        return await self.write_note(path, existing + content)
