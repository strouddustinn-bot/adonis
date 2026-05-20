
"""
tools/discovery.py
======================
Autonomous Schema Discovery.
Fetches OpenAPI/Swagger specs and converts them into Adonis ToolSchemas.
"""
import httpx, json, logging
from typing import List, Dict, Any
from tools.registry import ToolSchema

log = logging.getLogger("discovery")

class SchemaDiscoverer:
    def __init__(self, registry):
        self.registry = registry

    async def discover_from_url(self, url: str, api_name: str):
        """Fetches an OpenAPI spec and auto-generates a ToolSchema."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                spec = resp.json()
            
            # Simplify OpenAPI spec into a a set of ToolSchemas
            # This assumes the spec is in OpenAPI 3.0 format
            paths = spec.get("paths", {})
            for path, methods in paths.items():
                for method, details in methods.items():
                    op_id = details.get("operationId", f"{method}_{path}")
                    
                    # Extract input schema (request body or query params)
                    input_schema = {}
                    # Handle requestBody
                    if "requestBody" in details:
                        # Simplified extraction of JSON schema from content/application/json/schema
                        content_type = details["requestBody"].get("content", {})
                        json_schema = content_type.get("application/json", {}).get("schema", {})
                        input_schema = json_schema
                    
                    # Extract response schema (200 OK)
                    response_schema = {}
                    responses = details.get("responses", {})
                    ok_resp = responses.get("200", {})
                    content_type = ok_resp.get("content", {})
                    json_schema = content_type.get("application/json", {}).get("schema", {})
                    response_schema = json_schema

                    # Register as a discrete tool
                    schema = ToolSchema(
                        name=op_id,
                        endpoint=f"{spec.get('servers', [{}])[0].get('url', '')}{path}",
                        input_schema=input_schema,
                        response_schema=response_schema
                    )
                    self.registry.register_tool(schema)
                    log.info(f"[DISCOVERY] Auto-registered tool: {op_id}")
            
            return True
        except Exception as e:
            log.error(f"[DISCOVERY] Discovery failed for {url}: {e}")
            return False
