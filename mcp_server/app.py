"""MCP server (JSON-RPC 2.0) for the supply chain stress-test resolver.

Exposes a single tool, ``run_supply_chain_stress_test``, over the Model Context
Protocol so a Databricks Multi-Agent Supervisor can call the pyomo optimizer via
a Unity Catalog HTTP (MCP) connection.

Endpoint: ``POST /api/mcp`` implementing the ``initialize``, ``tools/list`` and
``tools/call`` methods. A ``GET /health`` route is provided for readiness checks.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import resolver

app = FastAPI(title="Supply Chain Stress-Test MCP Server")

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "run_supply_chain_stress_test",
        "description": (
            "Run a supply chain stress-test optimization for a disruption "
            "scenario. Solves a time-to-recover (TTR) profit-loss model with and "
            "without the disruption plus a time-to-survive (TTS) model, using the "
            "network topology and operational data in Unity Catalog. Use this for "
            "any question about what happens when one or more nodes go down, how "
            "long recovery takes, resulting profit loss, or mitigation "
            "recommendations. Node IDs look like T1_5 (product), T2_4 (direct "
            "supplier), T3_10 (sub-supplier)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "disrupted": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of disrupted node IDs, e.g. [\"T2_4\"]. These nodes "
                        "produce nothing during the recovery window."
                    ),
                },
                "ttr": {
                    "type": "number",
                    "description": (
                        "Time to recover: number of time units the disrupted "
                        "node(s) take to return to normal operation."
                    ),
                },
            },
            "required": ["disrupted", "ttr"],
        },
    }
]


def _result(request_id: Any, result: Any) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/mcp")
async def mcp(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return _error(None, -32700, "Parse error")

    method = body.get("method")
    request_id = body.get("id")
    params = body.get("params") or {}

    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "supply-chain-stress-test",
                    "version": "1.0.0",
                },
            },
        )

    if method in ("notifications/initialized", "initialized"):
        # Notification: no response body expected.
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {}})

    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name != "run_supply_chain_stress_test":
            return _error(request_id, -32601, f"Unknown tool: {name}")

        disrupted = arguments.get("disrupted", [])
        ttr = arguments.get("ttr")
        if not isinstance(disrupted, list) or ttr is None:
            return _error(
                request_id,
                -32602,
                "Invalid params: 'disrupted' (list) and 'ttr' (number) are required.",
            )

        try:
            output = resolver.run_stress_test(disrupted, float(ttr))
        except Exception as exc:  # surface solver/data errors as tool content
            return _result(
                request_id,
                {
                    "content": [{"type": "text", "text": f"Error running stress test: {exc}"}],
                    "isError": True,
                },
            )

        return _result(
            request_id,
            {"content": [{"type": "text", "text": output}], "isError": False},
        )

    return _error(request_id, -32601, f"Method not found: {method}")
