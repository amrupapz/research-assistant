"""MCP server (Day 7 stretch): the agent's read-only tools for any MCP client (Claude Code, Desktop).
The ONLY file that imports `mcp`.

    uv run research-mcp --tenant default        (stdio: started BY the client, not by you)
    claude mcp add research-assistant -- uv --directory /path/to/repo run research-mcp --tenant default

The model on the other side is NOT ours: treat every call as untrusted input.
- Only EXPOSED tools are listed or callable. send_email and draft_email are not exposed.
- Every call still goes through the same gate as the workflow: flow.call_tool = validate (Pydantic)
  → authorize() (default-deny; only "allow" runs) → execute in one transaction → traced.
- Input schemas are generated from the same Pydantic models the agent uses (tools.tool_schema),
  so what the client is told and what the server accepts can't drift apart.
- One tenant per server process (--tenant): its connection sees only that tenant's leads/research.
- Tools that call the model check the tenant's daily token budget first and record their usage.
"""

import argparse
import json
import threading
import traceback

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from research_assistant import state, tracing
from research_assistant.config import get_settings
from research_assistant.flow import FlowError, call_tool
from research_assistant.llm import token_meter
from research_assistant.log import log_run
from research_assistant.tools import REGISTRY, ToolContext, ToolError, tool_schema

EXPOSED = ("search_company", "get_lead", "qualify_lead")  # read-only; nothing leaves the building
USES_MODEL = frozenset({"search_company", "qualify_lead"})  # cost money → budget check + metering


def tool_list() -> list[types.Tool]:
    """What tools/list returns: name, description, JSON schema straight from the Pydantic model."""
    return [
        types.Tool(name=n, description=REGISTRY[n].description, input_schema=tool_schema(REGISTRY[n])["parameters"])
        for n in EXPOSED
    ]


def _error(message: str) -> types.CallToolResult:
    """A tool error the calling model can read and correct (is_error=True), not a protocol error."""
    return types.CallToolResult(content=[types.TextContent(text=message)], is_error=True)


def mcp_call(name: str, arguments: dict | None, ctx: ToolContext) -> types.CallToolResult:
    """One tools/call → the gate → an MCP result. Written by Claude at my request (explained line by
    line in LEARNING.md, "MCP stretch").

    Never raises: every failure becomes an is_error result, and its message never contains internals
    (stack traces go to the run log). Order matters: exposure check → budget → gate → result."""
    if name not in EXPOSED:
        return _error(f"unknown tool {name!r}; available: {', '.join(EXPOSED)}")
    try:
        if name in USES_MODEL and (problem := state.daily_limit_problem(ctx.conn, starting_run=False)):
            return _error(problem)
        with token_meter() as used:
            try:
                output = call_tool(ctx, name, arguments or {})
            finally:
                if used[0]:
                    state.record_usage(ctx.conn, used[0], f"mcp:{name}")
    except (ToolError, FlowError) as e:
        return _error(str(e))
    except Exception as e:  # noqa: BLE001 -- e.g. OpenAI down after retries: say what, not how
        log_run({"kind": "mcp_error", "tool": name, "error": traceback.format_exc()}, get_settings().runs_dir)
        return _error(f"{name} failed ({type(e).__name__}); details are in the server's run log")
    data = json.loads(json.dumps(output, default=str))  # dates etc. → strings: valid JSON for the wire
    return types.CallToolResult(content=[types.TextContent(text=json.dumps(data))], structured_content=data)


def build_server(ctx: ToolContext) -> Server:
    """An MCP server bound to one ToolContext (one tenant's connection). Calls run one at a time in a
    worker thread: the tools are blocking (DB + model) and share one connection."""
    lock = threading.Lock()

    def locked_call(name: str, arguments: dict | None) -> types.CallToolResult:
        with lock:
            return mcp_call(name, arguments, ctx)

    async def on_list_tools(_ctx, _params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tool_list())

    async def on_call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        return await anyio.to_thread.run_sync(locked_call, params.name, params.arguments)

    return Server(
        "research-assistant",
        instructions="Company research (cited), sales lead lookup and ICP qualification. Read-only: no tool sends email.",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def main(argv: list[str] | None = None) -> None:
    from research_assistant.db import connect

    p = argparse.ArgumentParser(prog="research-mcp", description="MCP server over stdio.", allow_abbrev=False)
    p.add_argument("--tenant", default="default", help="whose leads and research this server can see")
    args = p.parse_args(argv)

    async def serve(server: Server) -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        with connect(autocommit=True, tenant=args.tenant) as conn:
            anyio.run(serve, build_server(ToolContext(conn=conn)))
    finally:
        tracing.shutdown()


if __name__ == "__main__":
    main()
