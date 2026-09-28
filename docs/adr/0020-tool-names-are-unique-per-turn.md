---
status: accepted
date: 2026-09-28
---

# 0020. A tool name is unique among the tools exposed in a turn

## Context

The model calls a tool by name, and a `ToolCallEvent` reaches every tool subscribed to
that name on the stream. Per-tool state keyed by name, such as `approval_required`'s
"always" answer, assumes one tool behind each name. Tools reach a turn from several
sources — `Agent(tools=...)`, `ask(tools=...)`, toolkits, plugins, sub-task and knowledge
tools, and names an `MCPToolkit` discovers at runtime — and nothing tied them together, so
two tools with one name both ran on a single call, and a human's decision about one of
them did not bind the other.

The options were: dispatch by a per-registration id instead of by name, disambiguate
duplicates automatically (e.g. a forced source prefix), or require unique names.

## Decision

The names a turn exposes to the model must be unique. `resolve_tool_schemas`
(`ag2/tools/executor.py`) resolves the schemas of every tool for the turn — after dynamic
discovery — and raises `ToolConflictError` naming both sources when a function name is
exposed twice or equals a built-in tool's name. `Agent` and `LiveAgent` both assemble
their tools through it. Built-in schemas of one type may repeat (several `MCPServerTool`
servers), since the provider, not ag2, tells those apart.

Dispatch by id was rejected: the model still sees only names, so two same-name schemas
stay indistinguishable to it and to the human asked to approve a call. Automatic prefixes
were rejected because they silently rename tools the prompt or the user refers to;
`MCPToolkit`'s explicit `tool_name_prefix` already covers the legitimate need.

## Consequences

- A turn with a duplicate name fails before any model call, rather than running several
  implementations for one call.
- Name-keyed state (the approval bypass, `known_tools`) can rely on a
  name identifying one tool within a turn.
- An MCP server that starts exposing a name already in use fails the next turn; the fix is
  a `tool_name_prefix` or renaming the local tool.
