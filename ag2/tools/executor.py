# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from typing import Any

from fast_depends.library.serializer import SerializerProto

from ag2.annotations import Context
from ag2.events import (
    ClientToolCallEvent,
    DataInput,
    ModelMessage,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolNotFoundEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.exceptions import HumanInputError, ToolConflictError, ToolNotFoundError
from ag2.middleware import BaseMiddleware

from .final import FunctionToolSchema
from .schemas import ToolSchema
from .tool import Tool


async def resolve_tool_schemas(
    tools: Iterable["Tool"],
    context: "Context",
) -> tuple[list["ToolSchema"], set[str]]:
    """Resolve the schemas ``tools`` expose for a turn and the names calls reach them by.

    A ``ToolCallEvent`` is dispatched by name to every tool subscribed to it,
    and the model tells tools apart only by name, so each callable name must
    belong to exactly one tool. A function name exposed twice, or equal to a
    built-in tool's name, raises :class:`~ag2.exceptions.ToolConflictError`
    naming both sources. Built-in schemas of one type may repeat (e.g. several
    ``MCPServerTool`` servers).

    Runs on the tools resolved for the turn, so names a toolkit discovers at
    runtime (e.g. from an MCP server) are checked too.
    """
    schemas: list[ToolSchema] = []
    functions: dict[str, Tool] = {}
    builtins: dict[str, Tool] = {}
    for tool in tools:
        for schema in await tool.schemas(context):
            schemas.append(schema)
            if isinstance(schema, FunctionToolSchema):
                name = schema.function.name
                owner = functions.get(name, builtins.get(name))
                if owner is not None:
                    raise ToolConflictError(name, sources=(_describe_tool(owner), _describe_tool(tool)))
                functions[name] = tool
            else:
                owner = functions.get(schema.type)
                if owner is not None:
                    raise ToolConflictError(schema.type, sources=(_describe_tool(owner), _describe_tool(tool)))
                builtins.setdefault(schema.type, tool)
    return schemas, functions.keys() | builtins.keys()


def _describe_tool(tool: "Tool") -> str:
    return f"{type(tool).__name__}({tool.name!r})"


class ToolExecutor:
    def __init__(self, serializer: SerializerProto) -> None:
        self.__serializer = serializer

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        tools: Iterable["Tool"] = (),
        known_tools: Iterable[str] = (),
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        stack.enter_context(context.stream.where(ToolCallsEvent).sub_scope(self.execute_tools))

        for tool in tools:
            tool.register(stack, context, middleware=middleware)

        # fallback subscriber to raise NotFound event
        stack.enter_context(
            context.stream.where(ToolCallEvent).sub_scope(_tool_not_found(known_tools)),
        )

    async def execute_tools(self, event: ToolCallsEvent, context: Context) -> None:
        results: list[ToolErrorEvent | ToolResultEvent] = []
        client_calls: list[ClientToolCallEvent] = []

        # Execute called tools in parallel
        tasks = [asyncio.ensure_future(_execute_call(context, call)) for call in event.calls]
        try:
            outcomes = await asyncio.gather(*tasks)
        except HumanInputError:
            # ``gather`` propagates the first failure but leaves the rest of the
            # batch running detached, so a sibling would carry on — side effects
            # included — after the caller has been told the turn failed. Stop
            # them and settle the batch before the exception leaves, which is
            # also what makes the history repair at the turn boundary
            # deterministic.
            #
            # Cancellation reaches a sibling's tool coroutine as an ordinary
            # ``CancelledError``, which is not an ``Exception``, so nothing
            # downstream turns it into a ``ToolErrorEvent``.
            #
            # A *sync* tool is the limit of this: it runs in a worker thread via
            # ``sync_to_thread``, and a thread cannot be cancelled. Its await
            # point goes away immediately, so nothing it returns is ever sent or
            # recorded — but a side effect already under way still lands, after
            # the caller has been told. Do not read the settling below as a
            # guarantee that every side effect was stopped; it guarantees that
            # none of them is still owed a result.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for event in outcomes:
            match event:
                case ClientToolCallEvent() as ev:
                    client_calls.append(ev)

                case ToolErrorEvent() as ev:
                    results.append(ev)

                case ToolResultEvent(result=result) as ev:
                    if result.final:
                        if len(result.parts) != 1:
                            raise ValueError("ToolResult with final=True must have exactly one part")
                        part = result.parts[0]
                        if isinstance(part, TextInput):
                            content = part.content
                        elif isinstance(part, DataInput):
                            content = self.__serializer.encode(part.data).decode()
                        else:
                            raise ValueError(f"Unsupported part type: {type(part)}")

                        await context.send(
                            ModelResponse(
                                message=ModelMessage(
                                    content,
                                    metadata=result.metadata,
                                ),
                                response_force=True,
                            )
                        )
                        return
                    else:
                        results.append(ev)

                case ev:
                    results.append(ev)

        if client_calls:
            await context.send(
                ModelResponse(
                    tool_calls=ToolCallsEvent(client_calls),
                    response_force=True,
                )
            )

        else:
            await context.send(ToolResultsEvent(results))


async def _execute_call(
    context: Context, call: ToolCallEvent
) -> ToolErrorEvent | ToolResultEvent | ClientToolCallEvent:
    async with context.stream.get(
        (ToolErrorEvent.parent_id == call.id)
        | (ToolResultEvent.parent_id == call.id)
        | (ClientToolCallEvent.id == call.id)
    ) as result:
        try:
            await context.send(call)
            return await result

        # Same reasoning as in FunctionTool: a middleware that asked for human
        # input and got nowhere has not produced a tool failure, and an approval
        # that was never asked for must not read as one.
        except HumanInputError:
            raise

        # tool-level middleware could leads to execution exceptions
        except Exception as e:
            return ToolErrorEvent.from_call(call, e)


def _tool_not_found(known_tools: Iterable[str]) -> Callable[..., Any]:
    async def _tool_not_found(event: "ToolCallEvent", context: "Context") -> None:
        if event.name not in known_tools:
            err = ToolNotFoundError(event.name)
            # Build via from_call so the event always carries a populated
            # ``result`` (the formatted error). Constructing it by hand is what
            # left ``result`` as ``None`` and crashed the provider mappers.
            await context.send(ToolNotFoundEvent.from_call(event, err))

    return _tool_not_found
