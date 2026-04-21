# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.annotations import Variable
from autogen.beta.events import ToolCallEvent
from autogen.beta.events.tool_events import ToolErrorEvent
from autogen.beta.tools import DuckDuckSearchTool


@pytest.mark.asyncio
class TestDuckDuckSearchToolVariable:
    # Client-side tool: Variables are resolved inside the function body at call time,
    # not in `schemas()`. FunctionTool.__call__ catches exceptions into ToolErrorEvent,
    # so a missing variable surfaces as an error event rather than a raised KeyError.

    async def test_resolved(self, make_context: Callable[..., Context]) -> None:
        mock_client = MagicMock()
        mock_client.text.return_value = []
        ctx = make_context(result_limit=7, region="ru-ru")
        tool = DuckDuckSearchTool(
            max_results=Variable("result_limit"),
            region=Variable("region"),
            client=mock_client,
        )

        event = ToolCallEvent(arguments=json.dumps({"query": "ag2"}), name="duckduckgo_search")
        await tool._tool(event, ctx)

        mock_client.text.assert_called_once_with("ag2", region="ru-ru", safesearch="moderate", max_results=7)

    async def test_missing_raises(self, context: Context) -> None:
        tool = DuckDuckSearchTool(max_results=Variable("result_limit"), client=MagicMock())

        event = ToolCallEvent(arguments=json.dumps({"query": "ag2"}), name="duckduckgo_search")
        result = await tool._tool(event, context)

        assert isinstance(result, ToolErrorEvent)
        assert isinstance(result.error, KeyError)
        assert "result_limit" in str(result.error)
