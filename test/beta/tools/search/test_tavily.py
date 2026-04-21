# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
from dirty_equals import IsPartialDict

from autogen.beta import Agent, MemoryStream
from autogen.beta.context import ConversationContext
from autogen.beta.events import ToolCallEvent, ToolCallsEvent, ToolResultEvent
from autogen.beta.events.types import ModelResponse
from autogen.beta.testing import TestConfig
from autogen.beta.tools.search import SearchResponse, SearchResult, TavilySearchTool

SAMPLE_RAW = {
    "query": "AG2 framework",
    "answer": "AG2 is an open-source multi-agent framework.",
    "results": [
        {
            "title": "AG2 Framework",
            "url": "https://ag2.ai",
            "content": "AG2 is an agent framework.",
            "score": 0.95,
            "raw_content": "# AG2\nFull text",
            "favicon": "https://ag2.ai/favicon.ico",
        },
        {
            "title": "GitHub - AG2",
            "url": "https://github.com/ag2ai/ag2",
            "content": "Open source repo.",
            "score": 0.82,
            "raw_content": None,
            "favicon": None,
        },
    ],
    "images": ["https://ag2.ai/img.png"],
}


def _make_config(query: str, tool_name: str = "tavily_search", final_reply: str = "done") -> TestConfig:
    call = ToolCallEvent(arguments=json.dumps({"query": query}), name=tool_name)
    return TestConfig(ModelResponse(tool_calls=ToolCallsEvent([call])), final_reply)


def _collect_results(stream: MemoryStream) -> list[SearchResponse]:
    results: list[SearchResponse] = []
    stream.where(ToolResultEvent).subscribe(lambda e: results.append(e.result.content))
    return results


@pytest.mark.asyncio
class TestSchema:
    async def test_defaults(self, context: ConversationContext) -> None:
        tavily = TavilySearchTool(client=MagicMock())

        [schema] = await tavily.schemas(context)

        assert schema.function.name == "tavily_search"
        assert schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_name_and_description(self, context: ConversationContext) -> None:
        tavily = TavilySearchTool(client=MagicMock(), name="my_search", description="Custom search tool.")

        [schema] = await tavily.schemas(context)

        assert schema.function.name == "my_search"
        assert schema.function.description == "Custom search tool."


@pytest.mark.asyncio
class TestSearchExecution:
    async def test_search_returns_structured_results(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = SAMPLE_RAW

        tavily = TavilySearchTool(client=mock_client)
        agent = Agent("a", config=_make_config("AG2 framework"), tools=[tavily])
        stream = MemoryStream()
        results = _collect_results(stream)

        await agent.ask("search", stream=stream)

        assert results == [
            SearchResponse(
                query="AG2 framework",
                results=[
                    SearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        snippet="AG2 is an agent framework.",
                        score=0.95,
                        raw_content="# AG2\nFull text",
                        favicon="https://ag2.ai/favicon.ico",
                    ),
                    SearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        snippet="Open source repo.",
                        score=0.82,
                    ),
                ],
                answer="AG2 is an open-source multi-agent framework.",
                images=["https://ag2.ai/img.png"],
            )
        ]
        mock_client.search.assert_called_once_with("AG2 framework")

    async def test_search_empty_results(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {"query": "nothing", "results": []}

        tavily = TavilySearchTool(client=mock_client)
        agent = Agent("a", config=_make_config("nothing"), tools=[tavily])
        stream = MemoryStream()
        results = _collect_results(stream)

        await agent.ask("search", stream=stream)

        assert results == [SearchResponse(query="nothing", results=[])]

    async def test_all_params_forwarded_to_client(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {"query": "q", "results": []}

        tavily = TavilySearchTool(
            client=mock_client,
            max_results=3,
            search_depth="advanced",
            topic="news",
            include_answer=True,
            include_raw_content="markdown",
            include_images=True,
            time_range="week",
            start_date="2024-01-01",
            end_date="2024-12-31",
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            country="US",
            auto_parameters=True,
            include_favicon=True,
        )
        agent = Agent("a", config=_make_config("q"), tools=[tavily])

        await agent.ask("search")

        mock_client.search.assert_called_once_with(
            "q",
            max_results=3,
            search_depth="advanced",
            topic="news",
            include_answer=True,
            include_raw_content="markdown",
            include_images=True,
            time_range="week",
            start_date="2024-01-01",
            end_date="2024-12-31",
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            country="US",
            auto_parameters=True,
            include_favicon=True,
        )

    async def test_none_params_omitted(self) -> None:
        # All optional params default to None and must not be forwarded to the SDK
        # so Tavily applies its own server-side defaults.
        mock_client = MagicMock()
        mock_client.search.return_value = {"query": "q", "results": []}

        tavily = TavilySearchTool(client=mock_client)
        agent = Agent("a", config=_make_config("q"), tools=[tavily])

        await agent.ask("search")

        mock_client.search.assert_called_once_with("q")

    async def test_custom_tool_name_routes_correctly(self) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = SAMPLE_RAW

        tavily = TavilySearchTool(client=mock_client, name="web_search")
        agent = Agent("a", config=_make_config("AG2 framework", tool_name="web_search"), tools=[tavily])
        stream = MemoryStream()
        results = _collect_results(stream)

        await agent.ask("search", stream=stream)

        assert results == [
            SearchResponse(
                query="AG2 framework",
                results=[
                    SearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        snippet="AG2 is an agent framework.",
                        score=0.95,
                        raw_content="# AG2\nFull text",
                        favicon="https://ag2.ai/favicon.ico",
                    ),
                    SearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        snippet="Open source repo.",
                        score=0.82,
                    ),
                ],
                answer="AG2 is an open-source multi-agent framework.",
                images=["https://ag2.ai/img.png"],
            )
        ]
