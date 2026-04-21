# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency


try:
    from .tavily import SearchResponse, SearchResult, TavilySearchTool
except ImportError as e:
    TavilySearchTool = missing_optional_dependency("TavilySearchTool", "tavily", e)  # type: ignore[misc]
    SearchResult = missing_optional_dependency("SearchResult", "tavily", e)  # type: ignore[misc]
    SearchResponse = missing_optional_dependency("SearchResponse", "tavily", e)  # type: ignore[misc]


try:
    from .duckduckgo import DuckDuckSearchTool, SearchResponse, SearchResult
except ImportError as e:
    DuckDuckSearchTool = missing_optional_dependency("DuckDuckSearchTool", "ddgs", e)  # type: ignore[misc]
    SearchResult = missing_optional_dependency("SearchResult", "ddgs", e)  # type: ignore[misc]
    SearchResponse = missing_optional_dependency("SearchResponse", "ddgs", e)  # type: ignore[misc]

__all__ = (
    "DuckDuckSearchTool",
    "SearchResponse",
    "SearchResult",
    "TavilySearchTool",
)
