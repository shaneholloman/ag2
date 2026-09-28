# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context, tool
from ag2.tools import AnthropicBashTool
from ag2.tools.precedence import resolve_tools


@pytest.mark.asyncio
@pytest.mark.parametrize("builtin_first", [True, False], ids=["builtin-first", "function-first"])
async def test_function_and_builtin_sharing_a_name_resolve_by_order(builtin_first: bool, context: Context) -> None:
    @tool
    def bash(command: str) -> str:
        return command

    builtin = AnthropicBashTool()
    tools = [builtin, bash] if builtin_first else [bash, builtin]

    resolved = await resolve_tools(tools, context)

    assert resolved.tools == [tools[-1]]
