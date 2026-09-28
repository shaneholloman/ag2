# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from ag2 import tool
from ag2.events import ToolCallEvent
from ag2.live import LiveAgent


@pytest.mark.asyncio
async def test_session_dispatches_a_shared_name_to_the_later_tool() -> None:
    runs: list[str] = []

    @tool(name="deploy")
    def agent_deploy() -> str:
        runs.append("agent")
        return "deployed"

    @tool(name="deploy")
    def run_deploy() -> str:
        runs.append("run")
        return "deployed"

    config = MagicMock()
    config.session.return_value = nullcontext()
    agent = LiveAgent("live", config=config, tools=[agent_deploy])

    async with agent.run(tools=[run_deploy]) as context:
        await context.send(ToolCallEvent(name="deploy", arguments="{}"))

    assert runs == ["run"]
    [schema] = config.session.call_args.kwargs["tools"]
    assert schema.function.name == "deploy"
