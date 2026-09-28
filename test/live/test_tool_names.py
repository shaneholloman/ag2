# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from ag2.exceptions import ToolConflictError
from ag2.live import LiveAgent


@pytest.mark.asyncio
async def test_session_rejects_two_tools_sharing_a_name() -> None:
    def deploy() -> str:
        return "deployed"

    config = MagicMock()
    config.session.return_value = nullcontext()
    agent = LiveAgent("live", config=config, tools=[deploy])

    with pytest.raises(ToolConflictError, match="`deploy`"):
        async with agent.run(tools=[deploy]):
            pass
