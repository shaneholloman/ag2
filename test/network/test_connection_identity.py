# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Wire control-plane ops are bound to the calling connection's identities.

A connection acts only as the agents it registered or re-attached to
with an authenticated ``HelloFrame``. Each test drives a real
``serve_ws`` loopback with ``ApiKeyAuth``: mallory holds only her own
key and tries to act as, or read the private state of, bob.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_REJECT,
    EV_TEXT,
    AccessDeniedError,
    ApiKeyAuth,
    AuthBlock,
    AuthRegistry,
    BaseHubListener,
    ChannelMetadata,
    ChannelState,
    Envelope,
    ErrorFrame,
    HelloFrame,
    Hub,
    HubClient,
    NoAuth,
    Passport,
    ProtocolError,
    ReceiptFrame,
    Resume,
    WsLink,
    WsLinkClient,
    serve_ws,
)
from ag2.task import TaskMetadata, TaskSpec, TaskState

from ._helpers import ScriptedConfig

_NAMES = ("alice", "bob", "carol", "mallory")


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


def _passport(name: str) -> Passport:
    return Passport(name=name, auth=AuthBlock(scheme="api_key", claim={"token": f"k-{name}"}))


class _InvitePosted(BaseHubListener):
    """Signals once the hub has posted a channel invite."""

    def __init__(self) -> None:
        self.posted = asyncio.Event()

    async def on_envelope_posted(self, envelope: Envelope, metadata: ChannelMetadata) -> None:
        if envelope.event_type == EV_CHANNEL_INVITE:
            self.posted.set()


@asynccontextmanager
async def _serve(*, allow_no_auth: bool = False) -> AsyncGenerator[tuple[Hub, str]]:
    api_key = ApiKeyAuth(keys={name: f"k-{name}" for name in _NAMES})
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        auth=AuthRegistry([NoAuth(), api_key] if allow_no_auth else [api_key]),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    try:
        async with serve_ws(hub, "127.0.0.1", 0) as server:
            yield hub, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        await hub.close()


class TestActingAsAnotherAgent:
    @pytest.mark.asyncio
    async def test_connection_acts_as_each_agent_it_registered_and_no_other(self) -> None:
        async with _serve() as (hub, url):
            shared_hc, bob_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await shared_hc.register(_agent("alice"), _passport("alice"), Resume())
                carol = await shared_hc.register(_agent("carol"), _passport("carol"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume(), skill_md="bob's skill")

                await shared_hc.set_skill(alice.agent_id, "alice's skill")
                await shared_hc.set_skill(carol.agent_id, "carol's skill")
                with pytest.raises(AccessDeniedError):
                    await shared_hc.set_skill(bob.agent_id, "overwritten")

                assert await hub.get_skill(carol.agent_id) == "carol's skill"
                assert await hub.get_skill(bob.agent_id) == "bob's skill"
            finally:
                await shared_hc.close()
                await bob_hc.close()

    @pytest.mark.asyncio
    async def test_post_envelope_as_another_agent_is_denied(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=channel.channel_id,
                            sender_id=bob.agent_id,
                            audience=None,
                            event_type=EV_TEXT,
                            event_data={"text": "spoofed"},
                        )
                    )

                wal = await hub.read_wal(channel.channel_id)
                assert not any(e.event_type == EV_TEXT for e in wal)
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_update_of_another_agents_task_is_denied(self) -> None:
        async with _serve() as (hub, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await bob_hc.observe_task(
                    TaskMetadata(
                        task_id="t-bob", owner_id=bob.agent_id, spec=TaskSpec(title="t"), state=TaskState.RUNNING
                    )
                )

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.update_task("t-bob", state=TaskState.FAILED, error="sabotaged")

                assert (await hub.get_task("t-bob")).state == TaskState.RUNNING
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_receipt_for_another_agent_does_not_advance_its_cursor(self) -> None:
        async with _serve() as (hub, url):
            mallory_hc = HubClient(WsLink(url))
            try:
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                # bob holds no connection, so nothing acks on his behalf.
                bob = await hub.register_identity(_passport("bob"), Resume())
                assert bob.agent_id is not None
                link = mallory_hc._client_link
                assert link is not None

                await link.send_frame(
                    ReceiptFrame(envelope_id="z" * 32, status="ack", recipient_id=bob.agent_id, channel_id="c-1")
                )
                # Frames on one connection are handled in order: once this
                # RPC answers, the receipt above has been processed.
                await mallory_hc.get_agent("mallory")

                assert hub.inbox_cursor(bob.agent_id, "c-1") == ""
            finally:
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_checkpoint_of_an_unobserved_task_belongs_to_its_first_writer(self) -> None:
        async with _serve() as (_, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await bob_hc.checkpoint_task("t-bob", {"step": 1})

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.read_task_checkpoint("t-bob")
                with pytest.raises(AccessDeniedError):
                    await mallory_hc.checkpoint_task("t-bob", {"step": 99})

                assert await bob_hc.read_task_checkpoint("t-bob") == {"step": 1}
            finally:
                await bob_hc.close()
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_observing_a_task_id_checkpointed_by_another_agent_is_denied(self) -> None:
        async with _serve() as (_, url):
            bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                await bob_hc.checkpoint_task("t-bob", {"step": 1})

                with pytest.raises(AccessDeniedError):
                    await mallory_hc.observe_task(
                        TaskMetadata(
                            task_id="t-bob",
                            owner_id=mallory.agent_id,
                            spec=TaskSpec(title="t"),
                            state=TaskState.RUNNING,
                        )
                    )

                assert await bob_hc.read_task_checkpoint("t-bob") == {"step": 1}
            finally:
                await bob_hc.close()
                await mallory_hc.close()


class TestChannelScope:
    @pytest.mark.asyncio
    async def test_only_participants_read_the_channel_wal(self) -> None:
        async with _serve() as (_, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                assert await bob_hc.read_wal(channel.channel_id)
                with pytest.raises(AccessDeniedError):
                    await mallory_hc.read_wal(channel.channel_id)
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_list_channels_without_agent_lists_only_the_connections_channels(self) -> None:
        async with _serve() as (_, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                assert [m.channel_id for m in await bob_hc.list_channels()] == [channel.channel_id]
                assert await mallory_hc.list_channels() == []
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_non_participant_content_is_refused_whatever_the_adapter_accepts(self) -> None:
        # The conversation adapter accepts non-text events from anyone.
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=channel.channel_id,
                            sender_id=mallory.agent_id,
                            audience=None,
                            event_type="app.note",
                            event_data={"note": "injected"},
                        )
                    )

                assert not any(e.event_type == "app.note" for e in await hub.read_wal(channel.channel_id))
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_non_participant_protocol_event_is_refused(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, bob_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                channel = await alice.open(type="conversation", target="bob")

                with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=channel.channel_id,
                            sender_id=mallory.agent_id,
                            audience=None,
                            event_type=EV_CHANNEL_CLOSED,
                            event_data={"text": "injected"},
                        )
                    )

                wal = await hub.read_wal(channel.channel_id)
                assert not any(e.sender_id == mallory.agent_id for e in wal)
            finally:
                await alice_hc.close()
                await bob_hc.close()
                await mallory_hc.close()

    @pytest.mark.asyncio
    async def test_non_participant_invite_reject_leaves_the_channel_pending(self) -> None:
        async with _serve() as (hub, url):
            alice_hc, mallory_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            # bob holds no connection, so the invite waits for his ack.
            await hub.register_identity(_passport("bob"), Resume())
            opening: asyncio.Task[object] | None = None
            try:
                alice = await alice_hc.register(_agent("alice"), _passport("alice"), Resume())
                mallory = await mallory_hc.register(_agent("mallory"), _passport("mallory"), Resume())
                invited = _InvitePosted()
                hub.register_listener(invited)
                opening = asyncio.create_task(alice.open(type="conversation", target="bob"))
                await asyncio.wait_for(invited.posted.wait(), 2.0)
                (pending,) = await hub.list_channels()

                with pytest.raises(ProtocolError, match="only accepts sends from participants"):
                    await mallory_hc.post_envelope(
                        Envelope(
                            channel_id=pending.channel_id,
                            sender_id=mallory.agent_id,
                            audience=None,
                            event_type=EV_CHANNEL_INVITE_REJECT,
                            event_data={"channel_id": pending.channel_id},
                        )
                    )

                assert (await hub.get_channel(pending.channel_id)).state == ChannelState.PENDING
            finally:
                if opening is not None:
                    opening.cancel()
                    await asyncio.gather(opening, return_exceptions=True)
                await alice_hc.close()
                await mallory_hc.close()


class TestReattach:
    @pytest.mark.asyncio
    async def test_reattach_moves_the_identity_to_the_new_connection(self) -> None:
        async with _serve() as (hub, url):
            old_hc, new_hc = HubClient(WsLink(url)), HubClient(WsLink(url))
            try:
                bob = await old_hc.register(_agent("bob"), _passport("bob"), Resume())
                await new_hc.attach(_agent("bob"), name="bob", passport=_passport("bob"))

                await new_hc.set_skill(bob.agent_id, "from the new connection")
                with pytest.raises(AccessDeniedError):
                    await old_hc.set_skill(bob.agent_id, "from the old connection")

                assert await hub.get_skill(bob.agent_id) == "from the new connection"
            finally:
                await old_hc.close()
                await new_hc.close()

    @pytest.mark.asyncio
    async def test_hello_with_a_scheme_other_than_the_registered_one_is_refused(self) -> None:
        async with _serve(allow_no_auth=True) as (_, url):
            bob_hc = HubClient(WsLink(url))
            mallory_link = WsLinkClient(url)
            try:
                await bob_hc.register(_agent("bob"), _passport("bob"), Resume())
                await mallory_link.open()

                await mallory_link.send_frame(HelloFrame(name="bob", auth_scheme="none", auth_claim={}))
                reply = await asyncio.wait_for(anext(aiter(mallory_link.frames())), 2.0)

                assert isinstance(reply, ErrorFrame)
                assert reply.code == "auth_failed"
            finally:
                await bob_hc.close()
                await mallory_link.close()
