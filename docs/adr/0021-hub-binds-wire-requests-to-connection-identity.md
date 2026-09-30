---
status: accepted
date: 2026-09-30
---

# 0021. The hub authorizes wire requests against the connection's bound identities

## Context

A network client proves who it is once per connection: the `register` op
validates the passport's auth claim, and a `HelloFrame` does the same when an
existing identity re-attaches. Either binds the connection's endpoint to that
`agent_id` (`Hub.bind_endpoint`), and one connection may hold several agents.

The wire control plane (`Hub._dispatch_request_op`) then maps each request to a
hub method. Those methods take the identity they act on — `agent_id`,
`creator_id`, an envelope's `sender_id`, a task's owner — as a plain argument,
because the same methods back in-process callers, the hub's own sweepers, and
expectation handlers, none of which have a connection. The endpoint binding was
used only to route notifies, so a request could name any agent.

## Decision

**Every wire request is authorized in the dispatcher, against the set of agents
bound to the calling endpoint, before the hub method runs.** Hub methods keep
their signatures and stay unchecked.

`Hub._authorize_request` classifies each op:

- **Unscoped** — `register` and discovery reads (agents, resumes, skills,
  rules, tasks). A fresh connection runs these before its `HelloFrame`.
- **Agent-scoped** — the named agent must be bound: identity mutation,
  `unregister`, `create_channel`, `can_send`, `pending_turns_for`,
  `report_turn_failure`, `record_observation`, the `sender_id` of
  `post_envelope`, the owner in `observe_task`.
- **Channel-scoped** — a bound agent must be a participant: `get_channel`,
  `close_channel`, `read_wal`, `find_envelope_by_causation`. `list_channels`
  without an `agent_id` lists only those channels.
- **Task-scoped** — when the hub has observed the task, its owner must be bound.
  A checkpoint of a task the hub has not observed belongs to the agents bound to
  the connection that first wrote it (recorded next to the checkpoint); later
  reads and writes need one of them, and so does `observe_task` for that id,
  since observing makes the caller the task's owner.

An op in no class is rejected, so a new op is unreachable until it is
classified. A `ReceiptFrame` for an agent not bound to the connection is dropped.
Denials raise `AccessDeniedError`, returned as an `access_denied` response; the
connection stays open.

The binding is only as strong as admission, so a `HelloFrame` re-attaching an
existing identity is validated with the scheme that identity registered with; a
Hello naming another scheme — e.g. `none` in a registry that also holds `NoAuth`
— is refused with `auth_failed`.

A `kind="remote_agent"` passport is different: its `auth.scheme` is a routing
label for a `RemoteAgentProxy`, not a credential, so registration never
validates it. The federation operator normally registers such identities on
the hub directly (`Hub.register_identity`) and its proxy posts for them
in-process; they have no connection of their own. Over the wire, therefore:

- registering a `remote_agent` needs a connection that already holds a
  non-remote agent of this hub. Those agents become its **owners**, persisted
  at `agents/{id}/owners.json` and reloaded on `hydrate`;
- a Hello for a `remote_agent` is accepted only on a connection holding one of
  its owners, and never on the strength of a scheme. One registered in-process
  has no owners and cannot be re-attached over the wire at all.

Without this, a connection with no credential could register a `remote_agent`
under an unused name, capturing traffic meant for it, or re-attach as one with
`auth_scheme="none"` wherever the registry holds `NoAuth`.

Every task id a wire request names (`task_id`, `metadata.task_id`,
`envelope.task_id`) must be one path segment — `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`,
which covers `uuid4().hex` and ids like `task-1` — because task ids become
store paths. Anything else is a `ProtocolError`.

Independently of the connection, `post_envelope` accepts events only from the
channel's participants, in the hub rather than per adapter, so an adapter that
forgets the check cannot let outsiders inject content. This covers protocol
events too — an outsider's `ag2.channel.invite.reject` would otherwise fail
another agent's handshake. Hub-generated protocol envelopes carry the creator,
and invitees are participants from creation. The one exception is
`ag2.task.cancel_request` shaped as the `tasks` tool sends it, which any peer
may post: its `task_id` names a live task the hub has observed in that same,
active channel, its audience is exactly `[task owner]`, and its `event_data` is
exactly `{"task_id", "reason"}` with a string `reason`. For that to mean anything, a task's channel
must be trustworthy, so `observe_task` over the wire requires the owner to be a
participant of the task's `channel_id`.

## Consequences

- The trust boundary is the wire. An in-process `HubClient` with a hub reference,
  and code holding the `Hub`, act as any agent — they already run inside the
  hub's process.
- A channel participant reads the whole WAL, including envelopes whose
  `audience` excludes it: the client re-folds adapter state from the full WAL,
  so filtering by audience would diverge it from the hub's fold.
- Re-attaching an agent from a new connection moves its authority there; the old
  connection's late requests and receipts for it are rejected or dropped.
- Task records are hub-wide reads: any admitted connection sees every task's
  spec, state, progress and result through `get_task` / `list_tasks`. This is
  intended — delegators poll and wait on tasks other agents own, and the tasks
  tool's `scope="all"` lists across owners — so agents must not put anything in
  a task spec or result that other agents on the hub may not read.
- A checkpoint written only in-process (or before writers were recorded) has no
  recorded writer and stays readable by any connection until its first write
  over the wire.
