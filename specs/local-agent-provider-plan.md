# Local Agent Provider Plan

## Goal

Enable an opt-in local agent lane that can run against an OpenAI-compatible endpoint without changing the default Warp Cloud agent behavior.

## Constraints

- Keep cloud flow untouched when local mode is not selected.
- Keep changes behind `FeatureFlag::LocalAgentProvider`.
- No `skip_login` behavior changes.
- No cloud sync assumptions for local-only runs.

## Architecture

Warp UI prompt  
-> `RequestParams` detects local mode  
-> local stream adapter (`app/src/ai/agent/local`)  
-> OpenAI-compatible endpoint (`/chat/completions`, stream)  
-> convert response into Warp `ResponseEvent`s  
-> existing history/controller pipeline renders output

## Rollout Phases

1. Scaffold (done in this pass)
   - Feature flag + settings group.
   - Request plumbing (`RequestParams.local_agent`).
   - Branch in `generate_multi_agent_output`.
2. MVP stream (done in this pass)
   - Local SSE adapter.
   - Emit `StreamInit` -> one `AddMessagesToTask` -> `StreamFinished`.
3. Tool calling (next)
   - Parse streamed tool calls.
   - Emit `ClientActions` for supported tool set.
   - Reuse existing executor pipeline.
4. UX and guardrails (next)
   - Settings UI controls for provider mode and endpoint/model.
   - Scoped login gate relaxation for local mode only.
5. Hardening (next)
   - Tests for event translation, cancellation, and error mapping.
   - Regression checks for cloud path with flag off/on.

## Testing Checklist

- `cargo check -p app`
- Local mode + endpoint reachable: response appears in conversation.
- Cancel mid-stream stops output.
- Flag off: cloud behavior remains unchanged.
