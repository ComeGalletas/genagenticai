# Rework Journal: chatbot + judge loop

Goal: higher quality answers with lower latency for the core LangGraph loop
(`detect_language -> chatbot -> tools -> chatbot -> judge -> END`).

Baseline from `backend/logs/app.log` (2026-08-31):

| Segment | Observed |
|---|---|
| Chatbot LLM call | 4 to 15 s each, up to 3 calls when stages escalate |
| Judge (full rewrite, same model) | 32 to 47 s |
| Whole turn | 57 to 70 s |

Root causes identified during review:

- Judge sees only the answer text, never the question or retrieved documents, so it cannot check grounding.
- Judge prompt asks it to call `retrieve_information` but the node binds no tools and routes only to END.
- Judge rewrites every answer, including good ones, on the large thinking model.
- `retrieval` state uses `operator.add` and is re-injected as a system message on every later turn, so prompts grow forever.
- LLM picks the retrieval stage, so a stage-2 question costs up to three LLM round trips.
- DuckDuckGo page fetches run sequentially with a 15 s timeout each.
- Both prompts spend tokens on producing valid HTML.

Target loop:

```
START -> prepare_turn -> chatbot --tool calls--> tools --> chatbot
                            |
                       final answer
                            v
                      should_judge?  -- no --> finalize -> END
                            | yes
                            v
                          judge  -- pass --> finalize -> END
                            | fail (attempt 0)
                            v
                   chatbot (with critique)  -- may call tools again
```

---

## Phase 0: Measurement harness

- [ ] Write a fixed test set of 10 to 15 questions covering: greeting / no-tool, static index hit, RAG (GPU + LangGraph), web search, Baloto results, Baloto suggestion, LinkedIn jobs, non-English query.
- [ ] Add a small script that runs the set through `run_agent` and records per-node timings from the logs plus the final answer, writing a dated results file.
- [ ] Run it once on the current code and store the baseline in this journal.

## Phase 1: Judge as a structured gate (biggest win, least code) — DONE 2026-09-14

- [x] Add a second `ChatOllama` instance for the judge, `reasoning=False`, `keep_alive` set. Model is `JUDGE_MODEL` env, defaulting to `CHAT_MODEL` (see hardware note below). `CHAT_MODEL` env now also drives the chatbot instead of a hardcoded name.
- [x] Define a pydantic verdict schema `JudgeVerdict` in `graph/judge/schema.py`: `passed`, `grounded`, `needs_more_info`, `issues`.
- [x] Enforce the schema with `with_structured_output(method="json_schema")`.
- [x] Rewrite `JUDGE_SYSTEM_PROMPT` to evaluate, not rewrite. Persona, HTML, and tool-call duties removed.
- [x] Feed the judge the last user question, the last 3 retrieval entries (1200 chars per doc), and the candidate answer.
- [x] Store the verdict in the `judge` state key. The judge no longer appends a message; the chatbot's answer is what the user receives.
- [x] `graph/judge/node.py` rewritten to hold the real judge node; the old broken copy is gone. `core/nodes.py` no longer defines a judge.
- [x] Judge failure (model down, bad JSON) falls back to `passed=True` with `error` set, so a broken judge never blocks a reply.

Smoke test (`judge_response` called directly, qwen3:latest warm):

| Case | Verdict | Time |
|---|---|---|
| Grounded answer (32 GB GDDR7) | passed, grounded | 0.5 s (6.4 s cold, includes model load) |
| Fabricated specs (48 GB HBM3, $999) | failed, 2 issues naming both fabrications | 0.7 s |
| "Could not find" while context had the answer | failed, 1 issue | 0.5 s |
| Greeting, no context | passed | 0.3 s |
| "Could not find", context genuinely empty | passed, needs_more_info=true | 0.4 s |

End-to-end turns through the full graph (RTX 5080, 16 GB VRAM):

| Judge model | Judge time inside a turn | Why |
|---|---|---|
| `qwen3:latest` (separate, 10 GB) | 5.5 to 11.3 s | Ollama evicts the 23 GB chat model to load the judge, then reloads the chat model next turn. The judge call itself is under 1 s. |
| `qwen3.6` (same as chatbot, thinking off) | 1.8 to 2.7 s warm | No model switch. Chosen as default. |

Hardware note: `ollama ps` shows qwen3.6 running 42% CPU / 58% GPU because 23 GB does not fit
in 16 GB. That is why chatbot calls take 27 to 37 s. A separate small judge only pays off when
both models fit in VRAM together. The single biggest latency lever available right now is
therefore `CHAT_MODEL=qwen3:14b` (9.3 GB, fits entirely on the GPU), which is the user's call.

Turn times after Phase 1 (qwen3.6 chat, judge on qwen3:latest, before switching the default):
44.7 s and 39.1 s, versus 57 to 70 s baseline. The remaining time is almost all the chat model.

Regression to watch: the chatbot now emits markdown (`**32 GB**`) despite the HTML instruction.
The old rewriting judge used to convert this to HTML; the gate judge does not. The frontend
renders non-HTML replies as plain text, so bold markers show literally. Phase 5 (server-side
markdown to HTML) fixes this properly and should be pulled forward, ahead of Phases 3 and 4.

## Phase 2: Critique loop back to the chatbot

- [ ] Add `judge_attempts: int` to `State`, reset at turn start.
- [ ] Add a conditional edge after `judge`: pass -> `finalize`; fail and attempts < 1 -> `chatbot`; otherwise -> `finalize`.
- [ ] When routing back, append the verdict issues as a message ("Revise your answer: ...") so the chatbot can rewrite or call tools again.
- [ ] Add a deterministic `should_judge` router after `chatbot`: skip the judge for short replies and turns that used no tools.
- [ ] Verify with the test set that greetings never hit the judge and factual answers do.

## Phase 3: Keep the prompt small

- [ ] Add a `prepare_turn` node (replaces or wraps `detect_language`) that resets `retrieval`, `judge`, and `judge_attempts` for the new turn.
- [ ] Change the `retrieval` reducer so it no longer accumulates across turns (or keep only the last N entries).
- [ ] Apply `trim_messages` with a token budget before invoking the chatbot LLM.
- [ ] Set `num_ctx` on the chatbot client to match the trimmed budget.

## Phase 4: Faster retrieval

- [ ] Make `retrieve_information` run the whole pipeline in one tool call using the engine's existing `run_pipeline` and early-exit heuristic, or add a deterministic stage pre-router modeled on the Baloto keyword scorer.
- [ ] Update the system prompt stage descriptions to match (stage 1 also contains LangGraph notes today).
- [ ] Parallelize `fetch_webpage` calls in `search/ddgo.py` with a thread pool or async httpx; lower per-page timeout to about 5 s.
- [ ] Add a small TTL cache keyed on (query, stage) for retrieval results.
- [ ] Fix `read_webpage`: return early when `fetch_url` returns `None` instead of passing it to the extractor.

## Phase 5: Deterministic formatting

- [ ] Change the chatbot prompt to request markdown instead of HTML; keep the persona there and only there.
- [ ] Add a `finalize` node that converts markdown to HTML server-side and sanitizes it.
- [ ] Confirm the frontend `MessageContent` path still renders correctly (DOMPurify already in place).
- [ ] Remove HTML formatting rules from the judge prompt entirely.

## Phase 6: Streaming

- [ ] Add a streaming endpoint using `graph.stream(stream_mode="messages")` behind a FastAPI `StreamingResponse` (SSE).
- [ ] Update `App.jsx` to append chunks as they arrive and fall back to the existing non-streaming endpoint.
- [ ] Decide how the judge interacts with streaming: stream the chatbot's final answer and only replace it if the judge fails, or hold until the verdict passes.

## Housekeeping found during review (optional, low risk)

- [ ] Stop writing `graph.png` at import time in `graph/core/graph.py`; move it behind a flag or a CLI command.
- [ ] `config.py` still requires `GOOGLE_API_KEY` and defaults to a Gemini model name; nothing uses either since the move to Ollama and DuckDuckGo.
- [ ] `tests/test_in_memory_database.py` imports a class that does not exist; replace or remove.
- [ ] `static_google_search` tool in `tools.py` is not bound to the LLM; remove or bind.
- [ ] Empty `cv_analysis` stubs: keep as placeholder or remove until the feature is started.

---

## Expected outcome

| Segment | Today | Target |
|---|---|---|
| Retrieval | 1 to 3 LLM round trips, sequential fetches | 1 round trip, parallel fetches |
| Judge | 30 to 47 s full rewrite | 1 to 3 s verdict, retry only on failure |
| Formatting | Two LLM passes emit HTML | Deterministic markdown to HTML |
| Whole turn | 57 to 70 s | Roughly 15 to 25 s, first token in a few seconds with streaming |

## Log

- 2026-09-14: Journal created after codebase review. No code changed yet.
- 2026-09-14: Phase 1 done. Judge is a structured gate on the chat model with thinking off. Files: `app/config.py`, `app/graph/judge/{node,schema,state,system_prompt}.py`, `app/graph/core/{nodes,graph}.py`. Next recommended: Phase 5 (markdown to HTML) then Phase 2 (critique loop).
