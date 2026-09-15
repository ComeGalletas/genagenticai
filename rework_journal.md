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

## Phase 0: Measurement harness — DONE 2026-09-14 (after Phases 1 to 5)

- [x] `backend/benchmarks/questions.json`: 15 questions in 8 categories (no-tool, tool-time, static, rag, web, baloto, jobs, language), with shared threads for follow-ups.
- [x] `backend/benchmarks/run_benchmark.py`: streams each question through the graph, records per-node time (from update-event deltas), path, tools used, retrieval stage and doc count, every judge verdict in order, attempts, language, and the reply. Writes `benchmarks/results/<timestamp>_<chat-model>.json` and a `.md` table. `--only` / `--skip` by category or id, `--baseline <json>` adds a delta column, `--quiet` hides node logs. Clears the retrieval cache first so every run starts cold.

```bash
cd backend
CHAT_MODEL=granite4.1:8b JUDGE_MODEL=granite4.1:8b python -m benchmarks.run_benchmark --quiet
python -m benchmarks.run_benchmark --only rag web --baseline benchmarks/results/<file>.json
```

- [x] Baseline recorded: `benchmarks/results/2026-09-14_230106_granite4.1-8b.{json,md}` (granite 8b chat + judge, commit 03b430e + uncommitted rework).

The first full run (`2026-09-14_225407`) exposed three judge defects that hand testing had
missed. All three were fixed before the baseline was recorded:

1. **Judge could not see tool outputs.** It only read the `retrieval` state, so answers built from
   `get_current_time`, `suggest_baloto_numbers` or `retrieve_job_postings` looked ungrounded: false
   failures, wasted revise loops, and for the Baloto suggestion a rewrite that fell back to the
   previous turn's answer. Fix: `_format_context` now appends this turn's ToolMessages
   (`JUDGE_MAX_TOOL_CHARS` 8000 each; the job list alone is 8.6k chars, the lottery JSON 1.5k).
2. **Judge assumed its training-cutoff date** and flagged real September 2026 posting dates as
   "future". Fix: the judge prompt now starts with today's date.
3. **Style nitpicks counted as failures** ("extraneous playful phrase nya~"). Fix: prompt says
   tone, verbosity and formatting are never a failure, and a premise the user stated is never
   an invention.

Baseline vs first run (granite 8b, warm):

| id | category | stage | first run | baseline | judge |
|---|---|---|---|---|---|
| greeting | no-tool | - | 0.5 | 0.3 | skipped |
| smalltalk | no-tool | - | 0.3 | 0.3 | skipped |
| time | tool-time | - | 8.5 | 0.9 | pass |
| static-game | static | 0 | 2.2 | 1.9 | pass (now "Team Cherry"; first run invented "Team Ninji" and the judge passed it) |
| static-framework | static | 0 | 6.1 | 6.1 | pass |
| rag-gpu | rag | 1 | 1.3 | 1.5 | pass |
| rag-gpu-followup | rag | 1 | 1.5 | 1.5 | pass |
| rag-langgraph | rag | 2 | 13.2 | 10.8 | pass (RAG missed MemorySaver at threshold 0.44; went to web) |
| web-current | web | 2 | 14.5 | 12.2 | pass (answer quality varies with the page DuckDuckGo returns) |
| web-general | web | 2 | 6.9 | 6.1 | pass |
| unknown | web | 2 | 9.1 | 18.2 | FAIL twice (judge too strict on an honest "not found"; answer returned anyway) |
| baloto-results | baloto | 1 | 2.7 | 2.5 | pass |
| baloto-suggest | baloto | 1 | 12.7 | 5.0 | pass (first run never surfaced the suggestion) |
| jobs | jobs | 2 | 19.0 | 18.7 | FAIL twice (chatbot asserts locations and dates the listings lack; judge is right) |
| spanish | language | 1 | 2.0 | 2.1 | pass, reply in Spanish |
| **sum / mean** | | | 100.6 / 6.7 | 88.0 / 5.9 | |

Follow-ups surfaced by the harness (not done):
- RAG threshold: "MemorySaver" is in `knowledge.txt` but scored below `RAG_SCORE_THRESHOLD` 0.44, so the question went to the web. Worth lowering to ~0.40 and re-checking with `--only rag`.
- Jobs prompt: tell the chatbot to say "not listed" for missing location or date instead of assuming Bogotá.
- Judge strictness on honest "not found" answers when the web returns tangential pages (`unknown` row) costs one revise loop (~9 s).
- Web answers can be confidently wrong when the fetched page is stale (`web-current` returned "Claude 3" in one run and "Fable 5.1" in another); the judge cannot detect staleness. A date-aware source ranking would help.

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

## Phase 2: Critique loop back to the chatbot — DONE 2026-09-14

- [x] `judge_attempts: int` in `State`; `judge`, `judge_attempts` and `final_response` are reset in `detect_language` (first node of every turn; Phase 3 moves this into `prepare_turn`).
- [x] `route_judge` after `judge`: pass -> `finalize`; fail with attempts < `JUDGE_MAX_RETRIES` (default 1) -> `revise`; otherwise `finalize` with the last answer.
- [x] `revise` node (`graph/judge/node.py`) appends a critique built by `graph/judge/critique.py` and increments the counter. The critique is a `HumanMessage` tagged `name="judge_critique"` so chat templates accept it, while `is_critique()` lets the judge, routers and turn detection ignore it. Its wording depends on the verdict: search again at a higher stage when `needs_more_info`, fix using existing context when not grounded, otherwise rewrite.
- [x] `should_judge` in `route_chatbot`: judge whenever tools were used this turn or the reply is at least `JUDGE_SKIP_MAX_CHARS` (300); short no-tool replies go straight to `finalize`. Turn boundaries are the last real user message, so tools from earlier turns do not count.
- [x] Judge's question lookup skips critiques, so it always compares the answer to the real user question.
- [x] 12 routing/critique unit tests in `tests/test_routing.py` (23 total with formatting). Graph edges verified: chatbot -> {tools, judge, finalize}, judge -> {finalize, revise}, revise -> chatbot.
- [x] Integration run with the first verdict forced to fail: greeting skipped the judge (2.4 s total); the factual turn went chatbot -> tools -> chatbot -> judge(fail) -> revise -> chatbot -> judge(pass) -> finalize. The rewrite added exactly the two facts the issues named (512-bit bus, 21,760 CUDA cores). Turn total 39 s including the extra 13 s chatbot call and 4 s judge for the retry; a passing turn stays at the Phase 5 numbers.

Graph after Phase 2:

```
START -> detect_language -> chatbot --tool calls--> tools --> chatbot
                               |
                          final answer
                               v
                 should_judge? (tools used or >= 300 chars)
                    no |                 | yes
                       v                 v
                   finalize            judge --pass--> finalize -> END
                                         | fail, attempts < 1
                                         v
                                       revise -> chatbot (may call tools again)
```

## Phase 3: Keep the prompt small — DONE 2026-09-14

- [x] `prepare_turn` node replaces `detect_language` as the first node: detects the language and resets `judge`, `judge_attempts`, `final_response`. `detect_language` is now a plain function returning the language name.
- [x] `retrieval` reducer is `bounded_retrievals` (`core/state.py`): appends, keeps only the newest `RETRIEVAL_MAX_ENTRIES` (3), and resets when a node writes `None`. Deviation from the plan: a bounded window instead of a per-turn reset, because follow-ups such as "and its TDP?" should still be answerable from the previous search and the old tool messages in history refer to that context. Growth is capped either way.
- [x] `trim_history` in `core/nodes.py`: `trim_messages` with `count_tokens_approximately`, `CHAT_HISTORY_MAX_TOKENS` (6000), strategy last, starts on a human message so tool-call/tool-result pairs stay intact, never drops the current turn. Only the LLM input is trimmed; the checkpoint keeps everything.
- [x] `num_ctx` set on both clients: `CHAT_NUM_CTX` 16384, `JUDGE_NUM_CTX` defaults to the same value when the judge shares the chat model (Ollama reloads the model whenever `num_ctx` changes between requests; the first run with 16384/8192 showed the reload in `ollama ps`). `keep_alive` 10m on the chat client too.
- [x] 9 new tests in `tests/test_turn_state.py`, including the reducer running inside a compiled LangGraph (32 tests total, all passing).

Model pairing for tests from here on: `CHAT_MODEL=granite4.1:8b JUDGE_MODEL=granite4.1:8b`
(5.3 GB, 100% GPU, no swapping). Judge smoke test on granite: 3b gets every pass/fail right in
0.2 s but returns empty `issues`, so the revise loop would have nothing concrete to fix; 8b returns
correct verdicts with specific issues in 0.4 to 1.1 s. Use the 8b for both roles.

Five-turn conversation on the granite pair after Phase 3:

| Turn | Path | Total |
|---|---|---|
| "hi there!" | prepare_turn > chatbot > finalize | 2.9 s |
| VRAM question | chatbot > tools > chatbot > judge > finalize | 11.2 s |
| "And what is its TDP?" (follow-up) | same | 9.1 to 13.2 s |
| "Who developed Elden Ring?" | ... judge FAIL > revise > chatbot > judge pass | 23.2 s (natural revise) |
| "Which studio made Hollow Knight?" | chatbot > tools > chatbot > judge > finalize | 9.4 s |

Retrieval entries stayed at 3 after the fourth search. History of 20 messages was under the
budget so nothing was trimmed. LLM calls are 0.4 to 3.7 s and the judge about 1 s on granite 8b.

Millisecond trace of two later turns (6.1 s each, granite 8b, everything warm):

| Step | Time |
|---|---|
| prepare_turn | < 0.2 s |
| chatbot (tool call) | 0.4 to 0.6 s |
| tools: `retrieve_information` at **stage 2** (DuckDuckGo + page fetches) | 3.4 to 3.5 s |
| chatbot (final answer) | 1.1 to 1.5 s |
| judge | 0.6 to 0.8 s |
| finalize | < 0.01 s |

Finding for Phase 4: granite jumps straight to stage 2 for every question, including "Which studio
made Hollow Knight?" (a stage-0 static hit) and "TDP of the RTX 5090" (a stage-1 RAG hit, 0.08 s
measured in isolation). The LLM-chosen stage number is the bottleneck, not the retrieval code.
Deterministic stage routing or running the pipeline with early exit inside one tool call would cut
most tool turns to about 3 s.

## Phase 4: Faster retrieval — DONE 2026-09-14

- [x] `retrieve_information` runs `RetrievalEngine.run_pipeline` in one call: static index -> local RAG -> web, stopping at the first stage with at least `RETRIEVAL_STOP_MIN_RESULTS` (1) results. Returns `(results, last_stage, next_stage)`; the tool message tells the LLM the stage used or, when nothing was found anywhere, says so explicitly.
- [x] Deterministic stage control (`resolve_start_stage`): a requested higher stage is honored only if every cheaper stage was already tried for the same normalised query (per the cache). Needed because granite passed `stage=2` on every single call regardless of the docstring, which skipped the cheap stages entirely. Escalation still works: a retry with the same query (e.g. after the judge's "search again at a higher stage" critique) goes to the web even if stage 1 returned a weak match.
- [x] Static index matcher rewritten (`search/static.py`): a match needs two distinct meaningful keyword tokens or an adjacent keyword pair as a phrase ("hollow knight", "elden ring"); stopwords and generic tokens ("the", "game", "movie", "python", ...) never count. Before, any single token including "the" matched, which would have made early exit return The Matrix for "What is the TDP".
- [x] System prompt stage descriptions updated (stage 1 now says NVIDIA RTX 50 GPUs and LangGraph); escalation instruction says stage=2 with the same query.
- [x] `search/ddgo.py` fetches result pages concurrently (`ThreadPoolExecutor`, `WEB_FETCH_WORKERS` 5) with `WEB_FETCH_TIMEOUT` 5 s per page.
- [x] TTL cache per (normalised query, stage) in the engine: `RETRIEVAL_CACHE_TTL` 300 s, `RETRIEVAL_CACHE_SIZE` 128, injectable clock for tests, results returned as copies.
- [x] `read_webpage` returns early with a clear message when the fetch fails.
- [x] 21 new tests (`tests/test_static_search.py`, `tests/test_retrieval_engine.py`); 53 total, all passing. Node mode (`USE_RETRIEVAL_PIPELINE_TOOL=False`) still uses `run_stage` and is unaffected.

Six-turn conversation on granite 8b, everything warm:

| Turn | Stage used | Total before Phase 4 | Total after |
|---|---|---|---|
| "hi there!" | none | 2.9 s | 0.5 s |
| VRAM question | 1 (RAG) | 11.2 s | 2.5 s |
| "And what is its TDP?" | 1 (RAG) | 9.1 s | 1.8 s |
| "Who developed Elden Ring?" | 0 (static) | 23.2 s | 1.6 s |
| "Which studio made Hollow Knight?" | 0 (static) | 9.4 s | 1.8 s |
| "Latest version of the Claude AI model?" | 2 (web, 7.7 s this run) | n/a | 10.2 s |

Every call still arrived with `stage=2` requested and was redirected to stage 0 by the engine.

### Addendum 2026-09-14: decisive vs tentative static hits

User-reported failure: "what is expedition 33" answered only with the video game. The static
matcher treated the phrase "expedition 33" as a decisive hit on the Clair Obscur entry, early exit
stopped at stage 0, and the ISS mission (which the web stage does return, ranked first) never
appeared. A single curated hit means "the query mentions this item", not "this is the only meaning".

Fix:
- Every static entry now has a `name`. A hit is **decisive** (confidence 0.95, `metadata.decisive`
  true) only when the full name appears in the query ("hollow knight", "clair obscur: expedition 33");
  a keyword or partial-phrase hit is **tentative** (0.6, decisive false).
- `RetrievalEngine._enough_information` counts only decisive results, so tentative hits stay as
  context while the pipeline continues to RAG and the web, and everything is merged.
- Chatbot prompt: when sources describe different things with the same name, present each.
- The Clair Obscur entries now have real names, developer (Sandfall Interactive), publisher and year.
- `finalize` falls back to the judge's `original_response` when a revision comes back empty. Found
  while testing this on qwen3.6, whose revision call ran 383 s and returned nothing, which used to
  produce an empty reply.
- Benchmark row `ambiguous` ("what is expedition 33") added; 70 unit tests pass.

Result on granite 8b: stage 0 tentative -> stage 1 (0 docs) -> stage 2 (5 docs incl. "Expedition 33
(ISS)") -> one answer describing both the 2012 ISS mission and the 2025 game, judge pass, 11 to 18 s.
Static fast path unchanged: Hollow Knight 2.0 s, LangGraph 3.2 s.

Note: `app/config.py` on disk defaults `CHAT_MODEL` to `qwen3.6` and `backend/.env` sets no model,
so any run without env vars uses qwen3.6 (23 GB, CPU spill, 20 to 380 s per call on this GPU). Set
`CHAT_MODEL=granite4.1:8b` in `backend/.env` to make the granite pair the default.
The web stage varies 2 to 8 s run to run; that is DuckDuckGo plus page latency, now bounded by the
5 s per-page timeout. Side benefit: the Elden Ring answer is now strictly grounded in the static
snippet (FromSoftware with George R.R. Martin) instead of adding an unsupported director claim.

## Phase 5: Deterministic formatting — DONE 2026-09-14

- [x] Chatbot prompt now asks for Markdown and forbids HTML tags; persona stays there only.
- [x] `finalize` node (`core/nodes.py`) renders the last AI message with `core/formatting.py`: markdown-it-py (commonmark + tables + strikethrough, raw HTML allowed) then a stdlib `HTMLParser` allowlist sanitizer (drops script/style/iframe with content, unwraps unknown tags, strips event handlers and non-http/mailto hrefs, balances tags, escapes text). No new download: markdown-it-py was already installed via rich; it is now an explicit dependency in `pyproject.toml`, `uv.lock` and `requirements.txt`.
- [x] Result stored in new `final_response` state key; `run_agent` returns it. Conversation history keeps the markdown so the LLM never sees HTML in its own prior turns.
- [x] Graph: `judge -> finalize -> END`. Phase 2 routing slots in between judge and finalize.
- [x] Judge prompt: "ignore Markdown and HTML markup".
- [x] Frontend unchanged: `MessageContent` detects the `<p>` tag and renders through DOMPurify; `App.css` already styles `.html-content` p/h1-h6/ul/ol/a/pre/code.
- [x] 11 unit tests in `tests/test_formatting.py` (`python -m unittest tests.test_formatting`).
- [x] `<think>...</think>` leakage is stripped before rendering.

End-to-end after Phase 5 (both models warm, qwen3.6 chat + judge):

| Turn | Total | Reply |
|---|---|---|
| "hi there!" | 10.8 s | `<p>Hello! ... nya~?</p>` |
| VRAM + TDP question (1 RAG call) | 19.8 s | `<p>... <strong>32 GB of GDDR7 VRAM</strong> ... <strong>575 W</strong>.</p>` |

Note on `requirements.txt`: the re-export changed ~1800 lines because the committed file was stale
(it predated trafilatura, ddgs, langdetect and babel). It now matches `uv.lock`, and includes the
dev group (pytest, ruff, mypy) as `uv export` does by default.

## Phase 6: Streaming — DONE 2026-09-14

- [x] `POST /api/chat/stream` (`server/api.py`): Server-Sent Events over a POST, `StreamingResponse` fed by `stream_agent_events` in `core/graph.py`, which wraps `graph.stream(stream_mode=["messages", "updates"])`. `POST /api/chat` is unchanged and remains the fallback.
- [x] Event translation is a pure function `translate_stream_events` (unit-tested, 4 tests): `status` progress hints (Thinking, Searching, Reading the page, Crunching Baloto numbers, Reviewing the answer, Improving the answer), `delta` chatbot tokens (markdown), `reset` when the chatbot turn was a tool call or the judge sent the answer back, `final` with the sanitized HTML, then `done` or `error`. Judge tokens are filtered out by node name. If finalize produced nothing, `final` is computed from state like `run_agent` does.
- [x] Judge interaction decision: stream the chatbot's answer while it is generated, run the judge afterwards, and on failure send `reset` and stream the revision. Best perceived latency; the user sees a draft within a second and the swap to final HTML is one repaint.
- [x] `App.jsx`: reads the SSE body with `fetch` + `ReadableStream` (a POST cannot use `EventSource`), keeps a draft assistant message showing raw markdown tokens with `white-space: pre-wrap` plus an italic pulsing status line, and swaps in the final HTML through the existing DOMPurify path. Any streaming failure before `final` (old backend, buffering proxy, network) falls back to `/api/chat` automatically. The separate "Thinking..." bubble is gone; the draft carries the status.
- [x] Verified over real HTTP (uvicorn + httpx, granite 8b): "Who developed Elden Ring?" produced 101 delta frames spread over 0.8 s, first token 0.9 s after the search, final HTML at 2.1 s. The FastAPI TestClient buffers the whole body, so it cannot show this; use a real server.
- [x] Verified in the browser (Vite dev server via `.claude/launch.json`, config `frontend`): one second after Send the bubble showed streaming markdown (`**FromSoftware**` still raw), at five seconds it showed the rendered HTML with bold, a list and a link. No console errors. `npm run build` passes.
- [x] 63 unit tests pass (`tests/test_stream_events.py` added).

Known limits: the draft is raw markdown (asterisks visible) until `final`; a markdown renderer in the frontend would need a new npm dependency. Status says "Reviewing the answer…" for a moment even when the judge is skipped.

## Link verification in the judge — DONE 2026-09-15

User-reported failure: after "what is expedition 33", asking about sequels or spinoffs pulled the
fictional "Clair Obscur: Writers Revenge" static entry (kept on purpose as a fixture), and the
chatbot presented it as real with a fake link and mixed web citations. The judge passed it because
every claim was supported by *some* context document and it never opened a link.

Design (`app/graph/judge/verify.py`, wired into `judge/node.py`):
- Before the verdict, the judge fetches the pages behind (a) every link the answer cites and
  (b) the source URL of every curated index entry the answer relied on, cited or not. Links whose
  origin is a tool output or an already-fetched web document are trusted and not re-fetched.
  Parallel, `JUDGE_VERIFY_TIMEOUT` 5 s, `JUDGE_MAX_VERIFY_LINKS` 5, `JUDGE_VERIFY_PAGE_CHARS` 2000.
- Page statuses: `ok`, `unreachable` (DNS, timeout, 404, 5xx), `blocked` (401/403/429/503: the page
  exists but refuses bots), `no_text`. Blocked and no_text are inconclusive; unreachable is not.
- The judge prompt gets a CITED PAGES section (role, status, excerpt per page) and rules: a cited
  link whose page is unreachable or silent on the claim goes in the new `unsupported_links`
  verdict field; a curated entry whose page is missing or empty is not evidence.
- Deterministic backstops, because the 8b judge names bad links in `issues` but often leaves
  `unsupported_links` empty: cited links that are unreachable or beyond the fetch cap are
  unsupported regardless of the model; a curated entry whose page is unreachable/no_text is an
  `unverified_source`, and an answer resting on one cannot pass.
- On revision the critique lists the unverified sources and says: drop those claims or state
  they are unverified, and do not search for them again (a repeated hit on the same entry proves
  nothing). The chatbot's retrieval context labels documents from unverified sources
  `[UNVERIFIED: source page missing or empty]`, and the context header no longer says "Verified".
- `finalize` last line of defense: unverified links are stripped (labels kept, empty parentheses
  cleaned) with "N reference(s) could not be verified and were removed"; if the final verdict still
  failed on verification, a note lists what could not be confirmed (judge issues, links hidden).
- `search/ddgo.py`: when a site returns a blocked status to httpx, retry with trafilatura's fetcher.
  Wikipedia returns 403 to httpx with every user agent tried but serves trafilatura; without this
  every Wikipedia-backed curated entry looked unverifiable. Benefits the web stage too.
- Benchmark rows `ambiguous-spinoff` and `fake-entry` (category `verify`). 88 unit tests pass
  (`tests/test_verify.py` added).

Live results on granite 8b:

| Question | Path | Result |
|---|---|---|
| "What is Clair Obscur: Writers Revenge?" (fake entry, decisive hit) | judge FAIL (unverified source) > revise > judge FAIL (invented details) > finalize | Fake link stripped, answer reframed around the verified game, note lists what could not be confirmed. 11 to 13 s. |
| "Does the video game have any sequels or spinoffs?" | web search, real interview coverage | pass, fake entry never retrieved (query did not match its keywords). 14.6 s |
| "what is expedition 33" | tentative static > RAG > web | pass, both meanings. 12.7 s |
| static-game / rag-gpu controls | unchanged paths | pass; verification adds ~0.5 to 1 s of page fetch when a curated entry was used |

Known limit, unchanged: the 8b judge still passes a specific invented name when the fetched page is
silent on it (Hollow Knight answered "Team Ninjalo" in one run with hollowknight.com fetched and
ok). Verification supplies evidence; weighing absence of evidence is judgment, and a stronger judge
model (e.g. qwen3:14b) is the lever for that, at the cost of VRAM alongside the chat model.

## Judge context and turn scoping — DONE 2026-09-15

User-reported failure (from `logs/app.log`): after "what is expedition 33", the spinoff question, and
"Who is Denia in Wuthering Waves?", the follow-up "what are the best teams for her?" was answered
correctly by the chatbot but failed by the judge, then rewritten into a request for clarification,
then shipped with a "could not be verified" note listing relevance complaints. Five causes:

1. The judge saw only the last user message, so "her" was unresolvable.
2. Verification and the judge's context used the whole retrieval window, so the Expedition 33 and
   fake Writers Revenge entries from earlier turns were re-fetched and re-judged on the Denia turn;
   the judge conflated wroters.com's `no_text` status with a Wuthering Waves guide link.
3. The critique relayed the judge's "clarify which her" as an instruction; the chatbot obeyed.
4. The critique said "search again at a higher stage" although the web stage had already run.
5. The finalize note was triggered by the stale wroters.com entry and printed every judge issue
   under a source-verification heading.

Fixes:
- Retrieval entries now carry the `tool_call_id` that produced them (`format_results`), and
  `core/router.py` gained `turn_tool_call_ids`, `turn_retrievals` and `web_searched_this_turn`.
- Judge prompt has a RECENT CONVERSATION section (last 3 user/assistant exchanges, 400 chars each,
  critiques and tool messages excluded) and a rule: resolving a reference the way the conversation
  implies is correct; never fail an answer for not asking the user to clarify.
- Judge context shows this turn's documents in full and earlier turns' documents as titles only,
  labelled background. Verification fetches uncited curated entries only from this turn; earlier
  entries are fetched only if the answer cites them. Web sources and tool URLs from the whole
  window stay trusted, since an answer may legitimately reuse an earlier source.
- Critique: opens with "keep answering the question as you understood it from the conversation";
  when the web stage already ran this turn (or a source is unverified), "search again" is replaced
  by "answer from what was verified and say what is not known".
- Finalize note is built from verification data only: it names the unverified curated entries
  retrieved this turn by title and says claims resting on them may be inaccurate. Judge prose never
  reaches the user. Triggered only when such an entry exists this turn.
- URL extraction drops backticks and markdown emphasis glued to links (a model wrote
  `https://wroters.com/\``, which verified as a 404 on a nonexistent path).
- 93 unit tests pass.

Live rerun of the reported scenario on granite 8b:

| Turn | Verification | Verdict | Time |
|---|---|---|---|
| what is expedition 33 | 1 page (its own curated entry) | pass | 12.5 s |
| sequels or spinoffs? | fake entry retrieved this turn; link stripped; note names "Writers Revenge" | FAIL twice, honest reply | 14.6 s |
| Who is Denia in Wuthering Waves? | nothing to fetch (no curated entry this turn) | pass | 14.1 s |
| what are the best teams for her? | nothing to fetch; stale entries not re-verified | pass, answered for Denia | 20.2 s |
| What is Clair Obscur: Writers Revenge? (fixture) | unchanged behaviour | FAIL twice, stripped, noted | 11.9 s |

## Housekeeping — DONE 2026-09-14

- [x] `graph.png` is no longer written at import; `python -m app.main graph [path]` renders it on demand (`save_graph_png`).
- [x] `config.py` rewritten: `GOOGLE_API_KEY`, the Gemini `Settings` dataclass, `get_settings` and its debug print are gone. Settings are grouped (models, context, retrieval, judge) with a module docstring; `USE_RETRIEVAL_PIPELINE_TOOL` is now env-configurable.
- [x] `backend/.env.example` added (models, context, retrieval and judge knobs). Both READMEs now point to it instead of a missing file.
- [x] `tests/test_in_memory_database.py` removed (imported a class that never existed). `python -m unittest discover tests` is green: 70 tests.
- [x] `static_google_search` removed from `tools.py` together with its unused imports.
- [x] `cv_analysis`: broken `nodes.py` (undefined `State`, `cv_graph`) and the empty `graph.py`/`prompts.py` removed; `state.py` kept with fixed imports and a docstring marking it as planned.
- [x] Web stage renamed from `query_ddu_google_search` / `RetrievalStage.GOOGLE` to `query_web_search` / `RetrievalStage.WEB` (it has been DuckDuckGo since the start).
- [x] Dead imports dropped (`attr.field` in `graph/retrieval/state.py`, `asdict`/`RetrievalResult` in `nodes.py`, `StrEnum` in `retrieval/schemas.py`).
- [x] `backend/README.md` rewritten: current setup, endpoints, commands and the current graph instead of the pre-rework Google-search diagram. Root README: tool list and prerequisites updated (no more llama3.1 / Google API).
- [x] `data/knowledge/knowledge.txt` "Ollama" and "Project Architecture" sections updated to the current layout. Run `python -m app.main rebuild knowledge` so the RAG collection picks it up.

Left alone on purpose (not broken, but worth a decision): the two tracked `*.egg-info/` directories and `backend/jobsresult_example.txt` are build/debug artifacts that could be removed from git; `backend/graph.png` is now a stale snapshot until regenerated.

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
- 2026-09-14: Phase 5 done. Markdown replies rendered to sanitized HTML by a `finalize` node. Whole turn now 11 to 20 s warm versus 57 to 70 s baseline. Next: Phase 2 (critique loop), then Phase 3.
- 2026-09-14: Phase 2 done. Failed verdicts loop back once through a `revise` node; short no-tool replies skip the judge. Files: `app/config.py`, `app/graph/core/{state,router,graph,nodes}.py`, `app/graph/judge/{node,critique}.py`, `tests/test_routing.py`. Next: Phase 3 (per-turn reset + trimming), then Phase 4.
- 2026-09-14: Phase 3 done. `prepare_turn` node, bounded retrieval window, history trimming, `num_ctx` on both clients. Test runs now use granite4.1:8b for chat and judge (user pulled granite4.1 3b and 8b). Files: `app/config.py`, `app/graph/core/{state,nodes,graph}.py`, `app/graph/judge/node.py`, `tests/test_turn_state.py`. Next: Phase 4 (retrieval speed).
- 2026-09-14: Phase 4 done. One-call pipeline with early exit, deterministic stage escalation, stricter static matcher, parallel page fetches, TTL cache, read_webpage fix. Local questions now 1.6 to 2.5 s on granite 8b. Files: `app/config.py`, `app/retrieval/engine.py`, `app/search/{static,ddgo}.py`, `app/graph/core/{tools,system_prompt}.py`, tests. Remaining: Phase 0 (measurement harness), Phase 6 (streaming), housekeeping.
- 2026-09-14: Phase 0 done. Benchmark harness in `backend/benchmarks/`, baseline stored. The harness found and I fixed three judge defects (blind to tool outputs, wrong assumed date, style nitpicks). 59 unit tests pass (`tests/test_judge_context.py` added); `tests/test_in_memory_database.py` remains the pre-existing broken file. Remaining: Phase 6 (streaming), housekeeping, and the follow-ups listed under Phase 0.
- 2026-09-14: Phase 6 done. SSE streaming endpoint with status/delta/reset/final events, React client with automatic fallback, verified over HTTP and in the browser. All six phases complete. Remaining: housekeeping list and the Phase 0 follow-ups.
- 2026-09-15: Judge context and turn scoping. Judge sees the recent conversation; verification and judge context are scoped to this turn's retrievals via `tool_call_id`; critique no longer asks to search again after a web search or to clarify; finalize note built from verification data only. Files: `app/graph/core/{router,tools,nodes}.py`, `app/graph/retrieval/state.py`, `app/graph/judge/{node,verify,critique,system_prompt}.py`, tests.
- 2026-09-15: Link verification in the judge. Pages behind cited links and curated sources are fetched before the verdict; unverifiable links are stripped and unconfirmed claims are noted in the final reply. Files: `app/graph/judge/{verify,node,schema,state,critique,system_prompt}.py`, `app/graph/core/nodes.py`, `app/search/ddgo.py`, `app/config.py`, `tests/test_verify.py`, `benchmarks/questions.json`.
- 2026-09-14: "what is expedition 33" fix. Static hits are decisive only on a full-name match; tentative hits no longer end the pipeline. Empty-revision fallback in `finalize`. Question set is now 16 rows. Files: `app/search/{static,static_data}.py`, `app/retrieval/engine.py`, `app/graph/core/{nodes,system_prompt}.py`, tests, `benchmarks/questions.json`.
