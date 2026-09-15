"""Run the fixed question set through the agent graph and record timings and answers.

Usage (from the backend directory, with Ollama running):

    python -m benchmarks.run_benchmark                      # all questions
    python -m benchmarks.run_benchmark --only rag web       # categories or ids
    python -m benchmarks.run_benchmark --skip jobs          # skip slow/rate-limited categories
    python -m benchmarks.run_benchmark --baseline benchmarks/results/<file>.json   # show deltas

Models come from the usual env vars (CHAT_MODEL, JUDGE_MODEL). Each run writes
benchmarks/results/<timestamp>_<chat-model>.json and a .md summary next to it.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.config import CHAT_MODEL, JUDGE_MODEL
from app.graph.core.graph import graph
from app.retrieval.service import retrieval_engine

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.json"
RESULTS_DIR = HERE / "results"
_TAG = re.compile(r"<[^>]+>")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def run_question(item: dict, thread_prefix: str) -> dict:
    """Run one question and collect per-node timings from the update stream."""
    config = RunnableConfig(configurable={"thread_id": f"{thread_prefix}-{item['thread']}"})
    node_times: dict[str, float] = {}
    node_visits: dict[str, int] = {}
    path: list[str] = []
    verdicts: list[dict] = []  # every judge verdict in order, not just the final one
    error: str | None = None

    start = time.perf_counter()
    previous = start
    try:
        for event in graph.stream({"messages": [HumanMessage(item["question"])]}, config=config, stream_mode="updates"):
            now = time.perf_counter()
            for node, update in event.items():
                path.append(node)
                node_times[node] = node_times.get(node, 0.0) + (now - previous)
                node_visits[node] = node_visits.get(node, 0) + 1
                if node == "judge" and update and update.get("judge"):
                    verdict = update["judge"]
                    verdicts.append({k: verdict.get(k) for k in ("passed", "grounded", "needs_more_info", "issues", "error")})
            previous = now
    except Exception as exc:  # keep going; the failure is part of the record
        error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - start

    state = graph.get_state(config).values
    retrievals = state.get("retrieval") or []
    last_retrieval = retrievals[-1] if retrievals else {}
    judge = state.get("judge") or None
    tools_used = sorted({
        call["name"]
        for message in state.get("messages", [])
        if getattr(message, "type", "") == "ai"
        for call in (getattr(message, "tool_calls", None) or [])
    })
    reply_html = state.get("final_response") or ""
    reply_text = _TAG.sub("", reply_html).strip()
    # Terms from an earlier question in the same thread that must not bleed into this reply.
    leaked = [term for term in item["forbid"] if term.lower() in reply_text.lower()] if item.get("forbid") else None

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "thread": item["thread"],
        "total_s": round(total, 2),
        "node_times_s": {k: round(v, 2) for k, v in node_times.items()},
        "node_visits": node_visits,
        "path": path,
        "tools_used": tools_used,
        "retrieval_stage": last_retrieval.get("retrieval_stage"),
        "retrieval_docs": len(last_retrieval.get("retrieval_documents") or []),
        "judge": None if judge is None else {
            "passed": judge.get("passed"),
            "grounded": judge.get("grounded"),
            "needs_more_info": judge.get("needs_more_info"),
            "issues": judge.get("issues"),
            "error": judge.get("error"),
        },
        "judge_attempts": state.get("judge_attempts", 0),
        "verdicts": verdicts,
        "user_language": state.get("user_language"),
        "leaked": leaked,
        "reply_text": reply_text,
        "reply_html": reply_html,
        "error": error,
    }


def _judge_label(result: dict) -> str:
    judge = result["judge"]
    if judge is None:
        return "skipped"
    label = "pass" if judge["passed"] else "FAIL"
    if result["judge_attempts"]:
        label += f" (+{result['judge_attempts']} revise)"
    return label


def _leak_label(result: dict) -> str:
    """'-' when the question had no forbidden terms, 'none' when none leaked, else the leaked terms."""
    if result.get("leaked") is None:
        return "-"
    return ", ".join(result["leaked"]) if result["leaked"] else "none"


def summary_markdown(run: dict, baseline: dict | None) -> str:
    base_by_id = {r["id"]: r for r in (baseline or {}).get("results", [])}
    lines = [
        f"# Benchmark {run['timestamp']}",
        "",
        f"chat={run['chat_model']} judge={run['judge_model']} commit={run['git_commit']}",
        "",
        "| id | category | total s | chatbot s | tools s | judge s | stage | tools used | judge | leak | " + ("delta s | " if baseline else "") + "reply |",
        "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if baseline else "") + "---|",
    ]
    for r in run["results"]:
        nt = r["node_times_s"]
        delta = ""
        if baseline:
            base = base_by_id.get(r["id"])
            delta = f"{r['total_s'] - base['total_s']:+.1f} | " if base else "n/a | "
        reply = r["reply_text"].replace("|", "/").replace("\n", " ")[:90]
        if r["error"]:
            reply = f"ERROR {r['error'][:80]}"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['total_s']} | {nt.get('chatbot', 0)} | {nt.get('tools', 0)} | "
            f"{nt.get('judge', 0)} | {r['retrieval_stage'] if r['retrieval_stage'] is not None else '-'} | "
            f"{', '.join(r['tools_used']) or '-'} | {_judge_label(r)} | {_leak_label(r)} | {delta}{reply} |"
        )
    totals = [r["total_s"] for r in run["results"] if not r["error"]]
    if totals:
        lines += ["", f"questions: {len(run['results'])} | errors: {sum(1 for r in run['results'] if r['error'])} | "
                  f"sum: {sum(totals):.1f} s | mean: {sum(totals) / len(totals):.1f} s | max: {max(totals):.1f} s"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=[], help="categories or ids to run")
    parser.add_argument("--skip", nargs="*", default=[], help="categories or ids to skip")
    parser.add_argument("--baseline", type=Path, help="previous results .json to compare against")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--quiet", action="store_true", help="hide per-node log lines")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "primp", "trafilatura", "app.db", "app.search", "JobSpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    items = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.only:
        items = [i for i in items if i["id"] in args.only or i["category"] in args.only]
    if args.skip:
        items = [i for i in items if i["id"] not in args.skip and i["category"] not in args.skip]
    if not items:
        sys.exit("No questions selected.")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    retrieval_engine.clear_cache()  # every run starts cold on retrieval, like a fresh server
    thread_prefix = f"bench-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    run = {
        "timestamp": timestamp,
        "chat_model": CHAT_MODEL,
        "judge_model": JUDGE_MODEL,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "results": [],
    }
    print(f"\nBenchmark {timestamp} | chat={CHAT_MODEL} judge={JUDGE_MODEL} | {len(items)} question(s)\n")
    for index, item in enumerate(items, 1):
        print(f"[{index}/{len(items)}] {item['id']}: {item['question']}")
        result = run_question(item, thread_prefix)
        run["results"].append(result)
        status = f"ERROR {result['error']}" if result["error"] else f"{result['total_s']}s | stage={result['retrieval_stage']} | judge={_judge_label(result)}"
        if result.get("leaked"):
            status += f" | LEAK: {', '.join(result['leaked'])}"
        print(f"    -> {status}\n       {result['reply_text'][:120]}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{timestamp}_{re.sub(r'[^A-Za-z0-9.-]+', '-', CHAT_MODEL)}"
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = summary_markdown(run, baseline)
    (RESULTS_DIR / f"{stem}.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nSaved {RESULTS_DIR / stem}.json and .md")


if __name__ == "__main__":
    main()
