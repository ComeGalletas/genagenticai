from __future__ import annotations

import logging
import json
from typing import Annotated
from datetime import datetime

import trafilatura
from trafilatura import fetch_url
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from ...db.chroma_store import load_vectorstore
from ...db.rag import query_knowledge
from ...db.baloto import BALOTO_COLLECTION, suggest_numbers
from ...search.linkedin import get_recent_jobs, HOURS_OLD, MAX_JOB_LIMIT
from ..retrieval.state import RetrievalState, RetrievalDocument
from ...retrieval.schemas import RetrievalResult
from ...retrieval.service import retrieval_engine


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_document(result: RetrievalResult) -> RetrievalDocument:
    """Converts a RetrievalResult object into a RetrievalDocument dictionary for LLM usage.
    Args:
        result: A RetrievalResult object containing the title, content, source URL, stage of retrieval, confidence score, and additional metadata.
    Returns:
        A RetrievalDocument dictionary containing the same information as the RetrievalResult object, formatted for LLM usage."""
    return {
        "title": result.title,
        "content": result.content,
        "source": result.source,
        "stage": result.stage,
        "confidence": result.confidence or None,
        "metadata": result.metadata,
    }


def format_results(
    query: str,
    results: list[RetrievalResult],
    stage: int = 0,
    next_stage: int = 0,
    tool_call_id: str | None = None,
) -> RetrievalState | None:
    """Format the retrieval results into a structured RetrievalState dictionary and adds extra metadata for the LLM to process.
    If no results are found, the status is set to "NO_MATCH" and None is returned.
    It does not know about the stages of the retrieval process, it just formats the results and adds the stage and next_stage metadata for the LLM to process.
    Args:
        query: The search query.
        results: A list of RetrievalResult objects.
        stage: The current stage of the search.
        next_stage: The next stage of the search.
    Returns:
        A RetrievalState object containing the formatted retrieval results, or None if no results are found.
    """
    if results:
        status = "FOUND"
    else:
        status = "NO_MATCH"

    payload = RetrievalState(
        retrieval_status=status,
        retrieval_query=query,
        retrieval_stage=stage,
        retrieval_next_stage=next_stage,
        retrieval_documents=[
            _to_document(doc) for doc in results
        ],
    )
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id

    return payload


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def retrieve_information(query: str, tool_call_id: Annotated[str, InjectedToolCallId], stage: int = 0) -> Command:
    """Searches for information about the query. One call tries the sources in order and stops at the
        first one that returns results: stage 0 is a small curated index (video games, films, dev
        frameworks), stage 1 is the local knowledge base (NVIDIA RTX 50 GPUs, LangGraph), stage 2 is a
        live web search. Local stages answer in milliseconds; the web stage takes a few seconds.
    Args:
        query: The search query.
        tool_call_id: The unique identifier for the tool call.
        stage: The first stage to try. Default 0. Pass 2 to escalate to a live web search when an
            earlier call with the same query returned nothing useful. A higher stage is only honored
            after the cheaper stages were tried for that query.
    Returns:
        A Command that stores the retrieved documents as verified context and tells you how many were found.
    """
    start_stage = retrieval_engine.resolve_start_stage(query, stage)
    logger.info("Retrieve_information tool called | requested_stage=%d start_stage=%d", stage, start_stage)
    results, last_stage, next_stage = retrieval_engine.run_pipeline(query=query, start_stage=start_stage)
    formatted = format_results(query=query, results=results, stage=last_stage, next_stage=next_stage, tool_call_id=tool_call_id)

    count = len(results)
    if count:
        note = f"Verified Retrieval Context - Retrieved {count} document{'s' if count != 1 else ''} (stage {last_stage})."
    elif next_stage == -1:
        note = "No documents found in any source, including the live web. Do not invent an answer."
    else:
        note = f"No documents found up to stage {last_stage}. Call again with stage={next_stage} to search further."

    return Command(update={"retrieval": [formatted], "messages": [ToolMessage(content=note, tool_call_id=tool_call_id)]})


@tool
def retrieve_baloto_results(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Retrieve Baloto results only from the dedicated Baloto vector collection.
        Use it for Baloto draws, dates, winning numbers, history, and month-based result questions.
    Args:
        query: The Baloto query.
        tool_call_id: The unique identifier for the tool call.
    Returns:
        A Command containing only Baloto retrieval results.
    """
    logger.info("Retrieve_baloto_results tool called")
    results = query_knowledge(question=query, collection=BALOTO_COLLECTION)
    formatted = format_results(query=query, results=results, stage=1, next_stage=-1, tool_call_id=tool_call_id)

    return Command(update={"retrieval": [formatted], "messages": [ToolMessage(content=f"Verified Baloto Context - Retrieved {len(results)} document{'s' if len(results) != 1 else ''}.", tool_call_id=tool_call_id)]})


@tool
def suggest_baloto_numbers(
    tool_call_id: Annotated[str, InjectedToolCallId],
    window: int = 100,
    half_life: float = 50.0,
    strategy: str = "positional",
    pool_size: int = 3,
    cold_weight: float = 0.5,
) -> Command:
    """Compute Baloto numbers for the next draw from the historical behaviour of every number.
        Use it when the user asks which numbers to play, for a prediction, or for the hottest or most overdue numbers.
        The result contains a deterministic ticket plus, per position, the runner-up candidates you may swap in.
        Never invent numbers: only use numbers present in the returned candidate lists.
    Args:
        tool_call_id: The unique identifier for the tool call.
        window: How many of the most recent draws to analyse. 0 means the whole history. Default 100.
        half_life: Number of draws after which an older draw counts half as much. 0 disables recency weighting. Default 50.
        strategy: "positional" and "overall" pick the most frequent (hot) numbers, "cold" picks the most overdue ones, "hybrid" blends hot and cold. Cold and hybrid need a large window, use 0 for the full history.
        pool_size: How many alternative numbers to return per position. Default 3.
        cold_weight: Only used by "hybrid". Share of the score coming from the overdue ranking, between 0.0 and 1.0. Default 0.5.
    Returns:
        A Command containing the deterministic ticket, the candidate pools, what the scores mean and the analysed draw range as JSON.
    """
    logger.info("Suggest baloto numbers tool called")

    vectorstore = load_vectorstore(BALOTO_COLLECTION)
    if vectorstore is None:
        content = "The Baloto history is unavailable, so no numbers can be computed."
    else:
        result = suggest_numbers(
            vectorstore,
            window=max(0, min(window, 5000)),
            half_life=max(0.0, min(half_life, 5000.0)),
            strategy=strategy if strategy in {"positional", "overall", "cold", "hybrid"} else "positional",
            pool_size=max(1, min(pool_size, 5)),
            cold_weight=max(0.0, min(cold_weight, 1.0)),
        )
        content = json.dumps(result, indent=2, ensure_ascii=False)

    return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]})


@tool
def retrieve_job_postings(query: str, location: str, tool_call_id: Annotated[str, InjectedToolCallId], remote_job: bool = False, limit: int = MAX_JOB_LIMIT, hours_old: int = HOURS_OLD) -> Command:
    """Retrieves job postings from LinkedIn. Use it if you need to search for recent job postings. It returns the last 'limit' job postings for the given query, location, and remote status.
        It is a slow tool that may take a few seconds to return results, but it is very comprehensive and it is updated.
        Consider data could be incomplete or missing, specially location.
    Args:
        query: The search query for the job title or keywords.
        location: The location to search for jobs. It must include the country, city is optional
        remote_job: A boolean indicating whether to filter for remote jobs only. If True, only remote jobs will be returned. If False, returns any other jobs.
        limit: The maximum number of job postings to retrieve. Defaults to 10.
        hours_old: The maximum age of job postings in hours. Defaults to 168 (7 days).
        tool_call_id: The unique identifier for the tool call.
    Returns:
        A Command object for the LLM containing the retrieved job postings in clean JSON format.
    """
    logger.info("Retrieve job postings tool called")

    jobs = get_recent_jobs(keyword=query, location=location, remote=remote_job, limit=limit, hours_old=hours_old)
    formatted = json.dumps(jobs, indent=2, ensure_ascii=False) if jobs else "No job postings returned by LinkedIn for this search. Do not invent any."

    return Command(update={"messages": [ToolMessage(content=formatted, tool_call_id=tool_call_id)]})


@tool
def get_current_time(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Return the current local date, time, and timezone of the machine running the application.
        Use it if you need to know the current date and time for any reason or queries.
    Returns:
        A Command object for the LLM containing the current local date, time, and timezone in clean JSON format.
    """
    logger.info("Get current time tool called")

    now = datetime.now().astimezone()
    result = {
        "Date": f"{now:%Y-%m-%d}",
        "Time": f"{now:%H:%M:%S}",
        "Timezone": now.tzname()
    }

    return Command(update={"messages": [ToolMessage(content=json.dumps(result, indent=2, ensure_ascii=False), tool_call_id=tool_call_id)]})

@tool
def read_webpage(url: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """ Returns clean markdown/text from any webpage.
        Use it if you need to know the content of a webpage for any reason or queries.
        It returns plenty of data so use it for specific urls not for general search.
    Args:
        url: The URL of the webpage to read.
    Returns:
        A Command object for the LLM containing the content of the webpage as a markdown string.
    """
    logger.info("Read webpage tool called for url: %s", url)

    downloaded = fetch_url(url)
    if not downloaded:
        logger.warning("Read webpage: fetch failed for %s", url)
        return Command(update={"messages": [ToolMessage(content=f"Failed to fetch page: {url}", tool_call_id=tool_call_id)]})

    result = trafilatura.extract(
        downloaded,
        include_formatting=True,
        include_links=True,
        include_images=False,
        output_format="markdown"
    )

    return Command(update={"messages": [ToolMessage(content=str(result if result else "No content extracted"), tool_call_id=tool_call_id)]})
