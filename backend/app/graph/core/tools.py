from __future__ import annotations

import logging
import json
from dataclasses import asdict
from typing import Annotated
from datetime import datetime

import trafilatura
from trafilatura import fetch_url
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from ...search.static import query_static_search
from ...search.linkedin import get_recent_jobs
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


def format_results(query: str, results: list[RetrievalResult], stage: int = 0, next_stage: int = 0) -> RetrievalState | None:
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

    return payload


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def retrieve_information(query: str, tool_call_id: Annotated[str, InjectedToolCallId], stage: int = 0) -> Command:
    """Retrieves information from a knowledge base and web search functions.
        It is a very slow tool that may take a few seconds to return results, but it is very comprehensive and it is updated.
        There are stages to it starting from 0. The higher the stage, the more sources are searched so it becomes significantly slower.
        If the next_stage is negative, it means there is no more stages to search and the tool will return an empty result set.
    Args.
        query: The search query.
        tool_call_id: The unique identifier for the tool call.
        stage: The stage of the search. Default is 0. Stage 0 is a basic static search, stage 1 is a knowledge database search, and stage 2 is a web search.
    Returns:
        A Command object for the LLM containing the static search results as a JSON string with a list of RetrievalResult objects, or empty if no results.
    """
    logger.info("Retrieve_information tool called")
    results, next_stage  = retrieval_engine.run_stage(query=query, stage=stage)
    formatted = format_results(query=query, results=results, stage=stage, next_stage=next_stage)

    return Command(update={"retrieval": [formatted], "messages": [ToolMessage(content=f"Verified Retrieval Context - Retrieved {len(results)} document{'s' if len(results) != 1 else ''}.", tool_call_id=tool_call_id)]})


@tool
def retrieve_job_postings(query: str, location: str, remote_job: bool, limit: int, hours_old: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
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
    formatted = json.dumps(jobs, indent=2, ensure_ascii=False)

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
    result = None
    if not downloaded:
        result = "Failed to fetch page"        

    result = trafilatura.extract(
        downloaded,
        include_formatting=True, 
        include_links=True,
        include_images=False,
        output_format="markdown"
    )

    return Command(update={"messages": [ToolMessage(content=str(result if result else "No content extracted"), tool_call_id=tool_call_id)]})


@tool
def static_google_search(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """ Returns a fixed set of results for certain queries, without making external network requests. 
        It is fast to access but it has very few data and it is not updated. 
        Use it if you need to know the answer to a question that requires external knowledge or recent information.
    Args:
        query: The search query.
    Returns:
        A Command object for the LLM containing the static search results as a JSON string with a list of RetrievalResult objects, or empty if no results.
    """
    logger.info("Google static search tool called")
    results = query_static_search(str(query))

    return Command(update={"messages": [ToolMessage(content=str(results), tool_call_id=tool_call_id)]})
