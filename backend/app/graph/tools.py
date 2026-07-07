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

from ..search.static import query_static_search
from ..search.linkedin import get_recent_jobs
from ..retrieval.service import retrieval_engine
from ..retrieval.schemas import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_results(query: str, results: list[RetrievalResult]) -> str:
    """Format the retrieval results into a structured JSON string."""
    if results:
        status = "FOUND"
        stage = results[0].stage
    else:
        status = "NO_MATCH"
        stage = "unknown"

    payload = {
        "stage": stage,
        "query": query,
        "status": status,
        "document_count": len(results),
        "documents": [asdict(r) for r in results]
    }   
    
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def retrieve_information(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """ Retrieves information from a knowledge base and web search functions. It includes a knowledge base and web search functions.
        It is a very very slow tool that may take a few seconds to return results, but it is very comprehensive and it is updated.
        Use it if you need to know the answer to a question that requires external knowledge or recent information.
    """
    logger.info("Retrieve_information tool called")
    results = retrieval_engine.run_pipeline(query)
    formatted = format_results(
        query=query,
        results=results,
    )

    return Command(update={"messages": [ToolMessage(content=formatted, tool_call_id=tool_call_id)]})

@tool
def retrieve_job_postings(query: str, location: str, remote_job: bool, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """ Retrieves job postings from LinkedIn. Use it if you need to search for recent job postings. It returns the last 10 job postings for the given query, location, and remote status.
        It is a slow tool that may take a few seconds to return results, but it is very comprehensive and it is updated.
        Consider data could be incomplete or missing, specially location.
        Limit is set to the last seven days or data.
        Location must include country, city is optional. For example: "France, Paris", "Chile" or "United States, New York".
        Use remote_job=True to filter for remote jobs only.
        
    """
    logger.info("Retrieve_job_postings tool called")

    jobs = get_recent_jobs(keyword=query, location=location, remote=remote_job)
    formatted = json.dumps(jobs, indent=2, ensure_ascii=False)

    return Command(update={"messages": [ToolMessage(content=formatted, tool_call_id=tool_call_id)]})

@tool
def get_current_time(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """ Return the current local date, time, and timezone of the machine running the application.
        Use it if you need to know the current date and time for any reason or queries.
    """
    logger.info("Get_current_time tool called")

    now = datetime.now().astimezone()
    result = {
        "Date": f"{now:%Y-%m-%d}",
        "Time": f"{now:%H:%M:%S}",
        "Timezone": now.tzname()
    }

    return Command(update={"messages": [ToolMessage(content=str(result), tool_call_id=tool_call_id)]})

@tool
def read_webpage(url: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """ Returns clean markdown/text from any webpage.
        Use it if you need to know the content of a webpage for any reason or queries.
        It returns plenty of data so use it for specific urls not for general search.
    """
    logger.info("read_webpage tool called for url: %s", url)

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
    """
    logger.info("Google static search tool called")
    results = query_static_search(str(query))

    return Command(update={"messages": [ToolMessage(content=str(results), tool_call_id=tool_call_id)]})