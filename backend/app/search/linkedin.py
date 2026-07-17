import json
import logging

from typing import List, Dict, Any
from jobspy import scrape_jobs
from pandas import DataFrame

logger = logging.getLogger(__name__)

MAX_JOB_LIMIT = 10  # Maximum number of jobs to fetch
HOURS_OLD = 168  # 7 days in hours

logging.getLogger("JobSpy").setLevel(logging.WARNING)

def clear_job_data(jobs: DataFrame) -> List[Dict[str, Any]]:
    """
    Cleans and structures job data for LLM consumption.

    Args:
        jobs: DataFrame of raw job data from scrape_jobs
    """
    structured_jobs = []
    for _, job in jobs.iterrows():
        #print(f"Processing job: \n{job} ")
        structured_jobs.append({
            "job_url": job.job_url,
            "position": job.title,
            "company": job.company,
            "company_url": job.company_url,
            #"location": job.location,
            "salary": job.salary_source if job.salary_source else "Not listed",
            "date_posted": str(job.date_posted) if str(job.date_posted).lower() != "nan" else None,
            "description": str(job.description)[:500] + "..." if str(job.description).lower() != "none" else None,  # truncated
            "remote": job.is_remote,
        })

    return structured_jobs

def get_recent_jobs(keyword: str, location: str = "", remote: bool = False, limit: int = MAX_JOB_LIMIT, hours_old: int = HOURS_OLD, **kwargs) -> DataFrame | List[Dict[str, Any]]:
    """
    Fetches recent LinkedIn jobs (last week) and returns structured data
    optimized for LLM consumption.
    
    Args:
        keyword: Job title/skill (e.g. "software engineer")
        location: City, country, or "Remote"
        remote: Filter for remote jobs, Default: False
        limit: Max number of jobs to return: Default is 10.
        hours_old: How many hours old the job postings should be: Default is 168 (7 days).

    Returns:
        DataFrame | List of clean job dictionaries
    """
    logger.info("Job Posting query: %s | Location: %s, Remote: %s, Limit: %s, Hours Old: %s", keyword, location, remote, limit, hours_old)

    try:
        jobs: DataFrame = scrape_jobs(site_name=["linkedin"], search_term=keyword, location=location, results_wanted=limit, hours_old=hours_old, remote=remote, **kwargs)
        structured_jobs = clear_job_data(jobs)

        logger.info("Structured %d jobs for LLM consumption.", len(structured_jobs))
        return structured_jobs
    
    except Exception as e:
        logger.error("Error fetching jobs: %s", e)
        return []


if __name__ == "__main__":
    """Test main for the job scraping function. This will fetch recent job postings for a specific query and print them in a structured format."""
    jobs = get_recent_jobs(
        keyword="software engineer agentic ai",
        location="US, New York",
        remote=False,
        limit=10
    )
    
    print(json.dumps(jobs, indent=2, default=str))