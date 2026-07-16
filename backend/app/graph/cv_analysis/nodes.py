





def cv_analysis_node(state: State):
    """Analyze the CV and extract relevant information."""
    cv_text = state.get("cv")
    job_postings = state.get("job_postings", [])
    
    result = cv_graph.invoke(
        {
            "cv": state["cv"],
            "job_postings": state["job_postings"],
        }
    )

    return {
        "cv_summary": result["analysis"]
    }