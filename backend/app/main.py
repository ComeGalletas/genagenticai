from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazily expose the FastAPI app so CLI commands avoid server imports."""
    if name == "app":
        from .server.api import app as fastapi_app
        return fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


USAGE = """Usage:
  python -m app.main rebuild [collection]   rebuild every registered Chroma collection, or one
  python -m app.main graph [path]           render the agent graph to a PNG (default: graph.png)
"""

if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "rebuild":
        from .db.chroma_store import rebuild_vectorstore
        target = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Rebuilding {target or 'all'} vector store collection(s) from knowledge files…")
        try:
            rebuild_vectorstore(target)
            print("Done — vector store rebuilt successfully.")
        except Exception as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    elif command == "graph":
        from .graph.core.graph import save_graph_png
        path = save_graph_png(sys.argv[2] if len(sys.argv) > 2 else "graph.png")
        print(f"Graph written to {path}")
    else:
        print(USAGE)
        sys.exit(1)
