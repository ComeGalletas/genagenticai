from __future__ import annotations

from .server.api import app

if __name__ == "__main__":
    """Main entry point for the vector store outside the agentic AI app. Rebuilds when starting the app as a python app with the 'rebuild' argument."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        from .db.chroma_store import rebuild_vectorstore
        print("Rebuilding vector store from knowledge files…")
        try:
            rebuild_vectorstore()
            print("Done — vector store rebuilt successfully.")
        except Exception as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        print("Usage: python -m app.main rebuild")
        sys.exit(1)
