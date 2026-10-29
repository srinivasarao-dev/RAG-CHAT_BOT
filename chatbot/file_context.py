"""Turn a conversation's uploaded files into model context: full text when small, retrieval when large."""
from __future__ import annotations

from langchain_core.tools import tool

from chatbot.documents import embed_query

FILES_INSTRUCTIONS = """The user has attached files to this conversation. When a question is about them:
- Base the answer on the files' content; quote short relevant passages and name the file (and the page,
  when the text shows [Page N] markers).
- If the files don't contain the answer, say so plainly, then answer from other sources if appropriate.
- Use live tools only for information the files can't provide (e.g. current prices or news)."""


def build_file_context(store, files: list[dict], question: str, budget_chars: int) -> tuple[str, list]:
    """Returns (system prompt addition, extra tools) for the conversation's files."""
    if not files:
        return "", []
    ids = [f["id"] for f in files]
    texts = store.get_file_texts(ids)
    total = sum(len(t or "") for t in texts.values())
    listing = "\n".join(f"- {f['name']} ({(f.get('size') or 0) // 1024} KB, {f.get('n_chunks') or 0} sections)" for f in files)

    if total <= budget_chars:
        # Small enough: give the model the complete text, which beats retrieval for summaries and overviews
        docs = "\n\n".join(f'<file name="{f["name"]}">\n{texts.get(f["id"], "")}\n</file>' for f in files)
        return f"{FILES_INSTRUCTIONS}\n\nAttached files:\n{listing}\n\n<files>\n{docs}\n</files>", []

    # Large: pre-fetch the best-matching passages for this question, and let the model search for more
    hits = store.search_chunks(ids, embed_query(question), k=6)
    passages = "\n\n".join(
        f'<passage file="{h["file_name"]}" section="{h["chunk_index"] + 1}">\n{h["content"]}\n</passage>' for h in hits
    )

    @tool
    def search_files(query: str) -> str:
        """Search the user's uploaded files for passages relevant to a query. Use it for any question about
        the attached documents that the passages already provided don't fully answer. Try specific queries."""
        results = store.search_chunks(ids, embed_query(query), k=6)
        if not results:
            return "No matching passages."
        return "\n\n".join(
            f'[{r["file_name"]}, section {r["chunk_index"] + 1}, relevance {r["score"]:.2f}]\n{r["content"]}' for r in results
        )

    context = (
        f"{FILES_INSTRUCTIONS}\n\nAttached files (too large to include in full):\n{listing}\n\n"
        f"Passages most relevant to the latest question:\n{passages}\n\n"
        "Use the search_files tool to look up other parts of the files."
    )
    return context, [search_files]
