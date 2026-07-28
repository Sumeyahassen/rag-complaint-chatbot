"""
CrediTrust Complaint Insights — Streamlit chat interface for the RAG pipeline.

This is a scaffold: replace `run_rag_pipeline()` with a call into your
existing retrieval + generation module (e.g. `from src.rag_pipeline import answer_question`).

Expected contract for that function:
    answer_question(question: str, k: int = 5) -> RagResult

Run with:
    streamlit run app.py
"""

from dataclasses import dataclass, field
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

APP_TITLE = "CrediTrust Complaint Insights"
APP_SUBTITLE = "Ask a question about customer complaints and get an evidence-backed answer."
DEFAULT_TOP_K = 5


@dataclass
class SourceChunk:
    """A single retrieved chunk shown to the user for trust/verification."""
    complaint_id: str
    product_category: str
    text: str
    score: float | None = None


@dataclass
class RagResult:
    """Structured result returned by the RAG pipeline."""
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)


# --------------------------------------------------------------------------
# TODO: Replace this stub with your real pipeline.
# Example:
#   from src.rag_pipeline import answer_question as run_rag_pipeline
# and delete the stub function below.
# --------------------------------------------------------------------------

def run_rag_pipeline(question: str, k: int = DEFAULT_TOP_K) -> RagResult:
    """Stub implementation — swap this out for your real retriever + LLM call."""
    raise NotImplementedError(
        "Wire this up to your existing RAG pipeline module, e.g. "
        "from src.rag_pipeline import answer_question"
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (question, RagResult)


def render_sources(sources: list[SourceChunk]) -> None:
    if not sources:
        st.caption("No supporting sources were retrieved for this answer.")
        return
    st.markdown("**Sources used to generate this answer:**")
    for i, s in enumerate(sources, start=1):
        label = f"Source {i} — {s.product_category} (complaint #{s.complaint_id})"
        with st.expander(label):
            st.write(s.text)
            if s.score is not None:
                st.caption(f"Relevance score: {s.score:.3f}")


def handle_question(question: str, k: int) -> None:
    """Run the pipeline with error handling so a bad query never crashes the app."""
    if not question or not question.strip():
        st.warning("Please enter a question before submitting.")
        return

    with st.spinner("Retrieving relevant complaints and generating an answer..."):
        try:
            result = run_rag_pipeline(question.strip(), k=k)
        except NotImplementedError as e:
            st.error(f"Pipeline not connected yet: {e}")
            return
        except Exception as e:
            # Broad catch is intentional here: this is the user-facing boundary,
            # so we never want a raw traceback surfacing to a non-technical user.
            st.error(
                "Something went wrong while generating an answer. "
                "Please try rephrasing your question or try again shortly."
            )
            st.caption(f"Technical detail (for debugging): {e}")
            return

    st.session_state.history.append((question.strip(), result))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💳", layout="centered")
    init_session_state()

    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)

    with st.sidebar:
        st.subheader("Settings")
        k = st.slider("Number of sources to retrieve (k)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()

    question = st.text_input(
        "Ask a question about customer complaints",
        placeholder='e.g. "Why are people unhappy with Credit Cards?"',
    )
    if st.button("Ask", type="primary"):
        handle_question(question, k)

    st.divider()

    # Most recent answer first
    for q, result in reversed(st.session_state.history):
        st.markdown(f"**Q: {q}**")
        st.write(result.answer)
        render_sources(result.sources)
        st.divider()


if __name__ == "__main__":
    main()