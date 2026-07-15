import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from squidpy_rag import GradeDoc, SquidpyRAGTool, decide_relevance


def _base_state(**overrides):
    state = {
        "query": "test query",
        "context": [],
        "filtered_context": [],
        "answer": "",
        "chat_history": [],
        "rewrite_attempts": 0,
    }
    state.update(overrides)
    return state


def test_decide_relevance_routes_to_generate_when_docs_present():
    state = _base_state(filtered_context=[Document(page_content="relevant doc")])
    assert decide_relevance(state) == "generate"


def test_decide_relevance_routes_to_rewrite_when_empty():
    state = _base_state(
        context=[Document(page_content="irrelevant")],
        filtered_context=[],
    )
    assert decide_relevance(state) == "rewrite_query"


def test_decide_relevance_loop_guard_at_max_rewrites():
    state = _base_state(filtered_context=[], rewrite_attempts=3)
    assert decide_relevance(state) == "generate"


@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_grade_documents_populates_filtered_context(mock_chat_openai, mock_setup_index):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="squidpy spatial neighbors", metadata={"source_repo": "squidpy"}),
        Document(page_content="unrelated utility code", metadata={"source_repo": "spatialdata"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_grader = MagicMock()
    mock_grader.invoke.side_effect = [
        GradeDoc(score="yes"),
        GradeDoc(score="no"),
    ]
    mock_llm.with_structured_output.return_value = mock_grader
    mock_llm.invoke.return_value = MagicMock(content="Generated answer")
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(
        _base_state(),
        config={"interrupt_after": ["grade_documents"]},
    )

    assert len(result["filtered_context"]) == 1
    assert result["filtered_context"][0].metadata["source_repo"] == "squidpy"


@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_rewrite_loop_guard_fires_at_max_attempts(mock_chat_openai, mock_setup_index):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="irrelevant", metadata={"source_repo": "squidpy"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_grader = MagicMock()
    mock_grader.invoke.return_value = GradeDoc(score="no")

    rewrite_responses = [
        MagicMock(content="rewritten query 1"),
        MagicMock(content="rewritten query 2"),
        MagicMock(content="rewritten query 3"),
    ]
    mock_llm.with_structured_output.return_value = mock_grader
    mock_llm.invoke.side_effect = rewrite_responses + [MagicMock(content="Final answer")]
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(_base_state(query="do the thing"))

    assert result["rewrite_attempts"] == 3
    assert result["answer"] == "Final answer"
    assert mock_retriever.invoke.call_count == 4
