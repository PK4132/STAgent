import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from squidpy_rag import (
    GeneratedAnswer,
    GradeDoc,
    SquidpyRAGTool,
    decide_execution,
    decide_relevance,
    _code_requires_data,
    _execution_failed,
    _prepare_code_for_execution,
)


def _base_state(**overrides):
    state = {
        "query": "test query",
        "context": [],
        "filtered_context": [],
        "answer": "",
        "generated_code": "",
        "execution_output": "",
        "execution_success": False,
        "chat_history": [],
        "rewrite_attempts": 0,
        "data_path": "",
    }
    state.update(overrides)
    return state


def _mock_tool_config(*, rag_exec_enabled=True, rag_max_rewrite_attempts=3):
    mock_config = MagicMock()
    mock_config.rag_exec_enabled = rag_exec_enabled
    mock_config.rag_exec_timeout = 30
    mock_config.rag_max_rewrite_attempts = rag_max_rewrite_attempts
    mock_config.max_output_length = 10000
    return mock_config


def _setup_structured_llm(mock_llm, *, grade_results=None, generated=None):
    mock_grader = MagicMock()
    if grade_results is not None:
        mock_grader.invoke.side_effect = grade_results
    else:
        mock_grader.invoke.return_value = GradeDoc(score="yes")

    mock_generator = MagicMock()
    mock_generator.invoke.return_value = generated or GeneratedAnswer(
        code="print('ok')",
        explanation="Final answer",
    )
    mock_llm.with_structured_output.side_effect = [mock_grader, mock_generator]
    return mock_grader, mock_generator


def test_decide_relevance_routes_to_generate_when_docs_present():
    state = _base_state(filtered_context=[Document(page_content="relevant doc")])
    assert decide_relevance(state) == "generate"


def test_decide_relevance_routes_to_rewrite_when_empty():
    state = _base_state(
        context=[Document(page_content="irrelevant")],
        filtered_context=[],
    )
    assert decide_relevance(state) == "rewrite_query"


@patch("squidpy_rag.get_tool_config")
def test_decide_relevance_loop_guard_at_max_rewrites(mock_get_tool_config):
    mock_get_tool_config.return_value = _mock_tool_config(rag_max_rewrite_attempts=3)
    state = _base_state(filtered_context=[], rewrite_attempts=3)
    assert decide_relevance(state) == "generate"


def test_decide_execution_routes_to_end_on_success():
    state = _base_state(execution_success=True)
    assert decide_execution(state) == "end"


def test_decide_execution_routes_to_rewrite_on_failure():
    state = _base_state(execution_success=False, rewrite_attempts=0)
    assert decide_execution(state) == "rewrite_query"


@patch("squidpy_rag.get_tool_config")
def test_decide_execution_loop_guard_at_max_rewrites(mock_get_tool_config):
    mock_get_tool_config.return_value = _mock_tool_config(rag_max_rewrite_attempts=3)
    state = _base_state(execution_success=False, rewrite_attempts=3)
    assert decide_execution(state) == "end"


def test_execution_failed_detects_traceback():
    assert _execution_failed("NameError('x is not defined')") is True
    assert _execution_failed("Execution timed out") is True
    assert _execution_failed("") is False
    assert _execution_failed("hello world") is False


def test_code_requires_data_heuristic():
    assert _code_requires_data("sdata = sd.read_zarr(DATA_PATH)") is True
    assert _code_requires_data("import squidpy as sq") is False


def test_prepare_code_injects_data_path():
    code = "import spatialdata as sd; sdata = sd.read_zarr(DATA_PATH)"
    zarr_path = r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr"
    prepared = _prepare_code_for_execution(code, zarr_path)
    assert prepared.startswith(f"DATA_PATH = {zarr_path!r}\n")
    assert "read_zarr(DATA_PATH)" in prepared


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_grade_documents_populates_filtered_context(
    mock_chat_openai, mock_setup_index, mock_get_tool_config
):
    mock_get_tool_config.return_value = _mock_tool_config()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="squidpy spatial neighbors", metadata={"source_repo": "squidpy"}),
        Document(page_content="unrelated utility code", metadata={"source_repo": "spatialdata"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    _setup_structured_llm(
        mock_llm,
        grade_results=[GradeDoc(score="yes"), GradeDoc(score="no")],
    )
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(
        _base_state(),
        config={"interrupt_after": ["grade_documents"]},
    )

    assert len(result["filtered_context"]) == 1
    assert result["filtered_context"][0].metadata["source_repo"] == "squidpy"


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_generate_structured_output(
    mock_chat_openai, mock_setup_index, mock_get_tool_config
):
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=False)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="spatial neighbors", metadata={"source_repo": "squidpy"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    _setup_structured_llm(
        mock_llm,
        generated=GeneratedAnswer(code="import squidpy", explanation="Use squidpy"),
    )
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(_base_state())

    assert result["answer"] == "Use squidpy"
    assert result["generated_code"] == "import squidpy"


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag._get_python_repl")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_execute_node_calls_repl(
    mock_chat_openai, mock_setup_index, mock_get_repl, mock_get_tool_config
):
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=True)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="version info", metadata={"source_repo": "squidpy"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_repl = MagicMock()
    mock_repl.run.return_value = "1.6.0"
    mock_get_repl.return_value = mock_repl

    mock_llm = MagicMock()
    _setup_structured_llm(
        mock_llm,
        generated=GeneratedAnswer(code="import squidpy; print(squidpy.__version__)", explanation="Print version"),
    )
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(_base_state())

    mock_repl.run.assert_called_once_with("import squidpy; print(squidpy.__version__)", timeout=30)
    assert result["execution_success"] is True
    assert result["execution_output"] == "1.6.0"


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag._get_python_repl")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_execute_skips_when_no_data_path(
    mock_chat_openai, mock_setup_index, mock_get_repl, mock_get_tool_config
):
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=True)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="read zarr", metadata={"source_repo": "spatialdata"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_repl = MagicMock()
    mock_get_repl.return_value = mock_repl

    mock_llm = MagicMock()
    _setup_structured_llm(
        mock_llm,
        generated=GeneratedAnswer(
            code="import spatialdata as sd; sd.read_zarr('/tmp/data.zarr')",
            explanation="Load zarr",
        ),
    )
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(_base_state(data_path=""))

    mock_repl.run.assert_not_called()
    assert result["execution_success"] is False
    assert "Execution skipped" in result["execution_output"]


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag._get_python_repl")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_execute_injects_data_path(
    mock_chat_openai, mock_setup_index, mock_get_repl, mock_get_tool_config
):
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=True)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="read zarr", metadata={"source_repo": "spatialdata"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_repl = MagicMock()
    mock_repl.run.return_value = "loaded"
    mock_get_repl.return_value = mock_repl

    mock_llm = MagicMock()
    _setup_structured_llm(
        mock_llm,
        generated=GeneratedAnswer(
            code="import spatialdata as sd; sd.read_zarr(DATA_PATH)",
            explanation="Load zarr",
        ),
    )
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    zarr_path = r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr"
    result = tool.rag_pipeline.invoke(_base_state(data_path=zarr_path))

    expected_code = _prepare_code_for_execution(
        "import spatialdata as sd; sd.read_zarr(DATA_PATH)",
        zarr_path,
    )
    mock_repl.run.assert_called_once_with(expected_code, timeout=30)
    assert result["execution_success"] is True
    assert result["execution_output"] == "loaded"


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_rewrite_loop_guard_fires_at_max_attempts(
    mock_chat_openai, mock_setup_index, mock_get_tool_config
):
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=False)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="irrelevant", metadata={"source_repo": "squidpy"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    mock_grader = MagicMock()
    mock_grader.invoke.return_value = GradeDoc(score="no")
    mock_generator = MagicMock()
    mock_generator.invoke.return_value = GeneratedAnswer(code="", explanation="Final answer")
    mock_llm.with_structured_output.side_effect = [mock_grader, mock_generator]

    rewrite_responses = [
        MagicMock(content="rewritten query 1"),
        MagicMock(content="rewritten query 2"),
        MagicMock(content="rewritten query 3"),
    ]
    mock_llm.invoke.side_effect = rewrite_responses
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(_base_state(query="do the thing"))

    assert result["rewrite_attempts"] == 3
    assert result["answer"] == "Final answer"
    assert mock_retriever.invoke.call_count == 4
