import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from squidpy_rag import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GeneratedAnswer,
    GradeDoc,
    RETRIEVED_DOC_TYPES,
    RepoWeightedRetriever,
    SquidpyRAGTool,
    build_combined_index,
    decide_execution,
    decide_relevance,
    _code_requires_data,
    _data_access_hint,
    _doc_type_for_source,
    _execution_failed,
    _format_api_index,
    _format_context,
    _index_fingerprint,
    _module_label,
    _prepare_code_for_execution,
    _preloaded_binding,
    _repo_filter,
    _resolve_question,
    _stale_repos,
    _unknown_api_symbols,
    _wrong_loader_reason,
)


# Mirrors the shape of spatialdata's real _LAZY_IMPORTS / _submodules surface.
_FAKE_API_SURFACE = {
    "version": "0.7.3",
    "symbols": ("SpatialData", "aggregate", "bounding_box_query", "get_extent", "read_zarr"),
    "submodules": ("datasets", "models", "transformations"),
}
_EMPTY_API_SURFACE = {"version": "", "symbols": (), "submodules": ()}


@pytest.fixture(autouse=True)
def _stub_installed_api_surface():
    """Keep the suite away from the real spatialdata import, which is slow and may be a
    different version than these tests describe."""
    with patch("squidpy_rag._installed_api_surface", return_value=_FAKE_API_SURFACE):
        yield


def _fingerprint(**overrides):
    base = {
        "schema": 2,
        "refs": {"squidpy": "", "spatialdata": "v0.7.3"},
        "commits": {"squidpy": "aaaaaaaaaaaa", "spatialdata": "bbbbbbbbbbbb"},
        "chunking": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
        "embedding_model": "nomic-embed-text-v1.5",
        "installed": {"squidpy": "1.6.5", "spatialdata": "0.7.3"},
    }
    base.update(overrides)
    return base


_REPOS = ("squidpy", "spatialdata")


def _base_state(**overrides):
    state = {
        "query": "test query",
        "original_query": "test query",
        "previous_queries": [],
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


def _mock_tool_config(
    *,
    rag_exec_enabled=True,
    rag_max_rewrite_attempts=3,
    rag_index_refresh="auto",
):
    mock_config = MagicMock()
    mock_config.rag_exec_enabled = rag_exec_enabled
    mock_config.rag_exec_timeout = 30
    mock_config.rag_max_rewrite_attempts = rag_max_rewrite_attempts
    mock_config.max_output_length = 10000
    mock_config.rag_index_refresh = rag_index_refresh
    mock_config.rag_spatialdata_ref = "v0.7.3"
    mock_config.rag_squidpy_ref = ""
    return mock_config


@contextmanager
def _stubbed_index_build(store_exists):
    """Let build_combined_index's decisions be observed without git, Chroma or embeddings."""
    with ExitStack() as stack:
        stack.enter_context(patch("squidpy_rag._sync_repo", return_value="deadbeef1234"))
        stack.enter_context(patch("squidpy_rag.os.path.exists", return_value=store_exists))
        stack.enter_context(patch("squidpy_rag._read_index_manifest", return_value={}))
        stack.enter_context(patch("squidpy_rag._get_lm_studio_embeddings"))
        stack.enter_context(patch("squidpy_rag.Chroma"))
        stack.enter_context(patch("squidpy_rag.RepoWeightedRetriever"))
        stack.enter_context(patch("squidpy_rag._write_index_manifest"))
        stack.enter_context(patch("squidpy_rag._print_index_summary"))
        yield stack.enter_context(patch("squidpy_rag._reindex_repo"))


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


def test_module_label_resolves_import_path():
    assert (
        _module_label(
            {
                "source_repo": "squidpy",
                "source": r"packages_available\squidpy\src\squidpy\gr\neighbors.py",
            }
        )
        == "squidpy.gr.neighbors"
    )
    assert (
        _module_label(
            {
                "source_repo": "squidpy",
                "source": r"packages_available\squidpy\src\squidpy\gr\__init__.py",
            }
        )
        == "squidpy.gr"
    )
    assert (
        _module_label(
            {
                "source_repo": "spatialdata",
                "source": r"packages_available\spatialdata\src\spatialdata\_io\io_zarr.py",
            }
        )
        == "spatialdata._io.io_zarr"
    )


def test_module_label_falls_back_outside_src():
    label = _module_label(
        {
            "source_repo": "squidpy",
            "source": r"packages_available\squidpy\.scripts\ci\download_data.py",
        }
    )
    assert label.startswith("squidpy: ")
    assert _module_label({"source_repo": "squidpy"}) == "squidpy"


def test_format_context_exposes_module_paths():
    docs = [
        Document(
            page_content="def spatial_neighbors(adata): ...",
            metadata={
                "source_repo": "squidpy",
                "source": r"packages_available\squidpy\src\squidpy\gr\_build.py",
            },
        )
    ]
    rendered = _format_context(docs)
    assert "# from squidpy.gr._build" in rendered
    assert "def spatial_neighbors" in rendered


def test_preloaded_binding_matches_extension():
    assert _preloaded_binding(r"C:\data\sample.zarr")[0] == "sdata"
    assert _preloaded_binding(r"C:\data\sample.h5ad")[0] == "adata"
    assert _preloaded_binding(r"C:\data\sample.csv") is None
    assert _preloaded_binding("") is None


def test_data_access_hint_describes_preloaded_object():
    zarr_hint = _data_access_hint(r"C:\data\sample.zarr")
    assert "already loaded" in zarr_hint.lower()
    assert "`sdata`" in zarr_hint
    assert "sdata.tables" in zarr_hint

    h5ad_hint = _data_access_hint(r"C:\data\sample.h5ad")
    assert "`adata`" in h5ad_hint

    # Unknown extensions have no preamble, so the model still gets DATA_PATH.
    assert "DATA_PATH" in _data_access_hint(r"C:\data\sample.csv")


def test_wrong_loader_reason_rejects_reloading_preloaded_data():
    zarr_path = r"C:\data\sample.zarr"
    h5ad_path = r"C:\data\sample.h5ad"

    reason = _wrong_loader_reason("sdata = sd.read_zarr(DATA_PATH)", zarr_path)
    assert "already loaded as `sdata`" in reason
    assert "already loaded as `adata`" in _wrong_loader_reason(
        "adata = sc.read_h5ad(DATA_PATH)", h5ad_path
    )
    assert _wrong_loader_reason("print(list(sdata.tables))", zarr_path) is None


def test_wrong_loader_reason_flags_package_ownership():
    """Attributing a loader to the wrong package is the failure mode this guard exists for."""
    assert "squidpy has no read_zarr" in _wrong_loader_reason("sq.read_zarr(DATA_PATH)", "")
    assert "squidpy has no read_h5ad" in _wrong_loader_reason("sq.read_h5ad(DATA_PATH)", "")
    assert "spatialdata has no read_h5ad" in _wrong_loader_reason(
        "spatialdata.read_h5ad(DATA_PATH)", ""
    )
    assert "scanpy has no read_zarr" in _wrong_loader_reason("sc.read_zarr(DATA_PATH)", "")
    assert _wrong_loader_reason("sd.read_zarr(DATA_PATH)", "") is None
    assert _wrong_loader_reason("sc.read_h5ad(DATA_PATH)", "") is None


def test_repo_weighted_retriever_splits_search_by_repo():
    """Most of the context budget must go to SpatialData regardless of query wording."""
    store = MagicMock()
    primary = MagicMock()
    primary.invoke.return_value = [Document(page_content="spatialdata chunk")]
    secondary = MagicMock()
    secondary.invoke.return_value = [Document(page_content="squidpy chunk")]
    store.as_retriever.side_effect = [primary, secondary]

    docs = RepoWeightedRetriever(store).invoke("how do I read a zarr store")

    primary_kwargs = store.as_retriever.call_args_list[0].kwargs["search_kwargs"]
    secondary_kwargs = store.as_retriever.call_args_list[1].kwargs["search_kwargs"]
    assert primary_kwargs["filter"] == _repo_filter("spatialdata", RETRIEVED_DOC_TYPES)
    assert secondary_kwargs["filter"] == _repo_filter("squidpy", RETRIEVED_DOC_TYPES)
    assert primary_kwargs["k"] > secondary_kwargs["k"]
    assert [doc.page_content for doc in docs] == ["spatialdata chunk", "squidpy chunk"]


def test_repo_filter_uses_compound_form_and_excludes_tests():
    """Chroma needs $and once two metadata fields are involved."""
    where = _repo_filter("spatialdata", ("code", "docs"))
    assert where == {
        "$and": [
            {"source_repo": {"$eq": "spatialdata"}},
            {"doc_type": {"$in": ["code", "docs"]}},
        ]
    }
    assert "test" not in RETRIEVED_DOC_TYPES


def test_doc_type_for_source_separates_tests_from_library_code():
    assert _doc_type_for_source(r"packages_available\spatialdata\src\spatialdata\_io\io_zarr.py") == "code"
    assert _doc_type_for_source(r"packages_available\spatialdata\tests\core\test_query.py") == "test"


def test_index_fingerprint_records_what_invalidates_the_index():
    fingerprint = _index_fingerprint({"spatialdata": "abc123"}, {"spatialdata": "v0.7.3"})
    assert fingerprint["commits"] == {"spatialdata": "abc123"}
    assert fingerprint["refs"] == {"spatialdata": "v0.7.3"}
    assert fingerprint["chunking"] == {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    assert "embedding_model" in fingerprint
    assert "spatialdata" in fingerprint["installed"]


def test_stale_repos_rebuilds_everything_without_a_manifest():
    assert _stale_repos({}, _fingerprint(), _REPOS) == list(_REPOS)


def test_stale_repos_only_rebuilds_the_repo_that_moved():
    """Re-embedding is the expensive step, so an untouched repo must be left alone."""
    current = _fingerprint(commits={"squidpy": "aaaaaaaaaaaa", "spatialdata": "cccccccccccc"})
    assert _stale_repos(_fingerprint(), current, _REPOS) == ["spatialdata"]


def test_stale_repos_detects_a_repin_even_at_the_same_commit():
    current = _fingerprint(refs={"squidpy": "", "spatialdata": "v0.7.2"})
    assert _stale_repos(_fingerprint(), current, _REPOS) == ["spatialdata"]


def test_stale_repos_rebuilds_all_when_every_vector_would_differ():
    previous = _fingerprint()
    assert _stale_repos(previous, _fingerprint(embedding_model="other-model"), _REPOS) == list(_REPOS)
    assert _stale_repos(
        previous,
        _fingerprint(chunking={"chunk_size": 500, "chunk_overlap": 0}),
        _REPOS,
    ) == list(_REPOS)


def test_stale_repos_ignores_installed_version_changes():
    """Chunks come from the checkout, so upgrading a wheel alone does not invalidate them."""
    current = _fingerprint(installed={"squidpy": "1.6.5", "spatialdata": "0.8.0"})
    assert _stale_repos(_fingerprint(), current, _REPOS) == []


@patch("squidpy_rag.get_tool_config")
def test_build_index_never_still_creates_a_missing_index(mock_get_tool_config):
    """Suppressing refreshes must not leave a brand new index empty."""
    mock_get_tool_config.return_value = _mock_tool_config(rag_index_refresh="never")
    with _stubbed_index_build(store_exists=False) as mock_reindex:
        build_combined_index()
    assert {call.args[1] for call in mock_reindex.call_args_list} == {"squidpy", "spatialdata"}


@patch("squidpy_rag.get_tool_config")
def test_build_index_never_leaves_an_existing_index_alone(mock_get_tool_config):
    mock_get_tool_config.return_value = _mock_tool_config(rag_index_refresh="never")
    with _stubbed_index_build(store_exists=True) as mock_reindex:
        build_combined_index()
    mock_reindex.assert_not_called()


@patch("squidpy_rag.get_tool_config")
def test_build_index_force_reembeds_everything(mock_get_tool_config):
    mock_get_tool_config.return_value = _mock_tool_config()
    with _stubbed_index_build(store_exists=True) as mock_reindex:
        build_combined_index(refresh="force")
    assert {call.args[1] for call in mock_reindex.call_args_list} == {"squidpy", "spatialdata"}


def test_format_api_index_lists_the_installed_surface():
    rendered = _format_api_index()
    assert "0.7.3" in rendered
    assert "sd.read_zarr" in rendered
    assert "sd.bounding_box_query" in rendered
    assert "sd.models" in rendered


def test_format_api_index_is_empty_when_the_surface_is_unknown():
    with patch("squidpy_rag._installed_api_surface", return_value=_EMPTY_API_SURFACE):
        assert _format_api_index() == ""


def test_unknown_api_symbols_flags_calls_the_package_does_not_have():
    assert _unknown_api_symbols("sd.read_zarr(DATA_PATH)") == []
    assert _unknown_api_symbols("sd.models.Image2DModel.parse(arr)") == []
    assert _unknown_api_symbols("adata = sd.read_h5ad(DATA_PATH)") == ["read_h5ad"]
    assert _unknown_api_symbols("spatialdata.spatial_neighbors(adata)") == ["spatial_neighbors"]


def test_unknown_api_symbols_stays_silent_when_the_surface_is_unknown():
    """A failed import must never turn into a false rejection."""
    with patch("squidpy_rag._installed_api_surface", return_value=_EMPTY_API_SURFACE):
        assert _unknown_api_symbols("sd.definitely_not_a_real_function()") == []


def test_unknown_api_symbols_does_not_confuse_sdata_for_sd():
    assert _unknown_api_symbols("print(list(sdata.tables))") == []


def test_resolve_question_prefers_original_over_rewrite():
    state = _base_state(
        original_query="How do I read a zarr store?",
        query="how to implement a timeout decorator",
    )
    assert _resolve_question(state) == "How do I read a zarr store?"


def test_prepare_code_injects_data_path_and_load():
    code = "print(list(sdata.tables))"
    zarr_path = r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr"
    prepared = _prepare_code_for_execution(code, zarr_path)

    assert prepared.startswith(f"DATA_PATH = {zarr_path!r}\n")
    assert "import spatialdata as sd" in prepared
    assert "sdata = sd.read_zarr(DATA_PATH)" in prepared
    assert prepared.endswith(code)


def test_prepare_code_injects_h5ad_load():
    prepared = _prepare_code_for_execution("print(adata)", r"C:\data\sample.h5ad")
    assert "adata = sc.read_h5ad(DATA_PATH)" in prepared


def test_prepare_code_without_known_extension_only_injects_path():
    prepared = _prepare_code_for_execution("print(DATA_PATH)", r"C:\data\sample.csv")
    assert prepared.startswith("DATA_PATH = ")
    assert "read_zarr" not in prepared
    assert _prepare_code_for_execution("print('hi')", "") == "print('hi')"


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
def test_generate_prompt_includes_module_paths(
    mock_chat_openai, mock_setup_index, mock_get_tool_config
):
    """The model must be told which module a retrieved symbol lives in."""
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=False)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="def spatial_neighbors(adata): ...",
            metadata={
                "source_repo": "squidpy",
                "source": r"packages_available\squidpy\src\squidpy\gr\_build.py",
            },
        ),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    _, mock_generator = _setup_structured_llm(mock_llm)
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    tool.rag_pipeline.invoke(_base_state())

    rendered = str(mock_generator.invoke.call_args.args[0])
    assert "# from squidpy.gr._build" in rendered


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_generate_prompt_is_spatialdata_first(
    mock_chat_openai, mock_setup_index, mock_get_tool_config
):
    """The prompt must lead with SpatialData and describe the pre-loaded sdata object."""
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=False)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="def read_zarr(path): ...",
            metadata={
                "source_repo": "spatialdata",
                "source": r"packages_available\spatialdata\src\spatialdata\_io\io_zarr.py",
            },
        ),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    _, mock_generator = _setup_structured_llm(mock_llm)
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    tool.rag_pipeline.invoke(_base_state(data_path=r"C:\data\sample.zarr"))

    rendered = str(mock_generator.invoke.call_args.args[0])
    assert "expert in SpatialData" in rendered
    assert "Prefer SpatialData APIs" in rendered
    assert "ALREADY LOADED" in rendered
    assert "sdata.tables" in rendered
    assert "SPATIALDATA GUIDANCE" in rendered


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_generate_prompt_includes_the_installed_api_index(
    mock_chat_openai, mock_setup_index, mock_get_tool_config
):
    """The model gets an authoritative symbol list, not just retrieved chunks."""
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=False)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="def read_zarr(path): ...", metadata={"source_repo": "spatialdata"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_llm = MagicMock()
    _, mock_generator = _setup_structured_llm(mock_llm)
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    tool.rag_pipeline.invoke(_base_state())

    rendered = str(mock_generator.invoke.call_args.args[0])
    assert "SPATIALDATA API INDEX" in rendered
    assert "sd.read_zarr" in rendered


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag._get_python_repl")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_execute_rejects_symbols_missing_from_installed_package(
    mock_chat_openai, mock_setup_index, mock_get_repl, mock_get_tool_config
):
    """Catch a hallucinated API before paying the execution timeout for it."""
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=True)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="neighbors", metadata={"source_repo": "spatialdata"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_repl = MagicMock()
    mock_get_repl.return_value = mock_repl

    mock_llm = MagicMock()
    _setup_structured_llm(
        mock_llm,
        generated=GeneratedAnswer(
            code="print(sd.spatial_neighbors(sdata))",
            explanation="Build a spatial graph",
        ),
    )
    mock_llm.invoke.return_value = MagicMock(content="rewritten query")
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(
        _base_state(data_path=r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr")
    )

    mock_repl.run.assert_not_called()
    assert result["execution_success"] is False
    assert "spatialdata.spatial_neighbors does not exist" in result["execution_output"]


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
            code="print(list(sdata.tables))",
            explanation="List the tables",
        ),
    )
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    zarr_path = r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr"
    result = tool.rag_pipeline.invoke(_base_state(data_path=zarr_path))

    expected_code = _prepare_code_for_execution("print(list(sdata.tables))", zarr_path)
    mock_repl.run.assert_called_once_with(expected_code, timeout=30)
    assert result["execution_success"] is True
    assert result["execution_output"] == "loaded"


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag._get_python_repl")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_execute_rejects_code_ignoring_data_path(
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
            code="def timeout(f, seconds):\n    return f\nprint(timeout)",
            explanation="A timeout decorator",
        ),
    )
    mock_llm.invoke.return_value = MagicMock(content="rewritten query")
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(
        _base_state(data_path=r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr")
    )

    mock_repl.run.assert_not_called()
    assert result["execution_success"] is False
    assert "Execution rejected" in result["execution_output"]


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag._get_python_repl")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_execute_rejects_code_that_reloads_preloaded_data(
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
            code="import squidpy as sq\nsdata = sq.read_zarr(DATA_PATH)\nprint(sdata)",
            explanation="Load the data",
        ),
    )
    mock_llm.invoke.return_value = MagicMock(content="rewritten query")
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    result = tool.rag_pipeline.invoke(
        _base_state(data_path=r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr")
    )

    mock_repl.run.assert_not_called()
    assert result["execution_success"] is False
    assert "Execution rejected" in result["execution_output"]
    assert "already loaded as `sdata`" in result["execution_output"]


@patch("squidpy_rag.get_tool_config")
@patch("squidpy_rag._get_python_repl")
@patch("squidpy_rag.SquidpyRAGTool.setup_combined_index")
@patch("squidpy_rag.ChatOpenAI")
def test_generate_answers_original_question_after_rewrite(
    mock_chat_openai, mock_setup_index, mock_get_repl, mock_get_tool_config
):
    """A failed execution must not let the rewriter change what is being answered."""
    mock_get_tool_config.return_value = _mock_tool_config(rag_exec_enabled=True)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="read zarr", metadata={"source_repo": "spatialdata"}),
    ]
    mock_setup_index.return_value = mock_retriever

    mock_repl = MagicMock()
    mock_repl.run.return_value = "Execution timed out"
    mock_get_repl.return_value = mock_repl

    mock_llm = MagicMock()
    _, mock_generator = _setup_structured_llm(
        mock_llm,
        generated=GeneratedAnswer(
            code="print(list(sdata.tables))",
            explanation="List the tables",
        ),
    )
    mock_llm.invoke.return_value = MagicMock(content="how to implement a timeout decorator")
    mock_chat_openai.return_value = mock_llm

    tool = SquidpyRAGTool()
    original = "Read the zarr SpatialData store and print element names"
    tool.rag_pipeline.invoke(
        _base_state(
            query=original,
            original_query=original,
            data_path=r"C:\Pascal's Folders\QIMR\SCOPE_sample_40.zarr",
        )
    )

    for call in mock_generator.invoke.call_args_list:
        rendered = str(call.args[0])
        assert original in rendered
        assert "timeout decorator" not in rendered


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
