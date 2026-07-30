import argparse
import importlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict, List, Dict, Any, Literal, Sequence
from dotenv import load_dotenv
from git import Repo
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from prompt import spatial_processing_prompt, spatialdata_processing_prompt
from config import get_data_path, get_tool_config

load_dotenv(Path(__file__).resolve().with_name(".env"))

SQUIDPY_REPO_PATH = "./packages_available/squidpy"
SPATIALDATA_REPO_PATH = "./packages_available/spatialdata"
COMBINED_PERSIST_DIR = "./db/chroma_combined_db"
INDEX_MANIFEST_PATH = f"{COMBINED_PERSIST_DIR}/index_manifest.json"

SQUIDPY_REPO_URL = "https://github.com/scverse/squidpy"
SPATIALDATA_REPO_URL = "https://github.com/scverse/spatialdata"

# SpatialData is the target API, so most of the context budget goes to it. A couple of
# Squidpy chunks stay reachable because SpatialData has no spatial-statistics equivalents.
PRIMARY_REPO = "spatialdata"
PRIMARY_K = 6
SECONDARY_REPO = "squidpy"
SECONDARY_K = 2

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Test files are indexed so they stay searchable, but they are kept out of retrieval:
# they are a large share of all chunks and their fixtures encourage the model to build
# synthetic objects instead of using the dataset it was given.
RETRIEVED_DOC_TYPES = ("code", "docs")

_DATA_HINTS = ("read_h5ad", "read_zarr", "SpatialData.read", "sq.read.", "adata", "sdata")

_python_repl = None


class SquidpyRAGState(TypedDict):
    query: str
    original_query: str
    previous_queries: List[str]
    context: List[Document]
    filtered_context: List[Document]
    answer: str
    generated_code: str
    execution_output: str
    execution_success: bool
    chat_history: List[AnyMessage]
    rewrite_attempts: int
    data_path: str


class GradeDoc(BaseModel):
    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the query, 'no' otherwise."
    )


class GeneratedAnswer(BaseModel):
    code: str = Field(
        description="Executable Python code only. No markdown fences. No prose."
    )
    explanation: str = Field(
        description="Concise explanation of the code and SpatialData/Squidpy concepts used."
    )


def _max_rewrite_attempts() -> int:
    return get_tool_config().rag_max_rewrite_attempts


def _resolve_question(state: "SquidpyRAGState") -> str:
    """The user's actual question, which query rewrites must never replace."""
    return state.get("original_query") or state["query"]


def decide_relevance(state: SquidpyRAGState) -> Literal["generate", "rewrite_query"]:
    if state["filtered_context"] or state["rewrite_attempts"] >= _max_rewrite_attempts():
        return "generate"
    return "rewrite_query"


def decide_execution(state: SquidpyRAGState) -> Literal["end", "rewrite_query"]:
    if state["execution_success"]:
        return "end"
    if state["rewrite_attempts"] >= _max_rewrite_attempts():
        return "end"
    return "rewrite_query"


def _execution_failed(output: str) -> bool:
    """Return True if PythonREPL output indicates an error."""
    if not output or not output.strip():
        return False
    failure_markers = (
        "Execution timed out",
        "External execution error",
        "exit_code=",
        "Execution skipped:",
        "Execution rejected:",
        "No code was generated",
    )
    if output.startswith(
        (
            "SyntaxError",
            "NameError",
            "TypeError",
            "ValueError",
            "KeyError",
            "ImportError",
            "ModuleNotFoundError",
            "FileNotFoundError",
            "AttributeError",
            "RuntimeError",
        )
    ):
        return True
    return any(marker in output for marker in failure_markers)


def _code_requires_data(code: str) -> bool:
    return any(hint in code for hint in _DATA_HINTS)


# The execution preamble loads the dataset itself, so the model only has to use the
# resulting variable. Maps file extension -> (variable name, load statements).
_PRELOAD_TEMPLATES = {
    ".zarr": ("sdata", "import spatialdata as sd\nsdata = sd.read_zarr(DATA_PATH)"),
    ".h5ad": ("adata", "import scanpy as sc\nadata = sc.read_h5ad(DATA_PATH)"),
}

_LOADER_NAMES = ("read_zarr", "read_h5ad")

# Attributing a loader to the wrong package was the most common generation failure, so
# each loader is pinned to the packages that actually own it.
_LOADER_OWNERS = {
    "read_zarr": {"spatialdata"},
    "read_h5ad": {"scanpy", "anndata"},
}

_PACKAGE_ALIASES = {
    "sq.": "squidpy",
    "squidpy.": "squidpy",
    "sd.": "spatialdata",
    "spatialdata.": "spatialdata",
    "sc.": "scanpy",
    "scanpy.": "scanpy",
    "ad.": "anndata",
    "anndata.": "anndata",
}


def _preloaded_binding(data_path: str) -> tuple[str, str] | None:
    """Variable name and load statements the preamble injects for this file type."""
    if not data_path:
        return None
    return _PRELOAD_TEMPLATES.get(Path(data_path).suffix.lower())


def _data_access_hint(data_path: str) -> str:
    """Describe the already-loaded object so the model never writes a load call."""
    suffix = Path(data_path).suffix.lower()
    if suffix == ".zarr":
        return (
            "The dataset is ALREADY LOADED for you as `sdata`, a spatialdata.SpatialData "
            "object read from the .zarr store.\n"
            "Do NOT import spatialdata, do NOT call read_zarr, do NOT reference DATA_PATH. "
            "Start your code by using `sdata` directly.\n"
            "`sdata` is a container, not an AnnData: its elements live in sdata.images, "
            "sdata.labels, sdata.points, sdata.shapes and sdata.tables.\n"
            "For any Squidpy call, first pull out the AnnData table, e.g. "
            "`adata = sdata.tables[next(iter(sdata.tables))]`."
        )
    if suffix == ".h5ad":
        return (
            "The dataset is ALREADY LOADED for you as `adata`, an AnnData object read from "
            "the .h5ad file.\n"
            "Do NOT call read_h5ad, do NOT reference DATA_PATH. Start your code by using "
            "`adata` directly."
        )
    return (
        "The file path is available as the variable DATA_PATH. Use DATA_PATH only; never "
        "hardcode a path string. A .zarr store is read with spatialdata.read_zarr and a "
        ".h5ad file with scanpy.read_h5ad."
    )


def _wrong_loader_reason(code: str, data_path: str) -> str | None:
    """Return why the code's data loading is wrong, or None if it is acceptable."""
    binding = _preloaded_binding(data_path)
    if binding and any(loader in code for loader in _LOADER_NAMES):
        variable = binding[0]
        return (
            f"the dataset is already loaded as `{variable}`; the code must use `{variable}` "
            "directly instead of calling read_zarr or read_h5ad again."
        )

    for loader, owners in _LOADER_OWNERS.items():
        if loader not in code:
            continue
        for prefix, package in _PACKAGE_ALIASES.items():
            if f"{prefix}{loader}" in code and package not in owners:
                return (
                    f"{package} has no {loader} function; {loader} belongs to "
                    f"{' or '.join(sorted(owners))}."
                )

    suffix = Path(data_path).suffix.lower()
    if suffix == ".zarr" and "read_h5ad" in code:
        return "DATA_PATH is a .zarr store but the code calls read_h5ad."
    if suffix == ".h5ad" and "read_zarr" in code:
        return "DATA_PATH is a .h5ad file but the code calls read_zarr."
    return None


def _prepare_code_for_execution(code: str, data_path: str) -> str:
    """Prepend a trusted DATA_PATH and, when the type is known, load the data too.

    Loading here rather than in the generated code removes the two failure modes the
    model kept hitting: mangling Windows paths, and attributing read_zarr to squidpy.
    """
    if not data_path:
        return code
    preamble = f"DATA_PATH = {data_path!r}\n"
    binding = _preloaded_binding(data_path)
    if binding:
        preamble += f"{binding[1]}\n"
    return f"{preamble}{code}"


def _get_python_repl():
    global _python_repl
    if _python_repl is None:
        from tools import PythonREPL

        _python_repl = PythonREPL()
    return _python_repl


def _get_lm_studio_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("LM_STUDIO_EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        check_embedding_ctx_length=False,
    )


def _sync_repo(repo_path: str, url: str, ref: str) -> str:
    """Clone or update a vendored repo, returning the short commit sha it now sits on.

    Pinning to a tag rather than tracking a branch is what keeps the index reproducible
    and lets it be matched against the installed wheel.
    """
    if not os.path.exists(repo_path):
        print(f"Cloning {url} into {repo_path}...")
        Repo.clone_from(url, to_path=repo_path)

    try:
        repo = Repo(repo_path)
    except Exception as exc:
        print(f"Warning: {repo_path} is not a git checkout ({exc}); indexing it as-is.")
        return "unknown"

    if ref:
        try:
            repo.remotes.origin.fetch(tags=True, prune=True)
            repo.git.checkout(ref)
            print(f"{repo_path} checked out at {ref}")
        except Exception as exc:
            print(
                f"Warning: could not check out {ref} in {repo_path} ({exc}); "
                "indexing whatever is currently checked out."
            )

    try:
        return repo.head.commit.hexsha[:12]
    except Exception:
        return "unknown"


def _doc_type_for_source(source: str) -> Literal["code", "test"]:
    parts = [part.lower() for part in Path(source).parts]
    if "tests" in parts or "test" in parts:
        return "test"
    return "code"


def _load_python_chunks(repo_path: str, source_repo: str) -> List[Document]:
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source_repo"] = source_repo
        chunk.metadata["doc_type"] = _doc_type_for_source(
            str(chunk.metadata.get("source") or "")
        )
    return chunks


def _load_markdown_chunks(repo_path: str, source_repo: str) -> List[Document]:
    """Index the prose docs alongside the source.

    The API reference pages describe what a function is *for*, which the implementation
    never states, and they enumerate the public surface rather than internals.
    """
    root = Path(repo_path)
    paths = sorted(root.glob("docs/**/*.md")) + sorted(root.glob("*.md"))

    documents: List[Document] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not text.strip():
            continue
        documents.append(Document(page_content=text, metadata={"source": str(path)}))

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source_repo"] = source_repo
        chunk.metadata["doc_type"] = "docs"
    return chunks


def _load_repo_chunks(repo_path: str, source_repo: str) -> List[Document]:
    code_chunks = _load_python_chunks(repo_path, source_repo)
    doc_chunks = _load_markdown_chunks(repo_path, source_repo)
    counts: Dict[str, int] = {}
    for chunk in code_chunks + doc_chunks:
        doc_type = str(chunk.metadata.get("doc_type"))
        counts[doc_type] = counts.get(doc_type, 0) + 1
    summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
    print(f"Loaded {len(code_chunks) + len(doc_chunks)} chunks from {source_repo} ({summary})")
    return code_chunks + doc_chunks


def _module_label(metadata: Dict[str, Any]) -> str:
    """Turn a chunk's file path into the module path used to import its symbols.

    Without this the model only sees the repo name and has to guess whether a
    function lives in e.g. squidpy.gr or squidpy.tl.
    """
    repo = str(metadata.get("source_repo") or "unknown")
    source = str(metadata.get("source") or "").replace("\\", "/")
    if not source:
        return repo

    parts = source.split("/")
    if "src" in parts:
        parts = parts[parts.index("src") + 1 :]
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][: -len(".py")]
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                return ".".join(parts)
    return f"{repo}: {source}"


def _installed_version(package: str) -> str:
    """Version of an installed distribution without importing it."""
    try:
        from importlib.metadata import version

        return str(version(package))
    except Exception:
        return ""


@lru_cache(maxsize=4)
def _installed_api_surface(package: str = "spatialdata") -> Dict[str, Any]:
    """Public symbols of the *installed* package.

    Read from the installed package rather than the vendored checkout so the prompt can
    never advertise an API that will not be there when the code actually runs.
    """
    empty = {"version": "", "symbols": (), "submodules": ()}
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        print(f"Warning: could not import {package} to read its API surface ({exc}).")
        return empty

    # spatialdata lists its whole public surface in _LAZY_IMPORTS; older releases and
    # other packages only expose __all__.
    lazy = getattr(module, "_LAZY_IMPORTS", None)
    if isinstance(lazy, dict) and lazy:
        symbols = tuple(sorted(str(name) for name in lazy))
    else:
        symbols = tuple(sorted(str(name) for name in (getattr(module, "__all__", None) or ())))

    submodules = getattr(module, "_submodules", None)
    if not isinstance(submodules, (set, frozenset, tuple, list)):
        submodules = ()

    return {
        "version": str(getattr(module, "__version__", "") or _installed_version(package)),
        "symbols": symbols,
        "submodules": tuple(sorted(str(name) for name in submodules)),
    }


def _format_api_index(package: str = "spatialdata", alias: str = "sd") -> str:
    """Render the installed public API as an authoritative list for the prompt."""
    surface = _installed_api_surface(package)
    symbols = surface["symbols"]
    if not symbols:
        return ""

    version = surface["version"] or "unknown version"
    lines = [
        f"These are the ONLY top-level names in the installed {package} {version}. "
        f"Call them as {alias}.<name>. Anything not in this list does not exist and "
        "will raise AttributeError:",
        ", ".join(f"{alias}.{name}" for name in symbols),
    ]
    if surface["submodules"]:
        lines.append(
            "Submodules: "
            + ", ".join(f"{alias}.{name}" for name in surface["submodules"])
        )
    return "\n".join(lines)


_TOP_LEVEL_ATTRIBUTE_PATTERN = re.compile(r"\b(?:sd|spatialdata)\.([A-Za-z_][A-Za-z0-9_]*)")


def _unknown_api_symbols(code: str, package: str = "spatialdata") -> List[str]:
    """Top-level attributes the code uses that the installed package does not define.

    Returns nothing when the surface could not be determined, so a failed import can
    never cause a false rejection.
    """
    surface = _installed_api_surface(package)
    known = set(surface["symbols"]) | set(surface["submodules"])
    if not known:
        return []

    used = {match.group(1) for match in _TOP_LEVEL_ATTRIBUTE_PATTERN.finditer(code)}
    return sorted(
        name for name in used if name not in known and not name.startswith("__")
    )


def _repo_filter(repo: str, doc_types: Sequence[str]) -> Dict[str, Any]:
    """Chroma requires the compound form once more than one metadata field is involved."""
    return {
        "$and": [
            {"source_repo": {"$eq": repo}},
            {"doc_type": {"$in": list(doc_types)}},
        ]
    }


class RepoWeightedRetriever:
    """Retriever that fills most of the context with one repo before topping up with another.

    An unweighted search over the combined index returns whichever chunks are closest in
    embedding space, which for spatial-analysis wording is usually Squidpy. Splitting the
    search by `source_repo` guarantees SpatialData dominates the context regardless of
    how the question is phrased.
    """

    def __init__(
        self,
        store,
        primary_repo: str = PRIMARY_REPO,
        primary_k: int = PRIMARY_K,
        secondary_repo: str = SECONDARY_REPO,
        secondary_k: int = SECONDARY_K,
        doc_types: Sequence[str] = RETRIEVED_DOC_TYPES,
    ):
        self._primary = store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": primary_k,
                "filter": _repo_filter(primary_repo, doc_types),
            },
        )
        self._secondary = store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": secondary_k,
                "filter": _repo_filter(secondary_repo, doc_types),
            },
        )

    def invoke(self, query: str) -> List[Document]:
        return list(self._primary.invoke(query)) + list(self._secondary.invoke(query))


REPO_PATHS = {"squidpy": SQUIDPY_REPO_PATH, "spatialdata": SPATIALDATA_REPO_PATH}
REPO_URLS = {"squidpy": SQUIDPY_REPO_URL, "spatialdata": SPATIALDATA_REPO_URL}

INDEX_SCHEMA_VERSION = 2
_EMBED_BATCH_SIZE = 200
_DELETE_BATCH_SIZE = 500


def _index_fingerprint(commits: Dict[str, str], refs: Dict[str, str]) -> Dict[str, Any]:
    """Everything that, if changed, means the stored embeddings no longer describe the source."""
    return {
        "schema": INDEX_SCHEMA_VERSION,
        "refs": dict(refs),
        "commits": dict(commits),
        "chunking": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
        "embedding_model": os.getenv("LM_STUDIO_EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
        "installed": {repo: _installed_version(repo) for repo in refs},
    }


def _read_index_manifest() -> Dict[str, Any]:
    try:
        with open(INDEX_MANIFEST_PATH, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_index_manifest(fingerprint: Dict[str, Any]) -> None:
    os.makedirs(COMBINED_PERSIST_DIR, exist_ok=True)
    with open(INDEX_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(fingerprint, handle, indent=2, sort_keys=True)


def _stale_repos(
    previous: Dict[str, Any], current: Dict[str, Any], repos: Sequence[str]
) -> List[str]:
    """Which repos need re-embedding.

    A change to chunking or the embedding model invalidates every vector, so those force
    a full rebuild. Otherwise only the repo whose commit moved is redone, because
    re-embedding through LM Studio costs far more than the checkout does.
    """
    if not previous:
        return list(repos)
    for key in ("schema", "chunking", "embedding_model"):
        if previous.get(key) != current.get(key):
            return list(repos)

    stale: List[str] = []
    for repo in repos:
        moved = previous.get("commits", {}).get(repo) != current.get("commits", {}).get(repo)
        repinned = previous.get("refs", {}).get(repo) != current.get("refs", {}).get(repo)
        if moved or repinned:
            stale.append(repo)
    return stale


def _warn_on_version_skew(fingerprint: Dict[str, Any]) -> None:
    """Warn when the indexed source and the installed wheel are different versions.

    This is the failure that prompt tuning cannot fix: the model is taught an API that
    the interpreter does not have, so every rewrite attempt is spent on a call that can
    never resolve.
    """
    for repo, ref in fingerprint.get("refs", {}).items():
        installed = str(fingerprint.get("installed", {}).get(repo) or "")
        expected = str(ref or "").lstrip("vV")
        if not expected or not installed:
            continue
        if not installed.startswith(expected):
            print(
                f"Warning: indexing {repo} source at {ref} but {repo} {installed} is "
                f"installed. Generated code may call APIs that do not exist at runtime. "
                f'Fix with: pip install "{repo}=={expected}"'
            )


def _add_documents_in_batches(store, chunks: List[Document]) -> None:
    """Embed in batches to stay under Chroma's batch ceiling and to show progress."""
    total = len(chunks)
    for start in range(0, total, _EMBED_BATCH_SIZE):
        batch = chunks[start : start + _EMBED_BATCH_SIZE]
        store.add_documents(batch)
        print(f"  embedded {min(start + len(batch), total)}/{total}")


def _reindex_repo(store, source_repo: str, repo_path: str) -> None:
    """Replace one repo's chunks in place, leaving the other repo's embeddings untouched."""
    try:
        existing = store.get(where={"source_repo": source_repo})
        ids = existing.get("ids") or []
    except Exception as exc:
        print(f"Warning: could not list existing {source_repo} chunks ({exc}).")
        ids = []

    if ids:
        print(f"Removing {len(ids)} stale {source_repo} chunks...")
        # Batched because a single delete of thousands of ids can exceed the SQLite
        # bound-variable limit underneath Chroma.
        for start in range(0, len(ids), _DELETE_BATCH_SIZE):
            store.delete(ids=ids[start : start + _DELETE_BATCH_SIZE])

    chunks = _load_repo_chunks(repo_path, source_repo)
    if not chunks:
        print(f"Warning: no chunks found for {source_repo} at {repo_path}.")
        return

    print(f"Embedding {len(chunks)} {source_repo} chunks (this is the slow step)...")
    _add_documents_in_batches(store, chunks)


def _print_index_summary(store) -> None:
    try:
        records = store.get(include=["metadatas"])
    except Exception as exc:
        print(f"Could not summarise the index ({exc}).")
        return

    counts: Dict[str, int] = {}
    for metadata in records.get("metadatas") or []:
        metadata = metadata or {}
        key = f"{metadata.get('source_repo', '?')}/{metadata.get('doc_type', '?')}"
        counts[key] = counts.get(key, 0) + 1

    print(f"Index contains {sum(counts.values())} chunks:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def build_combined_index(refresh: str | None = None) -> RepoWeightedRetriever:
    """Sync the vendored repos, refresh the index if stale, and return a retriever."""
    tool_config = get_tool_config()
    refresh = refresh or tool_config.rag_index_refresh
    refs = {
        "squidpy": tool_config.rag_squidpy_ref,
        "spatialdata": tool_config.rag_spatialdata_ref,
    }

    # "never" must not touch the network on every app start, so skip fetch/checkout and
    # just report whatever the clone already sits on.
    sync_refs = refresh != "never"
    commits = {
        repo: _sync_repo(path, REPO_URLS[repo], refs[repo] if sync_refs else "")
        for repo, path in REPO_PATHS.items()
    }

    fingerprint = _index_fingerprint(commits, refs)
    _warn_on_version_skew(fingerprint)

    store_exists = os.path.exists(COMBINED_PERSIST_DIR)
    previous = _read_index_manifest() if store_exists else {}

    if refresh == "force" or not store_exists:
        stale = list(REPO_PATHS)
    else:
        stale = _stale_repos(previous, fingerprint, list(REPO_PATHS))

    # "never" suppresses refreshes of an index that exists; it must not leave a missing
    # index empty, which would make every retrieval return nothing.
    if stale and refresh == "never" and store_exists:
        print(
            f"Warning: the index is stale for {', '.join(stale)} but "
            "RAG_INDEX_REFRESH=never, so the existing index will be served as-is."
        )
        stale = []

    store = Chroma(
        persist_directory=COMBINED_PERSIST_DIR,
        embedding_function=_get_lm_studio_embeddings(),
    )

    if stale:
        print(f"Refreshing index for: {', '.join(stale)}")
        for repo in stale:
            _reindex_repo(store, repo, REPO_PATHS[repo])
        _write_index_manifest(fingerprint)
        _print_index_summary(store)
    else:
        print(f"Index at {COMBINED_PERSIST_DIR} is up to date.")

    return RepoWeightedRetriever(store)


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"# from {_module_label(doc.metadata)}\n{doc.page_content}" for doc in docs
    )


def _format_run_response(response: Dict[str, Any]) -> str:
    parts = [response.get("answer") or ""]
    generated_code = response.get("generated_code", "")
    execution_output = response.get("execution_output", "")

    if generated_code:
        parts.append(f"\n\n--- Generated code ---\n{generated_code}")
    if execution_output:
        label = "Execution output" if response.get("execution_success") else "Execution error"
        parts.append(f"\n\n--- {label} ---\n{execution_output}")
    return "".join(parts)


class SquidpyRAGTool:
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
    ):
        self.model = model or os.getenv("LM_STUDIO_MODEL", "gemma4-e2b")
        self.base_url = base_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.api_key = api_key or os.getenv("LM_STUDIO_API_KEY", "lm-studio")

        self.retriever = self.setup_combined_index()
        self.rag_pipeline = self.create_squidpy_rag_pipeline()

    def setup_combined_index(self):
        """Setup and index Squidpy + SpatialData repositories into a single Chroma DB."""
        return build_combined_index()

    def create_squidpy_rag_pipeline(self):
        """Create the self-reflective RAG pipeline for Squidpy and SpatialData."""
        llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0,
        )
        grader_llm = llm.with_structured_output(GradeDoc)
        generator_llm = llm.with_structured_output(GeneratedAnswer)
        rag_exec_enabled = get_tool_config().rag_exec_enabled

        rewrite_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a query rewriter for a SpatialData and Squidpy code RAG system. "
                "Rewrite the user's query to be more specific and retrievable from a Python "
                "codebase index, favouring SpatialData terminology (elements, tables, "
                "coordinate systems, transformations, queries) over Squidpy terminology "
                "unless the query is specifically about spatial statistics or plotting. "
                "Keep the same topic, dataset and intent as the original query: never "
                "substitute a different subject. Do not answer the query. Output only the "
                "rewritten query, nothing else.",
            ),
            (
                "user",
                "Original query: {query}\n\n{previous_attempts}Rewritten query:",
            ),
        ])
        rewriter = rewrite_prompt | llm

        def retrieve(state: SquidpyRAGState):
            retrieved_docs = self.retriever.invoke(state["query"])
            return {"context": retrieved_docs}

        def grade_documents(state: SquidpyRAGState):
            filtered: List[Document] = []
            for doc in state["context"]:
                grade_prompt = (
                    f"Query: {state['query']}\n\n"
                    f"Document:\n{doc.page_content[:1500]}\n\n"
                    "Is this document relevant to answering the query?"
                )
                result = grader_llm.invoke(grade_prompt)
                if result.score == "yes":
                    filtered.append(doc)
            return {"filtered_context": filtered}

        def generate(state: SquidpyRAGState):
            docs = state["filtered_context"] or state["context"]
            context_content = _format_context(docs)
            chat_history = state["chat_history"]
            data_path = state.get("data_path", "")

            error_feedback = ""
            if state.get("execution_output") and not state.get("execution_success", True):
                error_feedback = (
                    "A previous attempt at this same question failed when executed:\n"
                    f"{state['execution_output']}\n"
                    "Fix the cause of that failure. Keep answering the same question; "
                    "do not switch topic to the error itself.\n\n"
                )

            data_path_instruction = (
                "No data path was provided. Keep the code example minimal and self-contained."
            )
            if data_path:
                data_path_instruction = (
                    "DATA ACCESS:\n"
                    f"{_data_access_hint(data_path)}"
                )

            prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder("chat_history"),
                (
                    "user",
                    "The above are the CHAT HISTORY between the user and the spatial "
                    "transcriptomics assistant. Take the chat history into account when "
                    "generating the response.",
                ),
                (
                    "user",
                    "You are an expert in SpatialData, specializing in providing authentic "
                    "Python code and explanations on its usage. SpatialData is your default "
                    "toolkit: use it for reading, inspecting, subsetting, transforming and "
                    "aggregating spatial omics data. Reach for Squidpy only for spatial "
                    "statistics and plotting that SpatialData does not provide, and run those "
                    "on an AnnData table taken out of the SpatialData object.\n"
                    "Return structured output with two fields:\n"
                    "- code: executable Python only (no markdown fences, no prose)\n"
                    "- explanation: concise explanation of the code\n\n"
                    "REMEMBER to specify shape = None for STARmap spatial transcriptomic data.\n"
                    "The code must be runnable as-is in a Python REPL.\n\n"
                    "CODE RULES:\n"
                    "- Answer only the user question; never substitute an unrelated example.\n"
                    "- Prefer SpatialData APIs. Do not use a Squidpy function when a "
                    "SpatialData one does the job.\n"
                    "- Do NOT write timeout, retry, signal, threading or multiprocessing "
                    "scaffolding; the execution harness already enforces a timeout.\n"
                    "- Call every function you define; never leave the result as an "
                    "uncalled function object.\n"
                    "- Print the values the user asked for.\n"
                    "- adata.obs_names holds CELL barcodes, never sample or group IDs. "
                    "To group by sample, print adata.obs.columns first and use a real "
                    "column such as adata.obs['sample'].\n"
                    "- Never loop over every cell or every row. Operate on the whole "
                    "object, or over a small number of groups.\n"
                    "- Do not assume a column like 'cell_type' exists; check "
                    "adata.obs.columns before using it.\n"
                    "- Let exceptions propagate. Do NOT wrap the analysis in try/except "
                    "blocks that only print the error.\n"
                    "- If you create matplotlib figures in a loop, close each one with "
                    "plt.close() to avoid exhausting memory.\n"
                    "- Import the top-level package and call symbols fully qualified, e.g. "
                    "`import spatialdata as sd` then `sd.bounding_box_query(...)`. Do NOT "
                    "write `from spatialdata._core.query import bounding_box_query` style "
                    "imports.\n"
                    "- Keep package ownership straight: read_zarr and the query, transform "
                    "and aggregate helpers belong to `spatialdata`; spatial_neighbors, "
                    "nhood_enrichment and the plotting helpers belong to `squidpy`. Calling "
                    "one on the other package raises AttributeError.\n"
                    "- Every context block below starts with `# from <module>` giving the "
                    "real module path of that code. Use those paths to decide where a "
                    "function lives; never guess a submodule name.\n\n"
                    "{api_index}\n\n"
                    "{data_path_instruction}\n\n"
                    "{error_feedback}"
                    "SPATIALDATA GUIDANCE:\n{spatialdata_processing_prompt}\n\n"
                    "SQUIDPY GUIDANCE (only for analysis and plotting steps):\n"
                    "{spatial_processing_prompt}\n\n"
                    "CONTEXT FROM CODEBASE:\n{context_content}\n\n",
                ),
                ("user", "USER QUESTION: {query}"),
            ])

            api_index = _format_api_index()
            if api_index:
                api_index = f"SPATIALDATA API INDEX:\n{api_index}"

            messages = prompt.invoke({
                "query": _resolve_question(state),
                "chat_history": chat_history,
                "api_index": api_index,
                "context_content": context_content,
                "spatialdata_processing_prompt": spatialdata_processing_prompt,
                "spatial_processing_prompt": spatial_processing_prompt,
                "data_path_instruction": data_path_instruction,
                "error_feedback": error_feedback,
            })
            result = generator_llm.invoke(messages)
            return {
                "answer": result.explanation,
                "generated_code": result.code,
            }

        def execute(state: SquidpyRAGState):
            code = state.get("generated_code", "").strip()
            if not code:
                return {
                    "execution_output": "No code was generated to execute.",
                    "execution_success": False,
                }

            data_path = state.get("data_path", "")

            if not data_path and _code_requires_data(code):
                return {
                    "execution_output": (
                        "Execution skipped: generated code references data files but no "
                        "data_path was provided."
                    ),
                    "execution_success": False,
                }

            binding = _preloaded_binding(data_path)
            expected_symbol = binding[0] if binding else "DATA_PATH"
            if data_path and expected_symbol not in code:
                return {
                    "execution_output": (
                        f"Execution rejected: a dataset was provided as `{expected_symbol}` but "
                        f"the generated code never uses `{expected_symbol}`, so it does not "
                        "answer the question about the dataset."
                    ),
                    "execution_success": False,
                }

            loader_problem = _wrong_loader_reason(code, data_path)
            if loader_problem:
                return {
                    "execution_output": f"Execution rejected: {loader_problem}",
                    "execution_success": False,
                }

            unknown_symbols = _unknown_api_symbols(code)
            if unknown_symbols:
                named = ", ".join(f"spatialdata.{name}" for name in unknown_symbols)
                return {
                    "execution_output": (
                        f"Execution rejected: {named} does not exist in the installed "
                        "spatialdata. Use only the names listed in the SPATIALDATA API "
                        "INDEX."
                    ),
                    "execution_success": False,
                }

            tool_config = get_tool_config()
            timeout = tool_config.rag_exec_timeout
            executable_code = _prepare_code_for_execution(code, data_path)
            output = _get_python_repl().run(executable_code, timeout=timeout)
            success = not _execution_failed(output)

            return {
                "execution_output": output[: tool_config.max_output_length],
                "execution_success": success,
            }

        def rewrite_query(state: SquidpyRAGState):
            previous = state.get("previous_queries") or []
            previous_attempts = ""
            if previous:
                tried = "\n".join(f"- {q}" for q in previous)
                previous_attempts = (
                    "These rewrites were already tried and did not help:\n"
                    f"{tried}\n\nProduce a different rewrite of the original query.\n\n"
                )
            response = rewriter.invoke({
                "query": _resolve_question(state),
                "previous_attempts": previous_attempts,
            })
            rewritten = response.content.strip()
            return {
                "query": rewritten,
                "previous_queries": previous + [rewritten],
                "rewrite_attempts": state["rewrite_attempts"] + 1,
            }

        graph_builder = StateGraph(SquidpyRAGState)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("grade_documents", grade_documents)
        graph_builder.add_node("generate", generate)
        graph_builder.add_node("rewrite_query", rewrite_query)

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "grade_documents")
        graph_builder.add_conditional_edges(
            "grade_documents",
            decide_relevance,
            {"generate": "generate", "rewrite_query": "rewrite_query"},
        )
        graph_builder.add_edge("rewrite_query", "retrieve")

        if rag_exec_enabled:
            graph_builder.add_node("execute", execute)
            graph_builder.add_edge("generate", "execute")
            graph_builder.add_conditional_edges(
                "execute",
                decide_execution,
                {"end": END, "rewrite_query": "rewrite_query"},
            )
        else:
            graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def run(
        self,
        query: str,
        chat_history: List[AnyMessage] = None,
        data_path: str = None,
    ) -> str:
        """Run the SpatialData/Squidpy RAG pipeline with the given query and chat history."""
        if chat_history is None:
            chat_history = []

        resolved_data_path = data_path or get_data_path() or ""

        response = self.rag_pipeline.invoke({
            "query": query,
            "original_query": query,
            "previous_queries": [],
            "chat_history": chat_history,
            "context": [],
            "filtered_context": [],
            "answer": "",
            "generated_code": "",
            "execution_output": "",
            "execution_success": False,
            "rewrite_attempts": 0,
            "data_path": resolved_data_path,
        })

        return _format_run_response(response)


_squidpy_rag: SquidpyRAGTool | None = None


def _get_squidpy_rag() -> SquidpyRAGTool | None:
    """Lazy initializer to avoid import-time dependency on LM Studio."""
    global _squidpy_rag
    if _squidpy_rag is not None:
        return _squidpy_rag
    try:
        _squidpy_rag = SquidpyRAGTool()
        return _squidpy_rag
    except Exception:
        return None


@tool
def squidpy_rag_agent(
    state: Annotated[Dict, InjectedState],
    query: str,
    data_path: str = "",
) -> str:
    """Tool that provides SpatialData and Squidpy code and explanations based on RAG.
    Generates SpatialData-first code for spatial transcriptomics analysis, falling back to
    Squidpy for spatial statistics and plotting, and optionally executing the result.

    Args:
        query: The query to answer using SpatialData/Squidpy knowledge
        data_path: Optional path to h5ad or zarr data for code execution

    Returns:
        str: Explanation, generated code, and execution output when enabled
    """
    chat_history = []

    rag = _get_squidpy_rag()
    if rag is None:
        return (
            "Squidpy/SpatialData RAG is unavailable in this environment. "
            "Reason: LM Studio is not reachable or the combined index could not be initialized. "
            "Ensure gemma4-e2b and an embedding model are loaded at http://localhost:1234/v1."
        )
    return rag.run(query, chat_history, data_path=data_path or None)


def _main() -> int:
    """Maintain the index explicitly, rather than as a side effect of starting the app."""
    parser = argparse.ArgumentParser(
        description="Maintain the SpatialData/Squidpy RAG index.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Sync the vendored repos to their pinned refs and refresh stale chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every repo even if nothing changed.",
    )
    args = parser.parse_args()

    if not args.rebuild_index:
        parser.print_help()
        return 1

    build_combined_index(refresh="force" if args.force else "auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
