import os
from pathlib import Path
from typing import Annotated, TypedDict, List, Dict, Any, Literal
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
from prompt import spatial_processing_prompt
from config import get_data_path, get_tool_config

load_dotenv(Path(__file__).resolve().with_name(".env"))

SQUIDPY_REPO_PATH = "./packages_available/squidpy"
SPATIALDATA_REPO_PATH = "./packages_available/spatialdata"
COMBINED_PERSIST_DIR = "./db/chroma_combined_db"

_DATA_HINTS = ("read_h5ad", "read_zarr", "SpatialData.read", "sq.read.", "adata")

_python_repl = None


class SquidpyRAGState(TypedDict):
    query: str
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
        description="Concise explanation of the code and Squidpy/SpatialData concepts used."
    )


def _max_rewrite_attempts() -> int:
    return get_tool_config().rag_max_rewrite_attempts


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


def _prepare_code_for_execution(code: str, data_path: str) -> str:
    """Prepend a trusted DATA_PATH so the LLM never needs to copy Windows paths."""
    if not data_path:
        return code
    return f"DATA_PATH = {data_path!r}\n{code}"


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
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source_repo"] = source_repo
    return chunks


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
        if not os.path.exists(SQUIDPY_REPO_PATH):
            print(f"Cloning Squidpy repository to {SQUIDPY_REPO_PATH}...")
            Repo.clone_from("https://github.com/scverse/squidpy", to_path=SQUIDPY_REPO_PATH)

        if not os.path.exists(SPATIALDATA_REPO_PATH):
            print(f"Cloning SpatialData repository to {SPATIALDATA_REPO_PATH}...")
            Repo.clone_from("https://github.com/scverse/spatialdata", to_path=SPATIALDATA_REPO_PATH)

        embeddings = _get_lm_studio_embeddings()

        if not os.path.exists(COMBINED_PERSIST_DIR):
            print("Creating combined Squidpy + SpatialData vector database...")

            squidpy_chunks = _load_python_chunks(SQUIDPY_REPO_PATH, "squidpy")
            print(f"Loaded {len(squidpy_chunks)} chunks from Squidpy")

            spatialdata_chunks = _load_python_chunks(SPATIALDATA_REPO_PATH, "spatialdata")
            print(f"Loaded {len(spatialdata_chunks)} chunks from SpatialData")

            all_chunks = squidpy_chunks + spatialdata_chunks
            print(f"Total {len(all_chunks)} text chunks for combined index")

            combined_store = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=COMBINED_PERSIST_DIR,
            )
            print(f"Created new Chroma database at {COMBINED_PERSIST_DIR}")
        else:
            combined_store = Chroma(
                persist_directory=COMBINED_PERSIST_DIR,
                embedding_function=embeddings,
            )
            print(f"Loaded existing Chroma database from {COMBINED_PERSIST_DIR}")

        return combined_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})

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
                "You are a query rewriter for a Squidpy and SpatialData code RAG system. "
                "Rewrite the user's query to be more specific and retrievable from a Python "
                "codebase index. Output only the rewritten query, nothing else.",
            ),
            (
                "user",
                "Original query: {query}\n\n{execution_feedback}Rewritten query:",
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
            context_content = "\n\n".join(
                f"[{doc.metadata.get('source_repo', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )
            chat_history = state["chat_history"]
            data_path = state.get("data_path", "")

            data_path_instruction = (
                "No data path was provided. Keep the code example minimal and self-contained."
            )
            if data_path:
                data_path_instruction = (
                    "A data file path will be injected as the variable DATA_PATH before your code runs.\n"
                    "Use DATA_PATH only — do NOT hardcode any file path string in the code.\n"
                    "Examples:\n"
                    "- h5ad: adata = sc.read_h5ad(DATA_PATH)\n"
                    "- zarr: import spatialdata as sd; sdata = sd.read_zarr(DATA_PATH)\n"
                    "Pick the loader appropriate for the file extension and query."
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
                    "You are an expert in Squidpy and SpatialData, specializing in providing "
                    "authentic Python code and explanations on their usage.\n"
                    "Return structured output with two fields:\n"
                    "- code: executable Python only (no markdown fences, no prose)\n"
                    "- explanation: concise explanation of the code\n\n"
                    "REMEMBER to specify shape = None for STARmap spatial transcriptomic data.\n"
                    "The code must be runnable as-is in a Python REPL.\n"
                    "{data_path_instruction}\n\n"
                    "Additional instructions:\n{spatial_processing_prompt}\n\n"
                    "CONTEXT FROM CODEBASE:\n{context_content}\n\n",
                ),
                ("user", "USER QUESTION: {query}"),
            ])

            messages = prompt.invoke({
                "query": state["query"],
                "chat_history": chat_history,
                "context_content": context_content,
                "spatial_processing_prompt": spatial_processing_prompt,
                "data_path_instruction": data_path_instruction,
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

            if not state.get("data_path") and _code_requires_data(code):
                return {
                    "execution_output": (
                        "Execution skipped: generated code references data files but no "
                        "data_path was provided."
                    ),
                    "execution_success": False,
                }

            tool_config = get_tool_config()
            timeout = tool_config.rag_exec_timeout
            executable_code = _prepare_code_for_execution(code, state.get("data_path", ""))
            output = _get_python_repl().run(executable_code, timeout=timeout)
            success = not _execution_failed(output)

            return {
                "execution_output": output[: tool_config.max_output_length],
                "execution_success": success,
            }

        def rewrite_query(state: SquidpyRAGState):
            execution_feedback = ""
            if state.get("execution_output") and not state.get("execution_success", True):
                execution_feedback = (
                    f"Previous code failed with:\n{state['execution_output']}\n\n"
                )
            response = rewriter.invoke({
                "query": state["query"],
                "execution_feedback": execution_feedback,
            })
            rewritten = response.content.strip()
            return {
                "query": rewritten,
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
        """Run the Squidpy/SpatialData RAG pipeline with the given query and chat history."""
        if chat_history is None:
            chat_history = []

        resolved_data_path = data_path or get_data_path() or ""

        response = self.rag_pipeline.invoke({
            "query": query,
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
    """Tool that provides Squidpy and SpatialData code and explanations based on RAG.
    Uses the Squidpy and SpatialData codebases to generate accurate code for spatial
    transcriptomics analysis, optionally executing the generated code.

    Args:
        query: The query to answer using Squidpy/SpatialData knowledge
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
