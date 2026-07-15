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

load_dotenv(Path(__file__).resolve().with_name(".env"))

SQUIDPY_REPO_PATH = "./packages_available/squidpy"
SPATIALDATA_REPO_PATH = "./packages_available/spatialdata"
COMBINED_PERSIST_DIR = "./db/chroma_combined_db"


class SquidpyRAGState(TypedDict):
    query: str
    context: List[Document]
    filtered_context: List[Document]
    answer: str
    chat_history: List[AnyMessage]
    rewrite_attempts: int


class GradeDoc(BaseModel):
    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the query, 'no' otherwise."
    )


def decide_relevance(state: SquidpyRAGState) -> Literal["generate", "rewrite_query"]:
    if state["filtered_context"] or state["rewrite_attempts"] >= 3:
        return "generate"
    return "rewrite_query"


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

        rewrite_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a query rewriter for a Squidpy and SpatialData code RAG system. "
                "Rewrite the user's query to be more specific and retrievable from a Python "
                "codebase index. Output only the rewritten query, nothing else.",
            ),
            ("user", "Original query: {query}\n\nRewritten query:"),
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
                    "authentic Python code and explanations on their usage. IMPORTANT: do not use "
                    "python bracket for the code. REPEAT: do not use python bracket for the code. "
                    "For each query, respond with:\n"
                    "1. Python code to solve the user's question (Squidpy or SpatialData as appropriate).\n"
                    "2. A concise explanation of the code, focusing on library-specific concepts, "
                    "methods, and relevant parameters.\n\n"
                    "3. REMEMBER to specify shape = None for STARmap spatial transcriptomic data.\n"
                    "The following are some additional instructions:\n"
                    "{spatial_processing_prompt}\n\n"
                    "CONTEXT FROM CODEBASE:\n{context_content}\n\n",
                ),
                ("user", "USER QUESTION: {query}"),
            ])

            messages = prompt.invoke({
                "query": state["query"],
                "chat_history": chat_history,
                "context_content": context_content,
                "spatial_processing_prompt": spatial_processing_prompt,
            })
            response = llm.invoke(messages)
            return {"answer": response.content}

        def rewrite_query(state: SquidpyRAGState):
            response = rewriter.invoke({"query": state["query"]})
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
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def run(self, query: str, chat_history: List[AnyMessage] = None):
        """Run the Squidpy/SpatialData RAG pipeline with the given query and chat history."""
        if chat_history is None:
            chat_history = []

        response = self.rag_pipeline.invoke({
            "query": query,
            "chat_history": chat_history,
            "context": [],
            "filtered_context": [],
            "answer": "",
            "rewrite_attempts": 0,
        })

        return response["answer"]


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
def squidpy_rag_agent(state: Annotated[Dict, InjectedState], query: str) -> str:
    """Tool that provides Squidpy and SpatialData code and explanations based on RAG.
    Uses the Squidpy and SpatialData codebases to generate accurate code for spatial
    transcriptomics analysis.

    Args:
        query: The query to answer using Squidpy/SpatialData knowledge

    Returns:
        str: Code and explanation for the query
    """
    chat_history = []

    rag = _get_squidpy_rag()
    if rag is None:
        return (
            "Squidpy/SpatialData RAG is unavailable in this environment. "
            "Reason: LM Studio is not reachable or the combined index could not be initialized. "
            "Ensure gemma4-e2b and an embedding model are loaded at http://localhost:1234/v1."
        )
    return rag.run(query, chat_history)
