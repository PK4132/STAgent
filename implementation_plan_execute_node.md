# Execute Node for Self-Reflective RAG

## Overview

Add an **`execute`** node to the Squidpy/SpatialData RAG graph in `src/squidpy_rag.py`. The node runs **after `generate`** and **before the graph can fall back to `rewrite_query`**, so generated code is validated by actually running it—not just returned as text.

### Current graph

```
Question → Retrieve → Grade → ◇ Any doc relevant?
                │                  ├─ Yes → Generate → END
                └──────────────────┘
                    Re-write query ←─ No
                         │
                         └──→ Retrieve (loop)
```

### Proposed graph

```
Question → Retrieve → Grade → ◇ Any doc relevant?
                │                  ├─ Yes → Generate → Execute → ◇ Code runs OK?
                │                  │                           ├─ Yes → END
                │                  │                           └─ No  → Re-write query → Retrieve
                └──────────────────┘
                    Re-write query ←─ No
                         │
                         └──→ Retrieve (loop)
```

The **`execute`** node sits on the **success branch** (after relevant docs are found and code is generated). If execution fails, control passes to **`rewrite_query`** so the pipeline can retrieve better context and try again—mirroring the self-reflective pattern already used at the grading step.

---

## User Review Required

> [!IMPORTANT]
> **Execution is real code, not sandboxed.** The `execute` node will run arbitrary Python via the existing `PythonREPL` class in `src/tools.py` (same backend as `python_repl_tool` in `graph_unified.py`). 
This can modify files, consume memory, and access the local filesystem. A **30 s timeout** (from `ToolConfig.repl_timeout`) should be enforced.

> [!IMPORTANT]
> **Most Squidpy/SpatialData examples need data.** Generated code often references `adata`, `.h5ad` paths, or on-disk datasets. Execution will frequently fail unless:
> - the user passes a `data_path` into the RAG call, or
> - the generate prompt injects a known default from `AppConfig.data_path`, or
> - the execute node detects missing-data errors and skips execution gracefully.
>
> See **Open Questions** below—pick one strategy before implementation.

> [!NOTE]
> **No new packages required.** Reuse `PythonREPL` from `tools.py` and existing `ToolConfig` settings. LM Studio wiring is unchanged.

---

## Open Questions

> **Decisions needed before coding:**

| # | Question | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | **How should code be extracted from `generate` output?** | (A) Structured LLM output (`code` + `explanation` fields via Pydantic) · (B) Regex/markdown fence parsing of free-text `answer` | **(A) Structured output** — reliable extraction, cleaner `execute` node |
| 2 | **What triggers `rewrite_query` after `execute`?** | (A) Any execution error/traceback · (B) Only errors + empty output · (C) LLM grades execution result | **(A) Any execution error** — simple, deterministic |
| 3 | **Should execution failure share the same rewrite budget as grading?** | (A) Single shared `rewrite_attempts` counter · (B) Separate `execute_attempts` counter | **(A) Shared counter** — one global loop guard (max 3) |
| 4 | **What data path should injected code use?** | (A) Optional `data_path` arg on `SquidpyRAGTool.run()` · (B) Always `AppConfig.data_path` · (C) Skip execution when no data path available | **(A + C)** — pass when available, skip with warning otherwise |
| 5 | **In-process vs external REPL?** | (A) `PythonREPL.run()` in-process · (B) `PythonREPL.run_external()` subprocess | **(A) In-process** for RAG smoke tests; external mode optional later |
| 6 | **What does `run()` return on success?** | (A) Explanation + execution stdout · (B) Explanation only · (C) Structured dict | **(A)** — user sees both code explanation and runtime output |

---

## Proposed Changes

### Component 1: State & Structured Generation

#### [MODIFY] [squidpy_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/squidpy_rag.py)

**Extend `SquidpyRAGState`:**

```python
class SquidpyRAGState(TypedDict):
    query: str
    context: List[Document]
    filtered_context: List[Document]
    answer: str                          # human-readable explanation (unchanged field name)
    generated_code: str                  # NEW — extracted Python to run
    execution_output: str                # NEW — stdout / error from REPL
    execution_success: bool              # NEW — True if code ran without exception
    chat_history: List[AnyMessage]
    rewrite_attempts: int
    data_path: str                       # NEW — optional h5ad path for code execution
```

**New Pydantic model for structured generation:**

```python
class GeneratedAnswer(BaseModel):
    code: str = Field(
        description="Executable Python code only. No markdown fences. No prose."
    )
    explanation: str = Field(
        description="Concise explanation of the code and Squidpy/SpatialData concepts used."
    )
```

**Update `generate` node** to use structured output instead of free-text:

```python
generator_llm = llm.with_structured_output(GeneratedAnswer)

def generate(state: SquidpyRAGState):
    ...
    result = generator_llm.invoke(messages)
    return {
        "answer": result.explanation,
        "generated_code": result.code,
    }
```

Update the generate prompt to require runnable, self-contained code and to use `{data_path}` when provided:

```
If a data path is provided, use it exactly: adata = sc.read_h5ad("{data_path}")
If no data path is provided, use a placeholder path and keep the example minimal.
Do not use markdown code fences in the structured code field.
```

---

### Component 2: Execute Node

#### [MODIFY] [squidpy_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/squidpy_rag.py)

**Import and instantiate REPL** (module-level singleton, same pattern as `graph_unified.py`):

```python
from tools import PythonREPL

_python_repl = PythonREPL()
```

**New helper — detect execution failure:**

```python
def _execution_failed(output: str) -> bool:
    """Return True if PythonREPL output indicates an error."""
    if not output or not output.strip():
        return False  # empty stdout is OK (side-effect-only code)
    failure_markers = (
        "Execution timed out",
        "External execution error",
        "exit_code=",
    )
    if output.startswith(("SyntaxError", "NameError", "TypeError", "ValueError",
                          "KeyError", "ImportError", "ModuleNotFoundError",
                          "FileNotFoundError", "AttributeError", "RuntimeError")):
        return True
    return any(marker in output for marker in failure_markers)
```

**New `execute` node:**

```python
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

    timeout = get_tool_config().repl_timeout
    output = _python_repl.run(code, timeout=timeout)
    success = not _execution_failed(output)

    return {
        "execution_output": output[: get_tool_config().max_output_length],
        "execution_success": success,
    }
```

**Optional helper `_code_requires_data`** — lightweight heuristic:

```python
_DATA_HINTS = ("read_h5ad", "read_zarr", "SpatialData.read", "sq.read.", "adata")
def _code_requires_data(code: str) -> bool:
    return any(hint in code for hint in _DATA_HINTS)
```

---

### Component 3: Routing & Graph Wiring

#### [MODIFY] [squidpy_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/squidpy_rag.py)

**New router after `execute`:**

```python
def decide_execution(state: SquidpyRAGState) -> Literal["end", "rewrite_query"]:
    if state["execution_success"]:
        return "end"
    if state["rewrite_attempts"] >= 3:
        return "end"   # give up — return answer + error output
    return "rewrite_query"
```

**Updated graph wiring** (replaces `generate → END`):

```python
graph_builder.add_node("execute", execute)

graph_builder.add_edge("generate", "execute")
graph_builder.add_conditional_edges(
    "execute",
    decide_execution,
    {"end": END, "rewrite_query": "rewrite_query"},
)
# rewrite_query → retrieve (unchanged)
```

**Update `rewrite_query` prompt** to include execution feedback when retrying after a failed run:

```python
("user",
 "Original query: {query}\n\n"
 "Previous code failed with:\n{execution_output}\n\n"
 "Rewritten query:"),
```

Pass `execution_output` from state when `execution_success` is False.

**Update `run()` signature and return value:**

```python
def run(
    self,
    query: str,
    chat_history: List[AnyMessage] = None,
    data_path: str = None,
) -> str:
    ...
    response = self.rag_pipeline.invoke({
        ...
        "generated_code": "",
        "execution_output": "",
        "execution_success": False,
        "data_path": data_path or get_data_path() or "",
    })

    parts = [response["answer"]]
    if response.get("generated_code"):
        parts.append(f"\n\n--- Generated code ---\n{response['generated_code']}")
    if response.get("execution_output"):
        label = "Execution output" if response["execution_success"] else "Execution error"
        parts.append(f"\n\n--- {label} ---\n{response['execution_output']}")
    return "".join(parts)
```

---

### Component 4: Tool & Config Integration

#### [MODIFY] [squidpy_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/squidpy_rag.py) — `squidpy_rag_agent` tool

Add optional `data_path` parameter so the unified agent can pass the user's dataset:

```python
@tool
def squidpy_rag_agent(
    state: Annotated[Dict, InjectedState],
    query: str,
    data_path: str = "",
) -> str:
    ...
    return rag.run(query, chat_history, data_path=data_path or None)
```

#### [MODIFY] [config.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/config.py) — `ToolConfig`

Add RAG execution settings:

```python
# RAG execution settings
rag_exec_timeout: int = 30          # mirrors repl_timeout; override via env
rag_exec_enabled: bool = True       # set False to skip execute node (generate-only mode)
rag_max_rewrite_attempts: int = 3
```

Load from env in `__post_init__`:

```python
self.rag_exec_enabled = os.getenv("RAG_EXEC_ENABLED", "true").lower() == "true"
```

#### [MODIFY] [.env.example](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/.env.example)

```dotenv
# RAG code execution (execute node)
RAG_EXEC_ENABLED=true
RAG_EXEC_TIMEOUT=30
```

#### [MODIFY] [prompt.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/prompt.py)

Update `squidpy_rag_agent` tool description:

```xml
<tool name="squidpy_rag_agent">
  - Purpose: Retrieve Squidpy/SpatialData code via RAG, generate an answer, and execute the code
  - Index: db/chroma_combined_db (LM Studio local)
  - Optional: pass data_path for code that reads h5ad/spatial datasets
  - Output: explanation + generated code + execution stdout (or error if execution fails)
  - When: Squidpy or SpatialData API/code guidance; pass data_path when code needs real data
</tool>
```

#### [NO CHANGE] [graph_unified.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/src/graph_unified.py)

Existing `squidpy_rag_agent` import stays as-is. The agent can optionally pass `data_path` once the tool signature is extended; no graph wiring changes required.

---

### Component 5: Bypass Mode (optional but recommended)

When `RAG_EXEC_ENABLED=false`, skip the execute node for faster generate-only responses:

```python
if get_tool_config().rag_exec_enabled:
    graph_builder.add_edge("generate", "execute")
    graph_builder.add_conditional_edges("execute", decide_execution, {...})
else:
    graph_builder.add_edge("generate", END)
```

This preserves backward-compatible behavior for environments where running arbitrary code is undesirable.

---

## Node Summary

| Node | Role | Input | Output |
|------|------|-------|--------|
| `retrieve` | Fetch docs from combined Chroma index | `query` | `context` |
| `grade_documents` | LLM relevance filter | `query`, `context` | `filtered_context` |
| `generate` | Structured code + explanation | `query`, `filtered_context` | `answer`, `generated_code` |
| **`execute`** | **Run `generated_code` via PythonREPL** | **`generated_code`, `data_path`** | **`execution_output`, `execution_success`** |
| `rewrite_query` | Rewrite query (now also uses execution errors) | `query`, `execution_output` | `query`, `rewrite_attempts++` |

---

## Verification Plan

### Automated Tests

#### [MODIFY] [tests/test_spatialdata_rag.py](file:///c:/Pascal's%20Folders/QIMR/STAgent/tests/test_spatialdata_rag.py)

Add tests (all mocked — no real code execution in CI):

| Test | Asserts |
|------|---------|
| `test_execute_success_routes_to_end` | `decide_execution` returns `"end"` when `execution_success=True` |
| `test_execute_failure_routes_to_rewrite` | `decide_execution` returns `"rewrite_query"` on failure, attempts < 3 |
| `test_execute_failure_loop_guard` | After 3 rewrite attempts, failure routes to `"end"` |
| `test_execute_node_calls_repl` | Mock `PythonREPL.run`; verify `generated_code` is passed |
| `test_execute_skips_when_no_data_path` | Code with `read_h5ad` + empty `data_path` → `execution_success=False`, no REPL call |
| `test_generate_structured_output` | Mock structured LLM; verify `generated_code` and `answer` populated |

Run:

```bat
pytest tests/test_spatialdata_rag.py -v
```

### Manual Verification (Anaconda Prompt)

Prerequisites: LM Studio running, `conda activate STAgent`, combined index already built.

#### 1. Simple code that runs without data

```bat
cd /d "c:\Pascal's Folders\QIMR\STAgent"
python -c "import sys; sys.path.insert(0, 'src'); from squidpy_rag import SquidpyRAGTool; rag=SquidpyRAGTool(); print(rag.run('Show a minimal squidpy import example that prints the squidpy version'))"
```

Expected path: `retrieve → grade → generate → execute → END` with stdout containing a version string.

#### 2. Code that needs data (pass data_path)

```bat
python -c "import sys; sys.path.insert(0, 'src'); from squidpy_rag import SquidpyRAGTool; from config import get_data_path; rag=SquidpyRAGTool(); print(rag.run('Load the dataset and print adata shape', data_path=get_data_path()))"
```

Expected: execution succeeds if the h5ad file exists at `AppConfig.data_path`.

#### 3. Force execution failure → rewrite loop

Ask for code that will fail (e.g. reference a nonexistent module), verify `rewrite_query` is invoked and `rewrite_attempts` increments.

#### 4. Disable execution (generate-only fallback)

```bat
set RAG_EXEC_ENABLED=false
python -c "import sys; sys.path.insert(0, 'src'); from squidpy_rag import SquidpyRAGTool; print(SquidpyRAGTool().run('How do I compute spatial neighbors in squidpy?'))"
```

Expected: code + explanation returned, no execution output section.

#### 5. Tool integration

```bat
python -c "import sys; sys.path.insert(0, 'src'); from squidpy_rag import squidpy_rag_agent; print(squidpy_rag_agent.invoke({'state': {'messages': []}, 'query': 'Print squidpy version', 'data_path': ''}))"
```

---

## Anaconda Testing Guide

All commands assume project root `c:\Pascal's Folders\QIMR\STAgent` and `conda activate STAgent`.

### One-time setup (if not already done)

```bat
conda env create -f environment.yml
conda activate STAgent
copy src\.env.example src\.env
```

Ensure `src\.env` contains LM Studio vars **and**:

```dotenv
RAG_EXEC_ENABLED=true
RAG_EXEC_TIMEOUT=30
```

### Daily workflow (after index is built)

```bat
conda activate STAgent
cd /d "c:\Pascal's Folders\QIMR\STAgent"
python -c "import sys; sys.path.insert(0, 'src'); from squidpy_rag import SquidpyRAGTool; print(SquidpyRAGTool().run('YOUR QUESTION HERE'))"
```

No re-indexing or env recreation needed unless dependencies change.

### Run tests

```bat
pytest tests/test_spatialdata_rag.py -v
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Execution timed out` | Increase `RAG_EXEC_TIMEOUT` or simplify generated code |
| `Execution skipped: no data_path` | Pass `data_path=` to `run()` or set `STAGENT_DATA_PATH` in `.env` |
| `ModuleNotFoundError` at runtime | Generated code imports a package not in the env — retry may fix via rewrite loop |
| Execute node never runs | Check `RAG_EXEC_ENABLED=true` in `src/.env` |
| Infinite rewrite loops | Verify `rewrite_attempts >= 3` guard in both `decide_relevance` and `decide_execution` |
| Structured output parse errors from Gemma | Fall back to fence-parsing in `generate` if `with_structured_output` is unreliable on local model |

---

## Implementation Order

1. Add state fields + `GeneratedAnswer` model
2. Refactor `generate` to structured output
3. Add `_execution_failed`, `_code_requires_data`, `execute` node
4. Add `decide_execution` router + rewire graph
5. Update `rewrite_query` prompt with execution feedback
6. Extend `run()` and `squidpy_rag_agent` with `data_path`
7. Add config/env toggles
8. Write tests
9. Manual smoke tests with LM Studio

Estimated scope: **~150–200 lines changed** in `squidpy_rag.py`, **~80 lines** of new tests, minor edits to `config.py`, `.env.example`, and `prompt.py`.
