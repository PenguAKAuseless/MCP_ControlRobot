# MCP Control Robot

MCP Control Robot is a local MCP bridge and tool server collection. It connects MCP stdio servers to a WebSocket endpoint and supports both local and remote MCP transports.

## Overview

This repository provides:

- A WebSocket-to-stdio bridge (`mcp_pipe.py`) with automatic reconnect.
- Config-driven multi-server startup (`mcp_config.json`).
- Example local MCP tools (`calculator.py`, `robot_control.py`).
- A standalone legal QA server (`legal_answer_server.py`) using a hybrid retrieval pipeline (Milvus + Neo4j + LLM).

## Features

- Bidirectional communication between MCP servers and a remote endpoint.
- Exponential-backoff reconnection for long-running sessions.
- Configurable server startup for `stdio`, `sse`, and `http` targets.
- Standalone legal answer workflow with retrieval, graph expansion, and grounded generation.

## Requirements

- Python 3.7+
- Dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Copy `.env.example` to `.env` and fill required values.

Required for bridge:

- `MCP_ENDPOINT`

Required for legal-answer pipeline:

- Cloud mode (default) requires at least one generation key (priority: OpenAI -> Claude -> Gemini -> TogetherAI):
	- `MCP_OPENAI_API_KEY`
	- `MCP_CLAUDE_API_KEY` (or aliases `MCP_ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY`)
	- `MCP_GEMINI_API_KEY`
	- `MCP_TOGETHER_API_KEY`
- Cloud mode also requires at least one embedding-capable key:
	- `MCP_OPENAI_API_KEY` or `MCP_GEMINI_API_KEY` or `MCP_TOGETHER_API_KEY`
	- Optional aliases are also accepted: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`
- Local mode (`MCP_MODE=local` or `MCP_LOCAL_MODE=1`) uses local Ollama models for both generation and embeddings, so cloud API keys are optional.
- `MCP_MILVUS_ENDPOINT` (or `MCP_MILVUS_URI` for local Milvus)
- `MCP_MILVUS_TOKEN` (required when using online/Zilliz endpoint)
- `MCP_MILVUS_COLLECTION`
- `MCP_NEO4J_URI`
- `MCP_NEO4J_USER`
- `MCP_NEO4J_PASSWORD`

Optional tuning:

- `MCP_TOP_K` (default `4`)
- `MCP_MAX_TOP_K` (default `8`)
- `MCP_CACHE_TTL_SECONDS` (default `300`)
- `MCP_CONTEXT_CHAR_BUDGET` (default `2800`)
- `MCP_MAX_CONTEXT_ITEMS` (default `8`)
- `MCP_MAX_ITEM_CHARS` (default `700`)
- `MCP_MILVUS_DATABASE` (default empty)
- `MCP_MILVUS_VECTOR_FIELD` (default `dense_vector`)
- `MCP_NEO4J_DATABASE` (default empty)
- `MCP_EMBEDDING_DIMENSIONS` (default empty, model default)
- `MCP_LLM_MODEL` (default `gpt-4o-mini`)
- `MCP_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `MCP_OPENAI_LLM_MODEL`, `MCP_OPENAI_EMBEDDING_MODEL`
- `MCP_CLAUDE_LLM_MODEL`
- `MCP_GEMINI_LLM_MODEL`, `MCP_GEMINI_EMBEDDING_MODEL`, `MCP_GEMINI_BASE_URL`
- `MCP_TOGETHER_LLM_MODEL`, `MCP_TOGETHER_EMBEDDING_MODEL`, `MCP_TOGETHER_BASE_URL`
- `MCP_PREFERRED_GENERATION_PROVIDER` (`openai|claude|gemini|togetherai`)
- `MCP_PREFERRED_EMBEDDING_PROVIDER` (`openai|gemini|togetherai`)
- `MCP_STRICT_PREFERRED_PROVIDER` (default `0`; when `1`, do not fallback to other providers)
- `MCP_VERIFY_PROVIDER_ON_STARTUP` (default `1`, fail startup if no provider can be reached)
- `MCP_VERIFY_PROVIDER_EMBEDDINGS` (default `1`, include embedding API probe at startup)
- `MCP_MODE` (`cloud|local`, default `cloud`)
- `MCP_LOCAL_MODE` (`0|1`, local mode shortcut)
- `MCP_LOCAL_BASE_URL` (default `http://127.0.0.1:11434`)
- `MCP_LOCAL_LLM_MODEL` (default `llama3.1:8b`)
- `MCP_LOCAL_EMBEDDING_MODEL` (default `nomic-embed-text`)
- `MCP_PROVIDER_TIMEOUT_SECONDS` (default `60` in cloud mode, `180` in local mode)

## Local Model Setup (16GB RAM)

Recommended local model pair for 16GB RAM / 512GB storage:

- LLM: `llama3.1:8b` (good quality/speed tradeoff)
- Embeddings: `nomic-embed-text`

Install Ollama, then pull both models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Windows quick setup (pull models, write `.env`, install dependencies):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local_mode.ps1
```

Enable local mode in `.env`:

```env
MCP_MODE=local
MCP_LOCAL_MODE=1
MCP_LOCAL_BASE_URL=http://127.0.0.1:11434
MCP_LOCAL_LLM_MODEL=llama3.1:8b
MCP_LOCAL_EMBEDDING_MODEL=nomic-embed-text
```

After that, start the legal answer server as usual (for example via `python mcp_pipe.py legal-answer`).

## Run

Run a single local script (backward compatible mode):

```bash
python mcp_pipe.py calculator.py
```

Run all enabled servers from `mcp_config.json`:

```bash
python mcp_pipe.py
```

Run a single configured server by name:

```bash
python mcp_pipe.py legal-answer
```

## Data Ingestion Pipeline

Use the ingestion pipeline to process raw files under `docs/`, write normalized chunks into `docs/processed/`, and import them into Milvus + Neo4j.

Process docs only:

```bash
python scripts/kb_pipeline.py process --enable-ocr
```

Import processed chunks only:

```bash
python scripts/kb_pipeline.py import
```

Process + import in one command:

```bash
python scripts/kb_pipeline.py run --enable-ocr
```

Start legal-answer server, then run process + import pipeline:

```bash
python scripts/kb_pipeline.py server-run --enable-ocr
```

### Processing Outputs

- `docs/processed/text/`: normalized extracted text per source document
- `docs/processed/chunks/`: chunked JSONL per document
- `docs/processed/chunks/all_chunks.jsonl`: combined chunk stream
- `docs/processed/ingestion_state.json`: state tracking for processed/imported hashes and status

## Test Runner

Run test suite and produce machine-readable + human-readable reports:

```bash
python scripts/run_test_suite.py --scenario test/scenario.json --output-dir test/reports --include-graph
```

Dry-run (validate dataset and report structure without model calls):

```bash
python scripts/run_test_suite.py --scenario test/scenario.json --output-dir test/reports --dry-run
```

Reports are written to `test/reports/` as:

- `test-report-<timestamp>.json`
- `test-report-<timestamp>.md`

## Config-Driven Servers

`mcp_pipe.py` loads config from:

1. `MCP_CONFIG` environment variable
2. `./mcp_config.json`

Supported server entry types:

- `type: "stdio"` with `command` and optional `args`
- `type: "sse"` or `type: "http"` with `url` (proxied via `python -m mcp_proxy`)

When no CLI argument is provided, all enabled servers are started (`disabled: true` entries are skipped).

## Legal Answer Pipeline Review

The legal-answer server exposes two MCP tools:

- `answer_service_healthcheck`
- `answer_legal_question(question, top_k=4, include_graph=True, use_cache=True)`

Pipeline in `legal_answer_server.py`:

1. Normalize user query text.
2. Embed query using provider fallback (OpenAI -> Claude -> Gemini -> TogetherAI; embeddings use providers that support embeddings).
3. Search Milvus collection by `dense_vector` similarity.
4. Expand related nodes from Neo4j using IDs from retrieved Milvus hits.
5. Build a bounded context window.
6. Generate a grounded legal answer with provider fallback and source tags.

## Knowledge Base Contract (What Data Is Required)

To produce grounded answers, the pipeline expects the following data shape.

Milvus collection (`MCP_MILVUS_COLLECTION`) must contain:

- `dense_vector`: embedding vector used for ANN search.
- `text`: chunk content (required; empty text is discarded).
- `article_id`: stable identifier for citation/linking.
- `doc_id`: parent document identifier (used for grouping and KG linking).
- `title`: optional but strongly recommended.
- `doc_type`: optional metadata.

Neo4j graph should contain nodes with at least one join key:

- `id` or `doc_id` or `article_id` or `clause_id`

And recommended descriptive fields:

- `title` or `name`
- `text` or `raw_text`

Relationships can be any type; the pipeline traverses `(n)-[r]-(m)` around seed nodes.

## What Is Missing Today To Reliably Answer

The runtime code is present, but the repository does not include the data build pipeline. To get reliable answers, these pieces are still needed in your knowledge base workflow:

- Milvus ingestion job that chunks legal documents, computes embeddings, and writes required fields.
- Neo4j ingestion job that mirrors document/article IDs from Milvus and creates meaningful relations.
- Schema/bootstrap scripts (or docs) for Milvus collection and Neo4j constraints/indexes.
- Data quality checks for non-empty `text`, stable `article_id` and `doc_id`, and one-to-one joinability between Milvus rows and Neo4j nodes.
- Coverage policy for legal corpus (jurisdiction, effective dates, updates, repealed rules).
- Evaluation set for citation accuracy and answer correctness.

Without these, the system may return "No relevant evidence found" or generate answers with weak grounding.

## Project Structure

- `mcp_pipe.py`: WebSocket bridge and process manager
- `mcp_config.json`: server definitions
- `calculator.py`: sample MCP calculator server
- `robot_control.py`: robot control MCP server
- `legal_answer_server.py`: hybrid legal QA server
- `.env.example`: required environment variables
- `requirements.txt`: Python dependencies

## Contributing

Contributions are welcome. Open a pull request with a clear description of changes and testing notes.

## License

This project is licensed under the MIT License.
