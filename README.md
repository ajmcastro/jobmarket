# Job-Market Intelligence Assistant

A prototype that answers questions about skills, roles, companies, and job-market trends from a sample
of Linkedin job postings, using two complementary techniques:

1. **Baseline RAG** — chunk → embed → retrieve → LLM-generate, with citations.
2. **Knowledge Graph** — a small NetworkX graph over postings/companies/roles/skills/locations, queried by traversal.
3. **Graph-RAG design** — an architectural write-up (not implemented) of how the two would combine.

The core logic (chunking, embedding, vector search, answer generation, graph building, graph queries)
lives in the importable [`jobmarket/`](jobmarket/) package. It is possible to access it in two ways:

- **[`mleng_take_home_task.ipynb`](mleng_take_home_task.ipynb)** — the primary deliverable, walking through
  both parts cell by cell with inline commentary and demo output.
- **[`api/`](api/)** — an optional FastAPI service exposing the same RAG and knowledge-graph capabilities
  over HTTP, with interactive Swagger docs. See [API](#api) below.

This README covers setup/run instructions and a design-note summary; see [`presentation.md`](presentation.md)
for the accompanying slide deck.

**Important Remark**

This work was done using Visual Code with Copilot mainly in ask mode. Parts of the code were done in this way. For example, in implementing the networkx code, which is a framework that I did not know before.

Agent mode was used to produce documentation, including this README.md and the presentation.md file.

The code architecture for the jobmarket package plus the API were done following my typical 'boilerplate code' for this kind of projects. Although the code architecture is based in production ready applications, the version presented here is incomplete, as such, not ready for production.

Some alternative approaches related with parts of the RAG process, for example, a different embeddings strategy or the use of turbovec, are based on developments I have done for other projects.

A critical analysis and review was done by me, although not exhaustive.


---

## Setup — from zip to running notebook

### 1. Prerequisites

- **Python 3.12+** — a `.python-version` file pins `3.12`; you don't need to install it yourself,
  `uv` will fetch a matching interpreter automatically if one isn't already on your machine.
- **[`uv`](https://docs.astral.sh/uv/)** — the package/dependency manager this project uses. Install it if you don't have it:

  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

  # or via Homebrew
  brew install uv
  ```

  Verify with `uv --version`.

### 2. Get the code

Unzip the archive (or `git clone` the repo, if you were given a URL) and `cd` into the project folder:

```bash
cd mleng_take_home_task   # or whatever the extracted folder is named
```

The `data/` directory with the sample CSVs is already included — no separate download needed.

### 3. Install dependencies

```bash
uv sync
```

This creates a local `.venv/` and installs everything pinned in `uv.lock` — `pandas`, `sentence-transformers`
(+ `torch`), `faiss-cpu`, `turbovec`, `anthropic`, `networkx`, plus `jupyter`/`nbconvert`/`ipykernel` as dev
dependencies. First install pulls in `torch`, so expect it to take a few minutes and a few hundred MB of
downloads on a clean machine.

### 4. (Optional) Set your Anthropic API key

The Baseline RAG's answer-generation step calls Claude (`claude-haiku-4-5`) to write grounded answers. This
is **entirely optional** — with no key configured (or if the call fails for any reason), the pipeline
automatically falls back to a deterministic mock answer built directly from the retrieved evidence, so the
notebook always runs end-to-end either way.

To use the real LLM, set the standard Anthropic SDK environment variable before launching Jupyter:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Nothing else in the notebook talks to a network API:
embeddings and the knowledge graph both run 100% locally.

### 5. Run the notebook

```bash
make notebook       # or: uv run jupyter lab
```

(or `uv run jupyter notebook`, or open the folder in VS Code / PyCharm with a Jupyter extension and pick the
`.venv` interpreter). Open `mleng_take_home_task.ipynb` and run it top to bottom
(**Kernel → Restart Kernel and Run All Cells**).

**Notes:**

- The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) from Hugging Face — needs internet
  access once; it's cached under `~/.cache/huggingface` afterward, so subsequent runs are offline-friendly.
- A full run (embedding ~16K chunks, building the graph, all demo queries) takes a few minutes on a laptop
  CPU — no GPU required.
- **Cells must run top-to-bottom.** Later cells depend on state from earlier ones (`postings`, `corpus`,
  `kg`, etc.). If you see a `NameError`, you most likely restarted the kernel or ran cells out of order —
  use *Run All* rather than running an individual cell in isolation.

**Non-interactive alternative** (reproduce the whole notebook from the terminal, no browser needed):

```bash
make notebook-execute   # or: uv run jupyter nbconvert --to notebook --execute --inplace mleng_take_home_task.ipynb
```

### 6. (Optional) Run the API instead

```bash
make api   # or: uv run uvicorn api.main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs. See [API](#api) below for details, or
run `make help` for the full list of shortcuts (notebook, API, demos, cache cleanup).

### 7. What you'll see

- **Part 1 (Baseline RAG)** — builds embeddings and both vector indices, then answers 4 demo questions with
  citations (`job_id`, title, company, location, excerpt). Without an API key, answers are printed as
  `[Mock mode]` output built directly from the retrieved evidence rather than LLM prose.
- **Part 2 (Knowledge Graph)** — builds a NetworkX graph (~6,760 nodes / ~33,300 edges) and answers "which
  companies hire for both data engineering and ML engineering roles, and what skills connect them" via graph edge traversal/neighbor lookup (no BFS/DFS search) and basic set aggregation (set intersection + frequency counting).
- **Part 3 (Graph-RAG design)** — an architectural write-up only (no code) of how graph traversal would
  augment the vector-search pipeline from Part 1.

---

## API

An optional FastAPI service — not part of the requested task, but exposes Parts 1 and 2 over HTTP for
anyone who'd rather query the assistant that way than run notebook cells. It imports the exact same
`jobmarket/` package the notebook does, so behavior (and demo output) is identical either way.

### Run it

```bash
make api   # or: uv run uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for Swagger UI (or `/redoc` for ReDoc). `ANTHROPIC_API_KEY` is read
the same way as in the notebook — optional, with mock-mode fallback.

### Demo via Makefile

The three required demo question types (direct retrieval + comparison/synthesis for RAG, the bridging
query for the KG) are wired up as `make` targets:

```bash
make demo-auto   # starts the API in the background, runs both demos below, then stops it — one command
# — or, with `make api` already running in another shell —
make demo-rag    # RAG: direct-retrieval + comparison/synthesis questions
make demo-kg     # KG: the required "companies bridging Data Engineer / ML Engineer roles" question
make demo        # both of the above
```

Run `make help` for the full list of targets (notebook, API, demos, cache cleanup).

### Startup cost & caching

Building the RAG index (embedding ~16K chunks) and the knowledge graph (~6.7K nodes) takes a few minutes
on first run. To avoid paying that cost on every restart, both are cached to disk under `.cache/` (the
corpus + embeddings for RAG, the pickled graph for the KG) the first time they're built, and loaded from
there on subsequent startups — a warm restart is near-instant. **Delete `.cache/` to force a rebuild**
(e.g. after changing the source CSVs, the chunking parameters, or the embedding model).

### Endpoints

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness/readiness — whether the RAG index and KG finished building |
| `POST /rag/ask` | Ask a natural-language question; returns an answer + citations (Part 1) |
| `GET /rag/backends` | List vector-search backends (`faiss`, `turbovec`) and index stats |
| `GET /kg/stats` | Node/edge counts by type |
| `GET /kg/roles` | Canonical roles (Graph standardized roles - not the original free text) and their posting counts |
| `GET /kg/roles/{role}/skills` | Top skills mentioned in postings for a role |
| `GET /kg/bridge` | Companies hiring for both of two roles + shared skills — the required KG demo query |

Example:

```bash
curl -X POST http://127.0.0.1:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which postings mention retrieval-augmented generation, vector databases, or LLM application development?"}'

curl "http://127.0.0.1:8000/kg/bridge?role_a=Data%20Engineer&role_b=Machine%20Learning%20Engineer"
```

---

## Repo structure

```
.
├── mleng_take_home_task.ipynb   # the whole solution — read top to bottom, in a notebook
├── jobmarket/                    # shared package: RAG + knowledge-graph logic, used by both
│   ├── config.py                  #   the notebook and the API
│   ├── data.py
│   ├── rag/                       # chunking, embeddings, vector stores, retrieval, generation
│   └── kg/                        # taxonomy, graph build, graph queries
├── api/                          # FastAPI service exposing jobmarket/ over HTTP
│   ├── main.py                    # app + lifespan startup (builds/loads cached index & graph)
│   ├── schemas.py                 # Pydantic request/response models
│   └── routers/                   # rag.py, kg.py
├── data/                        # sample_postings.csv, sample_companies.csv, sample_skills.csv,
│                                 # sample_skill_lookup.csv, sample_industries.csv
├── .cache/                       # gitignored — disk cache for the API's index/graph (see API section)
├── Makefile                      # shortcuts: notebook, api, demo-rag/demo-kg/demo-auto, cache cleanup — see `make help`
├── pyproject.toml                # dependency spec (uv)
├── uv.lock                       # locked dependency versions, for reproducible installs
├── .python-version                # pins Python 3.12
├── README.md                     # this file
├── presentation.md               # 15-minute presentation deck (mermaid diagrams)

```

---

## Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | No | Enables real Claude-generated answers in Part 1 (model: `claude-haiku-4-5`). Without it, the pipeline automatically falls back to a deterministic mock answer built from retrieved evidence — see step 4 above. |

No other secrets or environment variables are used anywhere in the project.

---

## Design note

### Architecture at a glance

All pipeline logic (chunking/embedding/retrieval/generation for Part 1, graph build/query for Part 2)
lives in the `jobmarket/` package, imported by both the notebook and the optional API — see [API](#api).

- **Part 1 — Baseline RAG.** Chunk posting text (~180 words, 40-word overlap) → embed locally with
  `sentence-transformers/all-MiniLM-L6-v2` (no API key needed to build the index) → index in **two**
  interchangeable vector stores behind a common interface — **FAISS** (`IndexFlatIP`, exact) and
  **turbovec** (`IdMapIndex`, 4-bit quantized) — → retrieve top-k, de-duplicated by posting → answer with
  Claude (`claude-haiku-4-5`) or a deterministic mock fallback → return `job_id`/title/company/location/
  excerpt citations for every answer.

  A typical split-chunk strategy based in a fixed size is implemented in **split_into_chunks**. A slighthly improved version that considers sentence-aware boundaries is implemented in **improved_split_into_chunks**.
- **Part 2 — Knowledge Graph.** A NetworkX `DiGraph` over `JobPosting`, `Company`, `Role`, `Skill`, and
  `Location` nodes. `Role` is derived from `title` via a small ordered keyword taxonomy; `Skill` is
  extracted from posting text via a curated ~50-term tech/data vocabulary (the raw `sample_skills.csv`
  categories — `IT`, `ENG`, `ANLS`, … — are too coarse to answer skill-level questions). Demonstrated on the
  required question — "which companies hire for both data engineering and ML engineering roles, and what
  skills connect those roles" — via traversal and set intersection/aggregation.
- **Part 3 — Graph-RAG design.** An architectural description (not implemented)
  of a 4-stage pipeline — query parsing → graph traversal → graph-guided vector retrieval → answer
  generation — that would combine Parts 1 and 2: the graph supplies exact, exhaustive structural answers
  (set intersections, frequency comparisons, co-occurrence) that a k-sized vector-search sample can't
  reliably produce, while vector search still supplies the quotable natural-language excerpts a graph alone
  can't.

### Key decisions & trade-offs

- **Two vector-search backends side by side (FAISS + turbovec)**, purely to compare an exact index against a
  quantized one at this corpus size — turbovec's compression doesn't pay for itself yet at a few thousand
  chunks, but the demo shows near-identical retrieval quality between the two, validating the quantized path
  for when the corpus scales up.
- **Embeddings and the knowledge graph run 100% locally**; only the final answer-generation step is
  optional/LLM-backed, with automatic mock-mode fallback on any failure (missing key, network error, rate
  limit) — this directly satisfies the "degrade gracefully" requirement.
- **`Company` nodes are keyed by normalized name, not `company_id`**, since `company_id` is frequently
  missing in the sample data. Postings with a missing `company_name` get **no** `POSTED_BY` edge at all,
  rather than being merged into a fake shared "Unknown company" node — an early bug (caught and fixed during
  testing) that was silently fabricating false "bridging companies" in the demo query.
- **`Role` and `Skill` extraction is plain keyword/regex matching**, not NER or an LLM call — fast,
  transparent, and fully auditable, at the cost of some false positives/negatives on ambiguous short tokens
  (documented inline next to each vocabulary in the notebook). An hybrid approach (Regex + Name Entity Recognition) could be an interesting improvement.
- **`Industry` and `Occupation` (ESCO/O*NET) nodes were deliberately not implemented** — no reliable join key
  exists between `sample_industries.csv` and postings/companies in this sample, and taxonomy enrichment is
  explicitly optional and out of scope for the target questions.

Full rationale for every decision — including why each piece of the suggested graph schema was or wasn't
implemented — is documented inline in the notebook's markdown cells, directly next to the code it explains.

### Demonstrated results (from the sample data)

- RAG index: 16,378 chunks from all 4,072 postings, 384-dim embeddings.
- Knowledge graph: 6,757 nodes (4,072 postings, 1,974 companies, 646 locations, 54 skills, 11 roles) and
  33,351 edges.
- Demo query: 12 companies (e.g. Capital One, TikTok, Akkodis, Booz Allen Hamilton) hire for both `Data
  Engineer` and `Machine Learning Engineer` roles in this sample; Python, SQL, AWS, Machine Learning, and
  Azure are the most common skills connecting the two role types.

---

## Data

Sample data lives in `data/` — a small subset of the
[LinkedIn Job Postings dataset on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings).
See that Kaggle page for the dataset's license/usage terms. 