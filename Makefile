.DEFAULT_GOAL := help

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
API_URL  := http://$(API_HOST):$(API_PORT)

.PHONY: help install notebook notebook-execute \
        api api-bg api-stop wait-for-api health \
        demo-rag demo-kg demo demo-auto \
        clean clean-cache

help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- Setup -----------------------------------------------------------------

install: ## Install/sync dependencies (uv sync)
	uv sync

## --- Notebook ----------------------------------------------------------------

notebook: ## Launch Jupyter Lab
	uv run jupyter lab

notebook-execute: ## Re-run the whole notebook headlessly, in place (Kernel > Restart & Run All, from the CLI)
	uv run jupyter nbconvert --to notebook --execute --inplace mleng_take_home_task.ipynb

## --- API -----------------------------------------------------------------

api: ## Run the FastAPI app in the foreground (auto-reload); Swagger docs at /docs
	uv run uvicorn api.main:app --reload --host $(API_HOST) --port $(API_PORT)

api-bg: ## Run the FastAPI app in the background (logs: .api.log, pid: .api.pid)
	uv run uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) > .api.log 2>&1 & echo $$! > .api.pid

api-stop: ## Stop the background API started with `make api-bg`
	@if [ -f .api.pid ]; then kill $$(cat .api.pid) 2>/dev/null || true; rm -f .api.pid; echo "Stopped."; else echo "No .api.pid found — nothing to stop."; fi

wait-for-api: ## Block until the API's /health endpoint responds (used by demo-auto)
	@echo "Waiting for API at $(API_URL) ..."
	@i=0; while [ $$i -lt 300 ]; do \
		code=$$(curl -s -o /dev/null -w "%{http_code}" $(API_URL)/health 2>/dev/null); \
		if [ "$$code" = "200" ]; then echo "API ready."; exit 0; fi; \
		sleep 1; i=$$((i+1)); \
	done; \
	echo "Timed out waiting for API at $(API_URL) — is it running? (make api / make api-bg)"; exit 1

health: ## Check API health
	curl -s $(API_URL)/health | python3 -m json.tool

## --- Demo (requires the API running — `make api` in another shell, or use demo-auto) --------

demo-rag: ## Part 1 (Baseline RAG) demo: direct-retrieval + comparison/synthesis questions
	@echo "=== RAG demo 1/2 — direct retrieval ==="
	@echo "Q: Which postings mention retrieval-augmented generation, vector databases, or LLM application development?"
	@curl -s -X POST $(API_URL)/rag/ask -H "Content-Type: application/json" \
		-d '{"question": "Which postings mention retrieval-augmented generation, vector databases, or LLM application development?"}' \
		| python3 -m json.tool
	@echo
	@echo "=== RAG demo 2/2 — comparison / synthesis ==="
	@echo "Q: Which skills distinguish Machine Learning Engineer roles from Data Scientist roles?"
	@curl -s -X POST $(API_URL)/rag/ask -H "Content-Type: application/json" \
		-d '{"question": "Which skills distinguish Machine Learning Engineer roles from Data Scientist roles?"}' \
		| python3 -m json.tool

demo-kg: ## Part 2 (Knowledge Graph) demo: the required question (companies bridging Data Engineer / ML Engineer roles)
	@echo "=== KG demo — required question ==="
	@echo "Q: Which companies hire for both data engineering and ML engineering roles, and what skills connect them?"
	@curl -s "$(API_URL)/kg/bridge?role_a=Data%20Engineer&role_b=Machine%20Learning%20Engineer" | python3 -m json.tool

demo: demo-rag demo-kg ## Run both the RAG and knowledge-graph demos

demo-auto: api-bg wait-for-api demo api-stop ## Self-contained demo: starts the API in the background, runs demo, then stops it

## --- Cleanup -----------------------------------------------------------------

clean-cache: ## Remove the on-disk RAG/KG build cache (forces a rebuild on next API startup)
	rm -rf .cache

clean: ## Remove Python/Jupyter cache artifacts
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ipynb_checkpoints
