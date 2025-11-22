include common.mk

export OPENAI_BASE_URL := https://api.moonshot.ai/v1
export OPENAI_API_KEY = $(strip $(shell cat api.key))
export MODEL := kimi-k2-thinking
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export ABSURD_DATABASE_URL := postgresql://absurd:absurd@127.0.0.1:5432/absurd?sslmode=disable

.PHONY: paper evals worker spawn absurd cleanabsurd absurdlogs play-llm

paper: .venv/ api.key
	uv run python paper.py
	$(call success)

evals: .venv/ api.key
	uv run python evals.py
	$(call success)

worker: .venv/
	uv run absurd_worker.py

spawn: .venv/ .venv/ .venv/ .venv/
	uv run spawn_stub.py

play-llm: .venv/ api.key
	uv run python -c 'from paper import play_llm_vs_llm_once; play_llm_vs_llm_once()'
	$(call success)

api.key:
	$(error 'error missing api.key')

absurd:
	docker compose -f local_infra/docker-compose.yml build
	docker compose -f local_infra/docker-compose.yml up -d absurd
	$(call success)

cleanabsurd:
	docker compose -f local_infra/docker-compose.yml down
	$(call success)

absurdlogs:
	docker compose -f local_infra/docker-compose.yml logs -f absurd
	$(call success)
