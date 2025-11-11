include common.mk

export OPENAI_BASE_URL := https://api.moonshot.ai/v1
export OPENAI_API_KEY = $(strip $(shell cat api.key))
export MODEL := kimi-k2-thinking-turbo
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

.PHONY: paper evals

paper: .venv/ api.key
	uv run python paper.py
	$(call success)

evals: .venv/ api.key
	uv run python evals.py
	$(call success)

api.key:
	$(error 'error missing api.key')
