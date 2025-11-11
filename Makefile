include common.mk

# Environment variables
export KIMI_API= http://hal:27000/v1
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

.PHONY: paper

paper: .venv/
	uv run python paper.py
	$(call success)

api.key:
	$(error 'error missing api.key')
