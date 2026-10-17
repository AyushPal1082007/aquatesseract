SHELL := /bin/bash
PY := ml/.venv/bin/python

.PHONY: help setup mock up down logs test lint check dev-server dev-client clean-mock

help:  ## list commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n",$$1,$$2}'

setup:  ## one-shot local install (node deps + python venv)
	@test -f .env || cp .env.example .env
	cd server && npm ci
	cd client && npm ci
	python3.11 -m venv ml/.venv || python3 -m venv ml/.venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r ml/requirements.txt
	$(PY) -m pip install -e ml
	@echo "ML members: also install torch for YOUR hardware, see ml/requirements-torch.txt"

mock:  ## generate mock run folder (same format as real runs)
	$(PY) ml/scripts/make_mock_run.py --out results

up:  ## full stack in docker (db + api + web)
	@test -f .env || cp .env.example .env
	docker compose up --build

down:  ## stop docker stack
	docker compose down

logs:  ## tail logs
	docker compose logs -f

dev-server:  ## api without docker (needs postgres up: docker compose up db)
	cd server && npm run dev

dev-client:  ## web without docker
	cd client && npm run dev

test:  ## all tests
	cd server && npm test
	$(PY) -m pytest ml/tests -q

lint:  ## lint everything
	cd server && npm run typecheck
	cd client && npm run typecheck
	$(PY) -m ruff check ml

check:  ## validate every run folder against data contract
	$(PY) ml/scripts/validate_run.py results/*/

clean-mock:  ## delete mock runs
	rm -rf results/MOCK_*
