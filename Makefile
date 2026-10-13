.PHONY: db api web dev test
db:
	docker compose up -d --wait db
api:
	cd apps/api && uv run uvicorn livegraph.main:app --reload --port 8000
web:
	cd apps/web && npm run dev
dev: db
	$(MAKE) -j2 api web
test:
	cd apps/api && uv run pytest -q
	cd apps/web && npm run typecheck
