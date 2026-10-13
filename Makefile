.PHONY: db api web dev test up down logs
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
up:
	docker compose up -d --build --wait
down:
	docker compose down
logs:
	docker compose logs -f api web
