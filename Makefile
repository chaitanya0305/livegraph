.PHONY: db api web dev test up down logs db-upgrade db-rev db-check db-stamp
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
# Alembic (apps/api/migrations; the API also runs `upgrade head` at startup). DATABASE_URL from apps/api/.env
db-upgrade:
	cd apps/api && uv run alembic upgrade head
db-rev:  # make db-rev m="add foo" (autogenerate; review the file before committing)
	cd apps/api && uv run alembic revision --autogenerate -m "$(m)"
db-check:  # models vs database: fails when a revision is missing
	cd apps/api && uv run alembic check
db-stamp:  # make db-stamp rev=0001 (mark a database as at a revision without running DDL)
	cd apps/api && uv run alembic stamp $(or $(rev),0001)
