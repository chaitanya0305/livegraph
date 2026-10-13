FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
WORKDIR /app
# deps first (cached layer), then the package itself
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY livegraph ./livegraph
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# secrets (OPENROUTER_API_KEY, LOGFIRE_TOKEN) come from the environment at runtime, never baked in
CMD ["uvicorn", "livegraph.main:app", "--host", "0.0.0.0", "--port", "8000"]
