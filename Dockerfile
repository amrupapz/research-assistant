# syntax=docker/dockerfile:1
# D7 M4. Two stages: build the venv with uv, then copy ONLY the venv + app into a clean image.
# No secret is ever an ARG/ENV here: they arrive at RUN time (compose env_file). .env is in .dockerignore.
# Base images pinned by digest (multi-arch index); the tag is just for humans. Bump both together.

FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS build
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is cached until uv.lock changes, so a code edit doesn't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
# Same base image + same /app path as the build stage, so the venv's python symlink and the
# editable install (.pth -> /app/src) still resolve. Code files stay root-owned: the app can't edit them.
COPY --from=build /app/.venv ./.venv
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini icp.toml ./
COPY data ./data
RUN mkdir runs outbox && chown app:app runs outbox
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
# 0.0.0.0 INSIDE the container (127.0.0.1 there is unreachable from outside); compose publishes it
# on the host's 127.0.0.1 only.
CMD ["uvicorn", "research_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
