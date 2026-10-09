# One image for both workloads (weekly batch job and web UI); the ECS task
# decides which command runs. Ordered for layer caching: dependencies are
# installed before the source is copied, so a code edit does not reinstall pandas.

# TODO 1: base image. Use the official slim Python matching pyproject's
#   requires-python. Form:  FROM python:<version>-slim-bookworm
#   docs: hub.docker.com/_/python  (why slim and not alpine? ask me)
FROM python:3.12-slim-bookworm


# The uv binary, copied straight from Astral's image (no pip install, no curl).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

# Compile .pyc at install time and never download a different Python at runtime.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# TODO 2: working directory inside the container. Form:  WORKDIR /app
WORKDIR /app


# TODO 3: copy ONLY the two files that define dependencies, then install them.
#   COPY pyproject.toml uv.lock ./
#   RUN uv sync --locked --no-dev --no-install-project
#   (--no-install-project: install deps but not fpl_model itself — src isn't copied yet.
#    --locked: fail if uv.lock is out of date rather than silently resolving.)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# TODO 4: now copy the source and the frozen reference data, then install the project.
#   COPY src/ src/
#   COPY data/reference/ data/reference/
#   COPY data/priors.yaml data/priors.yaml
#   RUN uv sync --locked --no-dev
COPY src/ src/
COPY data/reference/ data/reference/
COPY data/priors.yaml data/priors.yaml
RUN uv sync --locked --no-dev


# Put the venv's bin on PATH so `fpl` and `python` resolve to it without `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

# TODO 5: default command. The ECS task overrides this per workload; a safe
#   default is the CLI's help. Form:  CMD ["fpl", "--help"]
CMD ["fpl", "--help"]

