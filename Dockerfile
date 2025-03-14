FROM python:3.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends git

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/mobilesam_segmentation ./src/mobilesam_segmentation

RUN uv sync

COPY . ./
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "mobilesam_segmentation.app:api", "--host", "0.0.0.0", "--port", "8000"]
