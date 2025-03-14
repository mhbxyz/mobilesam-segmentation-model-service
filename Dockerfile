FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends git

ADD . /app
WORKDIR /app

RUN uv sync --frozen
EXPOSE 8000

CMD ["uv", "run", "python", "-m", "mobilesam_segmentation"]
