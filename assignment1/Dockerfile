# Dockerfile
FROM python:3.12-slim-bookworm

# Install tools needed to install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (your project uses uv to manage deps)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Workdir in the container
WORKDIR /code

# Copy manifests first for better build caching
COPY pyproject.toml uv.lock /code/

# Install dependencies exactly as locked
RUN uv sync --frozen

# Install the spaCy model into the image (needed by /embed)
RUN uv run python -m spacy download en_core_web_lg

# Copy your application code
COPY ./app /code/app

# Optional: document the exposed port
EXPOSE 80

# Run the app on 0.0.0.0 so it's reachable from host
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]