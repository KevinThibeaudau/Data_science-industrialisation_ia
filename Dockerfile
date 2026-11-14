
FROM python:3.13-slim

# Copier uv depuis son image officielle
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

WORKDIR /app

# Utiliser le Python déjà présent dans l'image
ENV UV_PYTHON_DOWNLOADS=0

# Copier la configuration des dépendances
COPY pyproject.toml uv.lock ./

# Installer uniquement les dépendances de production
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Copier l'application et le modèle
COPY app/ ./app/
COPY frequence/ ./frequence/
COPY artifacts/ ./artifacts/

# Utiliser le Python de l'environnement virtuel créé par uv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
