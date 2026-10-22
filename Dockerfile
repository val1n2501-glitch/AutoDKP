FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TEMPLATE_PATH=/app/resources/templates/dkp-agreement_new_logo.pdf \
    PDF_FONT_PATH=/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates fonts-liberation2 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY resources ./resources

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 autodkp \
    && chown -R autodkp:autodkp /app

USER autodkp

CMD ["python", "-m", "autodkp.cli", "bot"]
