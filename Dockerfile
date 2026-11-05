FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user . .

# Expose port 8080
EXPOSE 8080

# Bind Uvicorn to port 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080"]
