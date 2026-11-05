FROM python:3.11-slim

# Install minimal runtime library for OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy application code and static folder into container
COPY --chown=user . .

# Run FastAPI backend using Uvicorn (Render supplies a PORT variable; default to 7860)[cite: 2]
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
