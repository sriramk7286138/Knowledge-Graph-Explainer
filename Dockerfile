# ---------- Stage 1: compile llama-cpp-python (needs a C++ compiler) ----------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# NATIVE=OFF makes the build work on ANY CPU (the build machine and the
# machine that runs the website are not always the same)
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_BLAS=OFF"
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels


# ---------- Stage 2: small runtime image ----------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    MODEL_REPO="mfuntowicz/SmolLM2-360M-Instruct-Q4_K_M-GGUF" \
    MODEL_FILE="smollm2-360m-instruct-q4_k_m.gguf" \
    MODEL_DIR="/home/user/models"

# Download the model while building, so the website starts fast
RUN python -c "import os; from huggingface_hub import hf_hub_download; hf_hub_download(os.environ['MODEL_REPO'], os.environ['MODEL_FILE'], local_dir=os.environ['MODEL_DIR'])"

COPY --chown=user . .

# Render gives us a PORT; if it is missing we use 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
