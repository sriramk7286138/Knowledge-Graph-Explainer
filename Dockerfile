FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B-Instruct'); AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B-Instruct')"

COPY --chown=user . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]