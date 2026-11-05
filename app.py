import os
import re
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# SETTINGS
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Qwen 2.5 model via Transformers...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float32, 
    device_map="cpu"
)
print("Model loaded.")

app = FastAPI(title="Knowledge Graph Explorer")
lock = threading.Lock()

class Query(BaseModel):
    text: str

SYSTEM = (
    "Extract knowledge graph facts from the text. "
    "Output ONLY lines in this exact pipe-separated format, with no extra text, markdown code blocks, or headers: "
    "HEAD | RELATION | TAIL | HEAD_TYPE | TAIL_TYPE. "
    "Types must be one of: PER, ORG, LOC, NUM, MISC. "
    "Use short relation words. If no facts, output NONE.\n\n"
    "Example 1:\n"
    "KNOWN ENTITIES: none\n"
    "SENTENCE: Elon Musk founded SpaceX in 2002.\n"
    "Elon Musk | founded | SpaceX | PER | ORG\n"
    "SpaceX | started in | 2002 | ORG | NUM\n\n"
    "Example 2:\n"
    "KNOWN ENTITIES: Albert Einstein (PER)\n"
    "SENTENCE: Albert Einstein developed the theory of relativity in Germany.\n"
    "Albert Einstein | developed | theory of relativity | PER | MISC\n"
    "Albert Einstein | in | Germany | PER | LOC"
)

def ask_model(sentence, known):
    entities = "; ".join(
        f"{name} ({etype})" for name, etype in list(known.items())[-6:] if etype != "NUM"
    ) or "none"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"KNOWN ENTITIES: {entities}\nSENTENCE: {sentence}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    with lock:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/build")
def build_graph(query: Query):
    text = query.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Split into sentences using simple regex
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text]

    sentences_data = []
    triples = []
    known_entities = {}

    for idx, sent in enumerate(raw_sentences, start=1):
        tokens = re.findall(r'\w+|[.,!?;]', sent)
        
        raw_output = ask_model(sent, known_entities)

        sent_triples = []
        sent_entities = []

        for line in raw_output.splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            
            # Guard against prompt template echoes
            if "HEAD" in line.upper() and "RELATION" in line.upper():
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 5:
                head, relation, tail, head_type, tail_type = parts
                head_type = head_type.upper()
                tail_type = tail_type.upper()

                if head_type not in ["PER", "ORG", "LOC", "NUM", "MISC"]:
                    head_type = "MISC"
                if tail_type not in ["PER", "ORG", "LOC", "NUM", "MISC"]:
                    tail_type = "MISC"

                known_entities[head] = head_type
                known_entities[tail] = tail_type

                t_obj = {
                    "head": head,
                    "type": relation,
                    "tail": tail,
                    "head_type": head_type,
                    "tail_type": tail_type,
                    "sentence_index": idx,
                    "sentence": sent
                }
                sent_triples.append(t_obj)
                triples.append(t_obj)

        seen_ent = set()
        for t in sent_triples:
            if t["head"] not in seen_ent:
                seen_ent.add(t["head"])
                sent_entities.append({"text": t["head"], "type": t["head_type"]})
            if t["tail"] not in seen_ent:
                seen_ent.add(t["tail"])
                sent_entities.append({"text": t["tail"], "type": t["tail_type"]})

        sentences_data.append({
            "index": idx,
            "text": sent,
            "tokens": tokens,
            "raw": raw_output,
            "entities": sent_entities
        })

    return {
        "sentences": sentences_data,
        "triples": triples
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
