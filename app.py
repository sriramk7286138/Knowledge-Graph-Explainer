import re
import json
import threading
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Load model only once when server starts
print("Loading SmolLM2 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading SmolLM2 model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

print("SmolLM2 loaded successfully!")


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="Knowledge Graph Extraction",
    version="1.0"
)

MAX_SENTENCES = 5
MAX_CHARS = 800

generation_lock = threading.Lock()


# ==========================================
# REQUEST MODEL
# ==========================================

class Query(BaseModel):
    text: str


# ==========================================
# SENTENCE SPLITTING
# ==========================================

def split_sentences(text: str):

    parts = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


# ==========================================
# EXTRACTION PROMPT
# ==========================================

def create_prompt(text):

    return f"""
You are an intelligent knowledge graph extraction system.

Read the complete document carefully.

Extract entities and factual relationships from
the entire document.

IMPORTANT INSTRUCTIONS:

1. Identify all important entities.
2. Identify relationships between entities.
3. Discover relationship labels automatically.
4. Do not use predefined relationship categories.
5. Resolve pronouns such as he, she, it, they,
   his, her and the company using document context.
6. Extract relationships across different sentences.
7. Replace pronouns with their actual entity names.
8. Use the original entity names consistently.
9. Do not invent relationships.
10. Avoid duplicate triples.
11. Extract only facts supported by the document.

Return ONLY valid JSON.

Use this exact structure:

{{
    "entities": [
        {{
            "name": "Elon Musk",
            "type": "PERSON"
        }}
    ],
    "triples": [
        {{
            "head": "Elon Musk",
            "type": "founded",
            "tail": "SpaceX",
            "sentence_index": 1,
            "evidence": "Elon Musk founded SpaceX."
        }}
    ]
}}

DOCUMENT:

{text}

JSON:
"""


# ==========================================
# MODEL GENERATION
# ==========================================

def generate_response(text):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert document-level "
                "open information extraction system. "
                "Return valid JSON only."
            )
        },
        {
            "role": "user",
            "content": create_prompt(text)
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    with generation_lock:

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    raw = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return raw.strip()


# ==========================================
# JSON PARSER
# ==========================================

def parse_json(raw):

    raw = raw.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Model did not return valid JSON.")

    return json.loads(
        raw[start:end + 1]
    )


# ==========================================
# ENTITY TYPE
# ==========================================

def entity_type(name: str, entities):

    n = name.lower().strip()

    for e in entities:

        if e["name"].lower().strip() == n:
            return e["type"]

    if any(ch.isdigit() for ch in n):
        return "NUM"

    for e in entities:

        t = e["name"].lower().strip()

        if n in t or t in n:
            return e["type"]

    return "OTHER"


# ==========================================
# CLEAN ENTITIES
# ==========================================

def clean_entities(entities):

    unique = {}

    for e in entities:

        name = str(e.get("name", "")).strip()

        if not name:
            continue

        key = name.lower()

        if key not in unique:
            raw_type = str(e.get("type", "OTHER")).upper()
            if raw_type in ["COMPANY", "ORGANISATION", "ORGANIZATION", "ORG"]:
                etype = "ORG"
            elif raw_type in ["PERSON", "PER"]:
                etype = "PER"
            elif raw_type in ["LOCATION", "PLACE", "LOC"]:
                etype = "LOC"
            elif raw_type in ["DATE", "NUMBER", "NUM"]:
                etype = "NUM"
            elif raw_type in ["MISC"]:
                etype = "MISC"
            else:
                etype = "OTHER"

            unique[key] = {
                "name": name,
                "type": etype
            }

    return list(unique.values())


# ==========================================
# CLEAN TRIPLES
# ==========================================

def clean_triples(triples, entities, sentences):

    all_triples = []
    seen = set()

    for t in triples:

        head = str(t.get("head", "")).strip()
        tail = str(t.get("tail", "")).strip()
        relation = str(t.get("type", "")).strip()

        if not head or not tail or not relation:
            continue

        if head.lower() == tail.lower():
            continue

        key = (
            head.lower(),
            relation.lower(),
            tail.lower()
        )

        if key in seen:
            continue

        seen.add(key)

        idx = t.get("sentence_index")

        if not isinstance(idx, int) or not (1 <= idx <= len(sentences)):
            idx = None

        evidence = str(
            t.get("evidence", "")
        ).strip()

        # Preserve original response fields
        all_triples.append({

            "head": head,

            "type": relation,

            "tail": tail,

            "head_type": entity_type(
                head,
                entities
            ),

            "tail_type": entity_type(
                tail,
                entities
            ),

            "sentence_index": idx,

            "sentence": evidence

        })

    return all_triples


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": str(model.device)
    }


# ==========================================
# MAIN KNOWLEDGE GRAPH API
# ==========================================

@app.post("/api/build")
def build_graph(q: Query):

    text = q.text.strip()

    sentences = split_sentences(text)

    if not sentences:

        raise HTTPException(
            status_code=400,
            detail="Please type a sentence."
        )

    if len(sentences) > MAX_SENTENCES:

        raise HTTPException(
            status_code=400,
            detail=f"Please use at most {MAX_SENTENCES} sentences."
        )

    if len(text) > MAX_CHARS:

        raise HTTPException(
            status_code=400,
            detail=f"Please keep it under {MAX_CHARS} characters."
        )

    try:

        # ==================================
        # TOKENIZATION
        # ==================================

        results = []

        for idx, sentence in enumerate(
            sentences,
            start=1
        ):

            tokens = tokenizer.tokenize(sentence)

            results.append({

                "index": idx,

                "text": sentence,

                "tokens": tokens,

                "raw": "",

                "entities": []

            })


        # ==================================
        # DOCUMENT-LEVEL EXTRACTION
        # ==================================

        raw = generate_response(text)

        parsed = parse_json(raw)


        # ==================================
        # ENTITY EXTRACTION
        # ==================================

        entities = clean_entities(
            parsed.get("entities", [])
        )


        # ==================================
        # RELATIONSHIP EXTRACTION
        # ==================================

        all_triples = clean_triples(
            parsed.get("triples", []),
            entities,
            sentences
        )


        # ==================================
        # SENTENCE RESULTS
        # ==================================

        for result in results:

            result["raw"] = raw

            result["entities"] = [

                {
                    "text": e["name"],
                    "type": e["type"]
                }

                for e in entities

                if e["name"].lower() in result["text"].lower()
            ]


        # ==================================
        # FINAL RESPONSE
        # ==================================

        return {

            "sentences": results,

            "triples": all_triples,

            "entities": entities,

            "total_entities": len(entities),

            "total_relationships": len(all_triples)

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Extraction error: {str(e)}"
        )


# ==========================================
# APPLICATION START
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860
    )