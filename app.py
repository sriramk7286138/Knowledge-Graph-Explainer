import os
import re
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from llama_cpp import Llama


# ==========================================
# SETTINGS (can be changed with environment variables on Render)
# ==========================================

MODEL_FILE = os.environ.get("MODEL_FILE", "smollm2-360m-instruct-q4_k_m.gguf")
MODEL_DIR = os.environ.get("MODEL_DIR", "/home/user/models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

N_CTX = int(os.environ.get("N_CTX", "768"))          # smaller = less RAM
N_THREADS = int(os.environ.get("N_THREADS", "2"))
MAX_SENTENCES = int(os.environ.get("MAX_SENTENCES", "5"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "800"))
MAX_SENTENCE_CHARS = 300

print("Loading model:", MODEL_PATH)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=128,          # small batches use less memory
    use_mmap=True,
    use_mlock=False,
    verbose=False,
)
print("Model loaded.")

app = FastAPI(title="Knowledge Graph Explorer")
lock = threading.Lock()   # the model can only do one job at a time


class Query(BaseModel):
    text: str


# ==========================================
# PROMPT (the "instructions" we give the model)
# ==========================================

SYSTEM = (
    "You extract knowledge-graph facts from text. "
    "Write one fact per line in this format: "
    "HEAD | RELATION | TAIL | HEAD_TYPE | TAIL_TYPE. "
    "Types are PER (person), ORG (organisation), LOC (place), NUM (date or number), MISC (other). "
    "If the sentence uses a pronoun such as he, she, it, they, his, her or its, "
    "replace it with the real name from KNOWN ENTITIES. "
    "Use short relation words. Output only the fact lines, or NONE if there are no facts."
)

FEW_SHOT = [
    ("user", "KNOWN ENTITIES: none\nSENTENCE: Marie Curie was born in Warsaw in 1867.\nFACTS:"),
    ("assistant", "Marie Curie | born in | Warsaw | PER | LOC\nMarie Curie | born in | 1867 | PER | NUM"),
    ("user", "KNOWN ENTITIES: Marie Curie (PER); Warsaw (LOC)\nSENTENCE: She won the Nobel Prize in Physics.\nFACTS:"),
    ("assistant", "Marie Curie | won | Nobel Prize in Physics | PER | MISC"),
]


def chatml(messages):
    """SmolLM2 and Qwen both understand this 'chat' text format."""
    text = ""
    for role, content in messages:
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def ask_model(sentence, known):
    entities = "; ".join(
        f"{name} ({etype})" for name, etype in list(known.items())[-6:] if etype != "NUM"
    ) or "none"

    messages = [("system", SYSTEM)] + FEW_SHOT + [
        ("user", f"KNOWN ENTITIES: {entities}\nSENTENCE: {sentence}\nFACTS:")
    ]

    with lock:
        out = llm(
            chatml(messages),
            max_tokens=120,
            temperature=0.0,
            repeat_penalty=1.0,
            stop=["<|im_end|>", "\n\n"],
        )
    return out["choices"][0]["text"].strip()


# ==========================================
# HELPERS
# ==========================================

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def show_tokens(sentence):
    """The pieces (tokens) the model sees, for the 'tokenization' step."""
    ids = llm.tokenize(sentence.encode("utf-8"), add_bos=False, special=False)
    tokens = []
    for i in ids:
        piece = llm.detokenize([i]).decode("utf-8", errors="ignore").strip()
        if piece:
            tokens.append(piece)
    return tokens


TYPE_MAP = {
    "PER": "PER", "PERSON": "PER",
    "ORG": "ORG", "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "COMPANY": "ORG",
    "LOC": "LOC", "LOCATION": "LOC", "PLACE": "LOC", "GPE": "LOC", "CITY": "LOC", "COUNTRY": "LOC",
    "NUM": "NUM", "DATE": "NUM", "NUMBER": "NUM", "YEAR": "NUM", "TIME": "NUM",
    "MISC": "MISC",
}


def norm_type(t):
    return TYPE_MAP.get(str(t).strip().upper(), "OTHER")


PERSON_PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
THING_PRONOUNS = {"it", "its", "they", "them", "their", "this", "that", "the company", "the firm"}


def resolve_pronoun(name, known):
    """Backup rule: replace he/she/it/they with the latest matching entity."""
    low = name.lower().strip()
    if low in PERSON_PRONOUNS:
        wanted = ["PER"]
    elif low in THING_PRONOUNS:
        wanted = ["ORG", "LOC", "MISC", "OTHER"]
    else:
        return name
    for k in reversed(list(known)):
        if known[k] in wanted:
            return k
    return None   # could not resolve -> the fact will be dropped


def canonical(name, known):
    """Make the same thing always have the same name ("Musk" -> "Elon Musk")."""
    n = re.sub(r"^(the|a|an)\s+", "", name.strip(), flags=re.I).strip() or name.strip()
    low = n.lower()
    for k in known:
        if k.lower() == low:
            return k
    if len(n.split()) == 1:
        for k in reversed(list(known)):
            if low in k.lower().split():
                return k
    return n


def fix_type(name, etype, known):
    if name in known and known[name] != "OTHER":
        return known[name]
    if any(ch.isdigit() for ch in name):
        return "NUM"
    return etype


def touch(known, name, etype):
    """Remember an entity and mark it as the most recent one."""
    old = known.pop(name, None)
    if old and old != "OTHER" and etype == "OTHER":
        etype = old
    known[name] = etype


def parse_facts(raw):
    facts = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if "|" not in line:
            continue
        parts = [p.strip(" .\"'") for p in line.split("|")]
        if len(parts) < 3:
            continue
        head, rel, tail = parts[0], parts[1], parts[2]
        head_type = parts[3] if len(parts) > 3 else ""
        tail_type = parts[4] if len(parts) > 4 else ""
        if not head or not rel or not tail:
            continue
        if head.lower() in ("head", "subject") or rel.lower() in ("relation", "relationship"):
            continue   # the model copied the template by mistake
        if max(len(head), len(rel), len(tail)) > 60:
            continue
        facts.append((head, rel, tail, head_type, tail_type))
    return facts


def memory_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024)
    except Exception:
        pass
    return None


# ==========================================
# ROUTES
# ==========================================

@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "running", "model": MODEL_FILE, "memory_used_mb": memory_mb()}


@app.post("/api/build")
def build_graph(q: Query):
    text = q.text.strip()
    sentences = split_sentences(text)

    if not sentences:
        raise HTTPException(status_code=400, detail="Please type a sentence.")
    if len(sentences) > MAX_SENTENCES:
        raise HTTPException(status_code=400, detail=f"Please use at most {MAX_SENTENCES} sentences.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Please keep it under {MAX_CHARS} characters.")
    if any(len(s) > MAX_SENTENCE_CHARS for s in sentences):
        raise HTTPException(status_code=400, detail=f"Each sentence must be under {MAX_SENTENCE_CHARS} characters.")

    try:
        known = {}            # entities remembered from earlier sentences
        results = []
        all_triples = []
        all_entities = {}
        seen = set()

        for idx, sentence in enumerate(sentences, start=1):
            raw = ask_model(sentence, known)
            sentence_entities = {}

            for head, rel, tail, head_type, tail_type in parse_facts(raw):
                head = resolve_pronoun(head, known)
                tail = resolve_pronoun(tail, known)
                if head is None or tail is None:
                    continue

                head = canonical(head, known)
                tail = canonical(tail, known)
                if head.lower() == tail.lower():
                    continue

                head_type = fix_type(head, norm_type(head_type), known)
                tail_type = fix_type(tail, norm_type(tail_type), known)

                key = (head.lower(), rel.lower(), tail.lower())
                if key in seen:
                    continue
                seen.add(key)

                touch(known, head, head_type)
                touch(known, tail, tail_type)
                sentence_entities[head] = head_type
                sentence_entities[tail] = tail_type
                all_entities[head] = head_type
                all_entities[tail] = tail_type

                # Was a name worked out from earlier sentences?
                low_sentence = sentence.lower()
                missing = [n for n in (head, tail) if n.lower() not in low_sentence]
                note = ""
                if missing:
                    note = (f'This sentence does not write "{missing[0]}" in full. '
                            f"The AI worked it out from the earlier sentences.")

                all_triples.append({
                    "head": head,
                    "type": rel,
                    "tail": tail,
                    "head_type": head_type,
                    "tail_type": tail_type,
                    "sentence_index": idx,
                    "sentence": sentence,
                    "note": note,
                })

            results.append({
                "index": idx,
                "text": sentence,
                "tokens": show_tokens(sentence),
                "raw": raw or "NONE",
                "entities": [{"text": n, "type": t} for n, t in sentence_entities.items()],
            })

        return {
            "sentences": results,
            "triples": all_triples,
            "entities": [{"name": n, "type": t} for n, t in all_entities.items()],
            "total_entities": len(all_entities),
            "total_relationships": len(all_triples),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
