"""Knowledge Graph Explorer - backend (Qwen2.5-0.5B version).

One small language model reads the WHOLE text at once and writes the facts
as JSON:  subject - relation - object.  Because it sees all the sentences
together, it can work out who "She" or "the scientist" is.

Safety nets (a 0.5B model sometimes makes things up):
  * the JSON format is forced by a grammar, so the output can never be broken;
  * every name must really appear in the text, otherwise the fact is dropped;
  * the sentence number is checked by looking where the object really is.
"""
import json
import os
import re

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from llama_cpp import Llama

# Official GGUF file of Qwen2.5-0.5B-Instruct (about 420 MB).
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q5_k_m.gguf",
    n_ctx=2048,
    n_threads=os.cpu_count() or 2,
    verbose=False,
)

app = FastAPI(title="Knowledge Graph Explorer")

TYPES = ["person", "place", "organization", "thing"]
LABEL_MAP = {"person": "PER", "organization": "ORG", "place": "LOC", "thing": "CONCEPT"}
FACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "sentence": {"type": "integer"},
            "subject": {"type": "string"},
            "subject_type": {"type": "string", "enum": TYPES},
            "relation": {"type": "string"},
            "object": {"type": "string"},
            "object_type": {"type": "string", "enum": TYPES},
        },
        "required": ["sentence", "subject", "subject_type", "relation", "object", "object_type"],
    }}},
    "required": ["facts"],
}

SYSTEM = (
    "You extract facts from numbered sentences to build a knowledge graph. "
    "Return JSON with a list called facts. Each fact has: sentence (number), "
    "subject, subject_type, relation, object, object_type. "
    "Rules: 1) The subject must be a full name copied from the text, never a pronoun "
    "(he, she, it, they) and never a description like 'the scientist' - replace those "
    "with the full name they refer to. 2) The object must be copied exactly from the text. "
    "3) The relation is 1 to 3 lowercase words, like 'born in' or 'discovered'. "
    "4) Only include facts that the text clearly states. "
    "5) Types are person, place, organization or thing."
)
EXAMPLE_IN = ("1. Alan Turing was born in London.\n2. He studied at Cambridge.\n"
              "3. The mathematician later worked for the British government.")
EXAMPLE_OUT = json.dumps({"facts": [
    {"sentence": 1, "subject": "Alan Turing", "subject_type": "person", "relation": "born in",
     "object": "London", "object_type": "place"},
    {"sentence": 2, "subject": "Alan Turing", "subject_type": "person", "relation": "studied at",
     "object": "Cambridge", "object_type": "organization"},
    {"sentence": 3, "subject": "Alan Turing", "subject_type": "person", "relation": "worked for",
     "object": "British government", "object_type": "organization"}]})

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SUBJECT_PRONOUN = re.compile(r"\b(he|she|it|they)\b", re.I)
PRONOUN_WORDS = {"he", "she", "it", "they", "him", "her", "his", "its", "them", "their", "i", "we", "you"}


class Req(BaseModel):
    text: str


def find_in(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I)


def locate(sentence, phrase):
    m = find_in(sentence, phrase) if phrase else None
    return [m.start(), m.end()] if m else None


def ask_model(sentences):
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": EXAMPLE_IN},
                  {"role": "assistant", "content": EXAMPLE_OUT},
                  {"role": "user", "content": numbered}],
        response_format={"type": "json_object", "schema": FACT_SCHEMA},
        temperature=0, max_tokens=600)
    try:
        return json.loads(out["choices"][0]["message"]["content"]).get("facts", [])
    except (ValueError, KeyError, IndexError):
        return []


def clean_facts(raw, sentences):
    """Keep only facts whose names really appear in the text."""
    full = " ".join(sentences)
    good, seen = [], set()
    for f in raw:
        s = re.sub(r"\s+", " ", str(f.get("subject", ""))).strip()
        o = re.sub(r"\s+", " ", str(f.get("object", ""))).strip()
        rel = re.sub(r"\s+", " ", str(f.get("relation", "")).replace("_", " ")).strip().lower()
        if not s or not o or not rel or len(rel) > 40 or s.lower() == o.lower():
            continue
        if s.lower() in PRONOUN_WORDS or o.lower() in PRONOUN_WORDS:
            continue
        if not find_in(full, o):
            continue
        if not find_in(full, s) and not any(find_in(full, w) for w in s.split() if len(w) > 2 and w[0].isupper()):
            continue
        cands = [i for i, sen in enumerate(sentences) if find_in(sen, o)]
        guess = int(f.get("sentence", 0)) - 1 if isinstance(f.get("sentence"), int) else -1
        idx = guess if guess in cands else cands[0]
        key = (s.lower(), rel, o.lower())
        if key in seen:
            continue
        seen.add(key)
        good.append({"s": s, "st": f.get("subject_type"), "p": rel, "o": o,
                     "ot": f.get("object_type"), "idx": idx})
    return good


@app.post("/api/graph")
def build_graph(req: Req):
    text = req.text.strip()[:1500]
    originals = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()][:10]
    if not originals:
        return {"sentences": [], "nodes": [], "edges": []}

    facts = clean_facts(ask_model(originals), originals)

    # "Curie" and "Marie Curie" are the same person -> use the longer name.
    people = {}
    for f in facts:
        for name, t in ((f["s"], f["st"]), (f["o"], f["ot"])):
            if t == "person":
                people[name.lower()] = name
    alias = {}
    for short in people:
        for longer in people:
            if short != longer and set(short.split()) < set(longer.split()):
                if short not in alias or len(longer) > len(alias[short]):
                    alias[short] = longer

    def name_of(t):
        low = t.lower()
        return people.get(alias.get(low, low), t)

    # Replace "She" with the full name, so the sentence says who does the action.
    sentences = []
    for i, orig in enumerate(originals):
        group = [f for f in facts if f["idx"] == i]
        reps = []
        for f in group:
            if find_in(orig, f["s"]) or any(r[2].lower() == name_of(f["s"]).lower() for r in reps):
                continue
            for m in SUBJECT_PRONOUN.finditer(orig):
                if all(m.start() != r[0] for r in reps):
                    reps.append((m.start(), m.end(), name_of(f["s"])))
                    break
        reps.sort()
        new, pos, corefs = "", 0, []
        for a, b, name in reps:
            new += orig[pos:a]
            ns = len(new)
            new += name
            corefs.append({"word": orig[a:b], "start": a, "end": b,
                           "new_start": ns, "new_end": ns + len(name), "refers_to": name})
            pos = b
        new += orig[pos:]
        sentences.append({"index": i, "original": orig, "text": new, "entities": [], "corefs": corefs})

    nodes, edges, seen = {}, [], set()

    def add_node(t, typ):
        label = LABEL_MAP.get(typ, "MISC")
        name = name_of(t)
        key = name.lower()
        if key not in nodes:
            nodes[key] = {"id": key, "text": name, "label": label}
        elif nodes[key]["label"] in ("CONCEPT", "MISC") and label not in ("CONCEPT", "MISC"):
            nodes[key]["label"] = label
        return key

    for f in facts:
        s = sentences[f["idx"]]
        ss = locate(s["text"], name_of(f["s"])) or locate(s["text"], f["s"])
        os_ = locate(s["text"], f["o"])
        for name, typ, sp in ((f["s"], f["st"], ss), (f["o"], f["ot"], os_)):
            if sp and not any(e["start"] == sp[0] for e in s["entities"]):
                s["entities"].append({"text": s["text"][sp[0]:sp[1]],
                                      "label": LABEL_MAP.get(typ, "MISC"), "start": sp[0], "end": sp[1]})
        a, b = add_node(f["s"], f["st"]), add_node(f["o"], f["ot"])
        for c in s["corefs"]:
            c["refers_to"] = name_of(c["refers_to"])
        if a == b or (a, f["p"], b) in seen:
            continue
        seen.add((a, f["p"], b))
        verb = None
        if ss and os_:
            first, second = (ss, os_) if ss[0] < os_[0] else (os_, ss)
            gap = s["text"][first[1]:second[0]]
            lead = len(gap) - len(gap.lstrip())
            if 0 < len(gap.strip()) <= 40:
                verb = [first[1] + lead, first[1] + lead + len(gap.strip())]
        edges.append({"source": a, "target": b, "label": f["p"], "sentence": f["idx"],
                      "evidence": {"subject": ss, "verb": verb, "object": os_}})

    for s in sentences:
        s["entities"].sort(key=lambda e: e["start"])
    return {"sentences": sentences, "nodes": list(nodes.values()), "edges": edges}


@app.get("/")
def index():
    return FileResponse("static/index.html")
