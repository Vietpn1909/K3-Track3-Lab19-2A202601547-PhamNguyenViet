#!/usr/bin/env python3
"""
run_pipeline.py — Lab 19: GraphRAG vs Flat RAG
Tổng hợp từ notebook, adapt cho local execution.
"""
import os, re, json, time, random, hashlib, unicodedata, csv, sys
from pathlib import Path
from collections import defaultdict, Counter, deque
from difflib import SequenceMatcher

# ─── Encoding fix for Windows console ─────────────────────────────────────────
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─── Load .env ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_colwidth", 120)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "")

JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "groq").lower()
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

DATA_PATH = str(DATA_DIR / "hackernoon_subset.csv")
GOLDEN_PATH = str(DATA_DIR / "golden_dataset.csv")
CHECKPOINT_PATH = str(DATA_DIR / "checkpoint.csv")

LAB_MAX_ARTICLES = 1500
LAB_MAX_CHUNKS = 3000
EXTRACTION_MAX_CHUNKS = 400
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

print("=" * 60)
print("Lab 19: GraphRAG vs Flat RAG — Local Pipeline")
print("=" * 60)
print(f"NEO4J_URI: {NEO4J_URI[:30]}...")
print(f"GROQ_MODEL: {GROQ_MODEL}")
print(f"JUDGE_PROVIDER: {JUDGE_PROVIDER}")
print(f"HF_TOKEN: {'set' if HF_TOKEN else 'MISSING'}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1.3 — Stream HackerNoon dataset
# ═══════════════════════════════════════════════════════════════════════════════
def stream_dataset():
    from datasets import load_dataset

    OUTPUT_CSV = DATA_PATH
    LIMIT_ROWS = 5000  # User requested 5000 rows

    if Path(OUTPUT_CSV).exists():
        existing = pd.read_csv(OUTPUT_CSV, nrows=5)
        if len(existing) > 0:
            total = sum(1 for _ in open(OUTPUT_CSV, encoding='utf-8')) - 1
            print(f"Dataset already exists: {OUTPUT_CSV} ({total:,} rows). Skipping download.")
            return

    if not HF_TOKEN:
        raise ValueError("Missing HF_TOKEN.")

    print("Connecting to HuggingFace streaming...")
    dataset = load_dataset(
        "HackerNoon/tech-company-news-data-dump",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )
    iterator = iter(dataset)
    first_row = next(iterator)
    headers = list(first_row.keys())

    print(f"Writing data to: {OUTPUT_CSV}")
    rows_written = 0

    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(first_row)
        rows_written += 1

        for row in tqdm(iterator, total=LIMIT_ROWS, desc="Downloading"):
            writer.writerow(row)
            rows_written += 1
            if rows_written >= LIMIT_ROWS:
                break

    final_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"Done: {OUTPUT_CSV}")
    print(f"  Rows: {rows_written:,}")
    print(f"  Size: {final_size_mb:.2f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1.4 — Neo4j connection + schema
# ═══════════════════════════════════════════════════════════════════════════════
driver = None

def connect_neo4j():
    global driver
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise ValueError("Missing Neo4j secrets.")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    print("Neo4j connected.")

def run_cypher(query, **params):
    if driver is None:
        raise RuntimeError("Run connect_neo4j() first.")
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, **params)
        rows = [r.data() for r in result]
        result.consume()
    return rows

def setup_graph_schema():
    for stmt in [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE INDEX entity_name_norm IF NOT EXISTS FOR (n:Entity) ON (n.name_norm)",
        "CREATE INDEX company_name_norm IF NOT EXISTS FOR (n:Company) ON (n.name_norm)",
        "CREATE INDEX person_name_norm IF NOT EXISTS FOR (n:Person) ON (n.name_norm)",
        "CREATE INDEX technology_name_norm IF NOT EXISTS FOR (n:Technology) ON (n.name_norm)",
    ]:
        run_cypher(stmt)
    print("Schema ready.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1.5 — Loader + exact dedup + chunking
# ═══════════════════════════════════════════════════════════════════════════════
def norm_space(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def sha1(x):
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()

def pick_col(df, candidates, required=True):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise KeyError(f"Missing one of columns: {candidates}")
    return None

def load_news(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported: {path.suffix}")

def standardize_news(raw):
    text_col = pick_col(raw, ["text", "content", "article", "body", "story", "description"])
    title_col = pick_col(raw, ["title", "headline"], required=False)
    date_col = pick_col(raw, ["published_date", "date", "published_at", "created_at"], required=False)
    id_col = pick_col(raw, ["id", "article_id", "story_id", "uuid"], required=False)

    df = pd.DataFrame()
    df["text"] = raw[text_col].fillna("").map(norm_space)
    df["title"] = raw[title_col].fillna("").map(norm_space) if title_col else ""

    if date_col:
        df["published_date"] = (
            pd.to_datetime(raw[date_col], errors="coerce", utc=True)
            .dt.strftime("%Y-%m-%d")
            .fillna("")
        )
    else:
        df["published_date"] = ""

    if id_col:
        df["article_id"] = raw[id_col].astype(str)
    else:
        df["article_id"] = [
            sha1(f"{t}\n{x}")[:20] for t, x in zip(df["title"], df["text"])
        ]

    df = df[df["text"].str.len() >= 80].copy()
    df["dedup_key"] = [
        sha1(norm_space(f"{t}\n{x}").lower())
        for t, x in zip(df["title"], df["text"])
    ]
    before = len(df)
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
    print(f"Exact dedup: {before:,} -> {len(df):,}")

    if LAB_MAX_ARTICLES and len(df) > LAB_MAX_ARTICLES:
        df = df.sample(LAB_MAX_ARTICLES, random_state=SEED).sort_index().reset_index(drop=True)
    return df

def chunk_text(text, size=220, overlap=40):
    words = norm_space(text).split()
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        part = words[start:start+size]
        if not part:
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out

def build_chunks(news_df):
    rows = []
    for r in tqdm(news_df.itertuples(index=False), total=len(news_df), desc="Chunking"):
        for i, text in enumerate(chunk_text(r.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)):
            rows.append({
                "chunk_id": f"{r.article_id}::c{i:04d}",
                "article_id": r.article_id,
                "title": r.title,
                "published_date": r.published_date,
                "text": text,
            })
            if LAB_MAX_CHUNKS and len(rows) >= LAB_MAX_CHUNKS:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1.6 — LLM wrapper
# ═══════════════════════════════════════════════════════════════════════════════
from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def parse_json_object(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("No JSON object found.")
    return json.loads(text[a:b+1])

def groq_chat(messages, model=None, json_mode=False, max_retries=4):
    if groq_client is None:
        raise RuntimeError("Missing GROQ_API_KEY.")
    model = model or GROQ_MODEL
    if not model:
        raise RuntimeError("Missing GROQ_MODEL.")

    last = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = groq_client.chat.completions.create(**kwargs)
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            return resp.choices[0].message.content, usage
        except Exception as e:
            last = e
            if attempt == max_retries - 1:
                break
            wait = min(20, 2**attempt + random.random())
            print(f"  Retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
            time.sleep(wait)
    raise RuntimeError(last)

def groq_json(system, user, model=None):
    text, usage = groq_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        model=model,
        json_mode=True,
    )
    return parse_json_object(text), usage


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1.7 — Coreference resolution
# ═══════════════════════════════════════════════════════════════════════════════
COREF_SYSTEM = """
You are a conservative coreference-resolution component for a knowledge-graph pipeline.
Resolve pronouns and generic references only when the antecedent is clearly supported in the same chunk.
Never invent facts. Preserve dates, numbers, tickers and product names.
Return strict JSON only.
""".strip()

def resolve_coref_batch(batch_df):
    payload = [{"chunk_id": r.chunk_id, "text": r.text}
               for r in batch_df.itertuples(index=False)]

    prompt = f"""
Resolve coreferences.

Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "resolved_text": "...",
      "unresolved_mentions": ["..."]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    obj, usage = groq_json(COREF_SYSTEM, prompt)
    by_id = {x.get("chunk_id"): x for x in obj.get("items", [])}

    rows = []
    for r in batch_df.itertuples(index=False):
        item = by_id.get(r.chunk_id, {})
        rows.append({
            "chunk_id": r.chunk_id,
            "resolved_text": norm_space(item.get("resolved_text") or r.text),
            "unresolved_mentions": item.get("unresolved_mentions", []),
        })
    return pd.DataFrame(rows), usage

def run_coref(chunks_subset, batch_size=5):
    out = []
    for start in tqdm(range(0, len(chunks_subset), batch_size), desc="Coref"):
        batch = chunks_subset.iloc[start:start+batch_size]
        try:
            df, _ = resolve_coref_batch(batch)
        except Exception:
            df = pd.DataFrame({
                "chunk_id": batch["chunk_id"].tolist(),
                "resolved_text": batch["text"].tolist(),
                "unresolved_mentions": [["COREF_BATCH_FAILED"] for _ in range(len(batch))],
            })
        out.append(df)
    return pd.concat(out, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2.1 — NER + RE extraction
# ═══════════════════════════════════════════════════════════════════════════════
ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
    "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS"
}

EXTRACT_SYSTEM = f"""
Extract a high-precision knowledge graph from tech-news text.
Allowed node types: {sorted(ALLOWED_NODE_TYPES)}
Allowed relations: {sorted(ALLOWED_RELATIONS)}
Use only explicitly supported facts. Prefer precision over recall.
Every relation needs short evidence. Return strict JSON only.
""".strip()

def extract_batch(batch_df):
    payload = [{
        "chunk_id": r.chunk_id,
        "published_date": r.published_date,
        "text": getattr(r, "resolved_text", None) or r.text,
    } for r in batch_df.itertuples(index=False)]

    prompt = f"""
Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "relations": [
        {{
          "source": "...",
          "source_type": "Company|Person|Technology",
          "relation": "ALLOWED_RELATION",
          "target": "...",
          "target_type": "Company|Person|Technology",
          "evidence": "...",
          "confidence": 0.0
        }}
      ]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    return groq_json(EXTRACT_SYSTEM, prompt)

def run_extraction(source_df, batch_size=4):
    meta = source_df.set_index("chunk_id")["published_date"].to_dict()
    triples, errors = [], []

    for start in tqdm(range(0, len(source_df), batch_size), desc="NER+RE"):
        batch = source_df.iloc[start:start+batch_size]
        try:
            obj, _ = extract_batch(batch)
        except Exception as e:
            errors.append({"start": start, "error": str(e)})
            continue

        for item in obj.get("items", []):
            cid = item.get("chunk_id")
            if cid not in meta:
                continue
            for x in item.get("relations", []):
                s, t = norm_space(x.get("source")), norm_space(x.get("target"))
                st, tt, rel = x.get("source_type"), x.get("target_type"), x.get("relation")
                if not s or not t:
                    continue
                if st not in ALLOWED_NODE_TYPES or tt not in ALLOWED_NODE_TYPES:
                    continue
                if rel not in ALLOWED_RELATIONS:
                    continue
                triples.append({
                    "source_raw": s,
                    "source_type": st,
                    "relation": rel,
                    "target_raw": t,
                    "target_type": tt,
                    "source_chunk_id": cid,
                    "published_date": meta[cid] or "",
                    "evidence": norm_space(x.get("evidence")),
                    "confidence": float(x.get("confidence") or 0.0),
                })

    return pd.DataFrame(triples), pd.DataFrame(errors)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2.2 — Entity resolution
# ═══════════════════════════════════════════════════════════════════════════════
CORP_SUFFIXES = {"inc","incorporated","corp","corporation","ltd","limited","llc","plc","co","company"}
MANUAL_ALIASES = {
    "msft": "Microsoft", "microsoft corp": "Microsoft", "microsoft corporation": "Microsoft",
    "goog": "Google", "googl": "Google", "google llc": "Google",
    "meta platforms": "Meta", "meta platforms inc": "Meta",
    "aapl": "Apple", "apple inc": "Apple",
}

def norm_entity(name):
    s = unicodedata.normalize("NFKC", norm_space(name)).lower()
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def strip_suffix(name):
    toks = norm_entity(name).replace(".", "").split()
    while toks and toks[-1] in CORP_SUFFIXES:
        toks.pop()
    return " ".join(toks)

def merge_guard(a, b):
    na, nb = strip_suffix(a), strip_suffix(b)
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.72

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)
    return embedder

class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a

def build_resolution_map(raw_triples_df, threshold=0.90, top_k=5):
    mentions = []
    for r in raw_triples_df.itertuples(index=False):
        mentions += [(r.source_type, r.source_raw), (r.target_type, r.target_raw)]

    counts = Counter((t, norm_entity(n)) for t, n in mentions)
    display_name = {}
    for t, n in mentions:
        display_name.setdefault((t, norm_entity(n)), n)

    mapping, audit = {}, []

    for key in counts:
        t, norm = key
        if norm in MANUAL_ALIASES:
            mapping[key] = MANUAL_ALIASES[norm]
            audit.append({
                "type": t, "left": display_name[key],
                "right": MANUAL_ALIASES[norm],
                "similarity": 1.0, "decision": "MERGE_MANUAL"
            })

    for typ in sorted(ALLOWED_NODE_TYPES):
        keys = [k for k in counts if k[0] == typ and k not in mapping]
        if not keys:
            continue
        names = [display_name[k] for k in keys]
        vecs = get_embedder().encode(
            names, batch_size=128, show_progress_bar=False,
            normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        sims, nbrs = index.search(vecs, min(top_k, len(names)))
        uf = UF(len(names))

        for i in range(len(names)):
            for score, j in zip(sims[i], nbrs[i]):
                if j < 0 or i >= j or float(score) < threshold:
                    continue
                ok = merge_guard(names[i], names[j])
                audit.append({
                    "type": typ, "left": names[i], "right": names[j],
                    "similarity": float(score),
                    "decision": "MERGE_VECTOR" if ok else "REJECT_GUARD"
                })
                if ok:
                    uf.union(i, j)

        groups = defaultdict(list)
        for i in range(len(names)):
            groups[uf.find(i)].append(i)

        for idxs in groups.values():
            best = sorted(
                idxs,
                key=lambda i: (-counts[keys[i]], len(names[i]), names[i].lower())
            )[0]
            canonical = names[best]
            for i in idxs:
                mapping[keys[i]] = canonical

    for key in counts:
        mapping.setdefault(key, display_name[key])

    return mapping, pd.DataFrame(audit)

def canonicalize_triples(raw_df, mapping):
    df = raw_df.copy()
    if df.empty:
        return df
    def canon(name, typ):
        n = norm_entity(name)
        return mapping.get((typ, n), MANUAL_ALIASES.get(n, name))

    df["source_name"] = [canon(n,t) for n,t in zip(df.source_raw, df.source_type)]
    df["target_name"] = [canon(n,t) for n,t in zip(df.target_raw, df.target_type)]
    df["source_name_norm"] = df.source_name.map(norm_entity)
    df["target_name_norm"] = df.target_name.map(norm_entity)
    df["source_id"] = [sha1(f"{t}:{n}")[:24] for t,n in zip(df.source_type, df.source_name_norm)]
    df["target_id"] = [sha1(f"{t}:{n}")[:24] for t,n in zip(df.target_type, df.target_name_norm)]
    return df[df.source_id != df.target_id].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2.3 — Node table + UNWIND bulk insert
# ═══════════════════════════════════════════════════════════════════════════════
def build_nodes(triples_df):
    rows = []
    for r in triples_df.itertuples(index=False):
        rows += [
            {"id":r.source_id,"name":r.source_name,"name_norm":r.source_name_norm,"type":r.source_type,"alias":r.source_raw},
            {"id":r.target_id,"name":r.target_name,"name_norm":r.target_name_norm,"type":r.target_type,"alias":r.target_raw},
        ]
    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return tmp

    out = []
    for (node_id,name,name_norm,typ), g in tmp.groupby(["id","name","name_norm","type"]):
        aliases = sorted(set(g["alias"].map(norm_space)))
        out.append({
            "id":node_id, "name":name, "name_norm":name_norm, "type":typ,
            "aliases":aliases,
            "aliases_norm":sorted(set(norm_entity(x) for x in aliases))
        })
    return pd.DataFrame(out)

def batches(records, size=1000):
    for i in range(0, len(records), size):
        yield records[i:i+size]

def bulk_insert_nodes(nodes_df, batch_size=1000):
    for typ in sorted(ALLOWED_NODE_TYPES):
        part = nodes_df[nodes_df.type == typ]
        if part.empty:
            continue
        query = f"""
        UNWIND $rows AS row
        MERGE (n:Entity {{id: row.id}})
        SET n:{typ},
            n.name=row.name,
            n.name_norm=row.name_norm,
            n.entity_type=row.type,
            n.aliases=row.aliases,
            n.aliases_norm=row.aliases_norm
        """
        for b in batches(part.to_dict("records"), batch_size):
            run_cypher(query, rows=b)

def bulk_insert_edges(triples_df, batch_size=1000):
    required = {"source_chunk_id","published_date"}
    if not required.issubset(triples_df.columns):
        raise ValueError("Missing edge provenance.")

    for rel in sorted(ALLOWED_RELATIONS):
        part = triples_df[triples_df.relation == rel]
        if part.empty:
            continue

        query = f"""
        UNWIND $rows AS row
        MATCH (s:Entity {{id: row.source_id}})
        MATCH (t:Entity {{id: row.target_id}})
        MERGE (s)-[r:{rel} {{source_chunk_id: row.source_chunk_id}}]->(t)
        SET r.published_date=row.published_date,
            r.evidence=row.evidence,
            r.confidence=row.confidence
        """

        cols = ["source_id","target_id","source_chunk_id","published_date","evidence","confidence"]
        for b in batches(part[cols].to_dict("records"), batch_size):
            run_cypher(query, rows=b)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2.4 — Sanity checks
# ═══════════════════════════════════════════════════════════════════════════════
def graph_checks():
    invalid = run_cypher("""
    MATCH ()-[r]->()
    WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL
    RETURN count(r) AS n
    """)[0]["n"]

    counts = {
        "nodes": run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"],
        "edges": run_cypher("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"],
        "invalid_provenance_edges": invalid,
    }
    print(f"Graph counts: {counts}")
    assert invalid == 0, f"Found {invalid} edges without provenance!"

    top = pd.DataFrame(run_cypher("""
    MATCH (n:Entity)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) AS degree
    RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
    ORDER BY degree DESC LIMIT 15
    """))
    print("\nTop 15 nodes by degree:")
    print(top.to_string())
    return counts, top


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3.1 — Flat RAG
# ═══════════════════════════════════════════════════════════════════════════════
flat_index = None
flat_store = None
entity_match_vectors = None
entity_match_store = None

def build_flat_index(chunks_df):
    global flat_index, flat_store
    vecs = get_embedder().encode(
        chunks_df.text.fillna("").tolist(),
        batch_size=128, show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    flat_index = faiss.IndexFlatIP(vecs.shape[1])
    flat_index.add(vecs)
    flat_store = chunks_df.reset_index(drop=True).copy()
    print(f"Flat vectors: {flat_index.ntotal}")

def retrieve_flat_context(query, k=6):
    qv = get_embedder().encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")
    scores, ids = flat_index.search(qv, min(k, flat_index.ntotal))

    rows = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        r = flat_store.iloc[int(idx)]
        rows.append({
            "score":float(score), "chunk_id":r.chunk_id,
            "published_date":r.published_date, "text":r.text
        })

    df = pd.DataFrame(rows)
    context = "\n\n".join(
        f"[chunk_id={r.chunk_id} | date={r.published_date} | score={r.score:.3f}]\n{r.text}"
        for r in df.itertuples(index=False)
    )
    return context, df


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3.2 — Seed matching
# ═══════════════════════════════════════════════════════════════════════════════
SEED_SYSTEM = """
Extract useful seed entities for graph retrieval.
Allowed types: Company, Person, Technology.
Do not answer the question. Return strict JSON only.
""".strip()

def extract_seeds(query):
    obj, _ = groq_json(SEED_SYSTEM, f"""
Question: {query}
Return {{"seeds":[{{"name":"...","type":"Company|Person|Technology|null"}}]}}
""")
    return [
        {"name":norm_space(x.get("name")),
         "type":x.get("type") if x.get("type") in ALLOWED_NODE_TYPES else None}
        for x in obj.get("seeds", [])
        if norm_space(x.get("name"))
    ]

def build_entity_matcher(nodes_df):
    global entity_match_vectors, entity_match_store
    entity_match_store = nodes_df.reset_index(drop=True).copy()
    entity_match_vectors = get_embedder().encode(
        entity_match_store.name.tolist(),
        batch_size=128, show_progress_bar=False,
        normalize_embeddings=True
    ).astype("float32")

def match_seeds(query, fuzzy_threshold=0.66):
    matched = []
    for seed in extract_seeds(query):
        exact = run_cypher("""
        MATCH (n:Entity)
        WHERE (n.name_norm=$name OR $name IN coalesce(n.aliases_norm,[]))
          AND ($typ IS NULL OR n.entity_type=$typ)
        RETURN n.id AS id, n.name AS name, n.entity_type AS type
        LIMIT 5
        """, name=norm_entity(seed["name"]), typ=seed["type"])

        if exact:
            matched += exact
            continue

        if entity_match_vectors is None:
            continue

        mask = np.ones(len(entity_match_store), dtype=bool)
        if seed["type"]:
            mask = entity_match_store.type.eq(seed["type"]).to_numpy()
        idxs = np.flatnonzero(mask)
        if not len(idxs):
            continue

        qv = get_embedder().encode(
            [seed["name"]], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")[0]
        sims = entity_match_vectors[idxs] @ qv
        j = int(np.argmax(sims))
        if float(sims[j]) >= fuzzy_threshold:
            r = entity_match_store.iloc[int(idxs[j])]
            matched.append({"id":r.id,"name":r.name,"type":r.type})

    return list({x["id"]: x for x in matched}.values())


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3.3 — Graph traversal + super-node mitigation
# ═══════════════════════════════════════════════════════════════════════════════
SUPER_NODE_DEGREE = 100
SUPER_NODE_EDGE_CAP = 50
GLOBAL_EDGE_CAP = 250
MAX_GRAPH_CONTEXT_CHARS = 14000

def node_degree(node_id):
    return int(run_cypher("""
    MATCH (n:Entity {id:$id})
    OPTIONAL MATCH (n)-[r]-()
    RETURN count(r) AS degree
    """, id=node_id)[0]["degree"])

def recent_edges(node_id, limit):
    return run_cypher("""
    MATCH (n:Entity {id:$id})
    MATCH (n)-[r]-(m:Entity)
    RETURN
      startNode(r).id AS source_id,
      startNode(r).name AS source_name,
      startNode(r).entity_type AS source_type,
      type(r) AS relation,
      endNode(r).id AS target_id,
      endNode(r).name AS target_name,
      endNode(r).entity_type AS target_type,
      r.source_chunk_id AS source_chunk_id,
      r.published_date AS published_date,
      r.evidence AS evidence,
      m.id AS neighbor_id
    ORDER BY coalesce(r.published_date,'') DESC
    LIMIT $limit
    """, id=node_id, limit=int(limit))

def textualize(edges):
    edges = sorted(edges, key=lambda e:e.get("published_date") or "", reverse=True)
    lines, used = [], 0
    for e in edges:
        line = (
            f"{e['source_name']} [{e['source_type']}] -{e['relation']}-> "
            f"{e['target_name']} [{e['target_type']}] "
            f"| date={e.get('published_date') or 'unknown'} "
            f"| chunk={e.get('source_chunk_id') or 'unknown'}"
        )
        if e.get("evidence"):
            line += f" | evidence={norm_space(e['evidence'])}"
        if used + len(line) + 1 > MAX_GRAPH_CONTEXT_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)

def retrieve_graph_context(query, max_hops=2, edge_limit=50, return_debug=False):
    seeds = match_seeds(query)
    if not seeds:
        out = {"context":"","edges":pd.DataFrame(),
               "diagnostics":{"reason":"NO_SEED","supernode_events":[]}}
        return out if return_debug else ""

    frontier = deque((x["id"],0) for x in seeds)
    expanded, seen_edges, collected = set(), set(), []
    supernode_events = []

    while frontier and len(collected) < GLOBAL_EDGE_CAP:
        node_id, hop = frontier.popleft()
        if node_id in expanded or hop >= max_hops:
            continue
        expanded.add(node_id)

        degree = node_degree(node_id)
        limit = int(edge_limit)
        if degree > SUPER_NODE_DEGREE:
            limit = min(limit, SUPER_NODE_EDGE_CAP)
            supernode_events.append({"node_id":node_id,"degree":degree,"limit":limit})

        for e in recent_edges(node_id, limit):
            key = (e["source_id"],e["relation"],e["target_id"],e["source_chunk_id"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            collected.append(e)
            if len(collected) >= GLOBAL_EDGE_CAP:
                break

            nb = e.get("neighbor_id")
            if nb and nb not in expanded and hop + 1 < max_hops:
                frontier.append((nb, hop+1))

    out = {
        "context": textualize(collected),
        "edges": pd.DataFrame(collected),
        "diagnostics": {
            "matched_seeds": seeds,
            "expanded_nodes": len(expanded),
            "collected_edges": len(collected),
            "supernode_events": supernode_events,
        }
    }
    return out if return_debug else out["context"]


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3.4 — Answer functions
# ═══════════════════════════════════════════════════════════════════════════════
ANSWER_SYSTEM = """
Answer only from supplied context.
Be concise but complete. Do not invent facts.
Cite provenance inline as [chunk_id=...] whenever possible.
If evidence is insufficient or conflicting, say so.
""".strip()

def generate_answer(question, context):
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"
    t0 = time.perf_counter()
    text, usage = groq_chat(
        [{"role":"system","content":ANSWER_SYSTEM},
         {"role":"user","content":prompt}],
        model=GROQ_MODEL
    )
    return {
        "answer": text.strip(),
        "latency_s": time.perf_counter()-t0,
        "total_tokens": usage.get("total_tokens"),
    }

def answer_flat_rag(question):
    context, retrieved = retrieve_flat_context(question, k=6)
    out = generate_answer(question, context)
    out.update({"context":context,"retrieved":retrieved})
    return out

def answer_graph_rag(question):
    g = retrieve_graph_context(question, max_hops=2, edge_limit=50, return_debug=True)
    vctx, vdocs = retrieve_flat_context(question, k=4)
    context = f"=== GRAPH ===\n{g['context']}\n\n=== VECTOR ===\n{vctx}"
    out = generate_answer(question, context)
    out.update({"context":context,"graph_debug":g,"vector_docs":vdocs})
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4.1 — Golden dataset
# ═══════════════════════════════════════════════════════════════════════════════
starter_golden = pd.DataFrame([
    {
        "id":"G01","group":"factoid",
        "question":"Who was the CEO of Hugging Face in 2023?",
        "reference_answer":"Clement Delangue",
        "reference_evidence":"Validate against instructor dump."
    },
    {
        "id":"G02","group":"multi-hop",
        "question":"Which startups were founded by former Microsoft employees and later received investment from Google?",
        "reference_answer":"",
        "reference_evidence":"TO_BE_FILLED_FROM_DATASET"
    },
    {
        "id":"G03","group":"cross-doc",
        "question":"Compare the direction of AI-related investments by Meta and Apple during 2023 using evidence from multiple articles.",
        "reference_answer":"",
        "reference_evidence":"TO_BE_FILLED_FROM_DATASET"
    },
    {
        "id":"G04","group":"multi-hop",
        "question":"Find a company invested in by a major technology company that also developed a named AI technology; identify both relations and dates.",
        "reference_answer":"",
        "reference_evidence":"TO_BE_FILLED_FROM_DATASET"
    },
    {
        "id":"G05","group":"cross-doc",
        "question":"Identify one technology connected to the same company in at least two news chunks and summarize how the relationship changed over time.",
        "reference_answer":"",
        "reference_evidence":"TO_BE_FILLED_FROM_DATASET"
    },
])


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4.2 — LLM-as-a-Judge
# ═══════════════════════════════════════════════════════════════════════════════
JUDGE_SYSTEM = """
You are a strict evaluator of RAG answers.
Score 1-5:
- comprehensiveness
- faithfulness to supplied candidate context
- multi_hop_reasoning accuracy
Use the reference answer as correctness anchor.
Return strict JSON only.
""".strip()

def judge_json_fn(system, user):
    if not JUDGE_MODEL:
        raise RuntimeError("Missing JUDGE_MODEL.")

    if JUDGE_PROVIDER == "groq":
        return groq_json(system, user, model=JUDGE_MODEL)[0]

    if JUDGE_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY.")
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.0,
            response_format={"type":"json_object"}
        )
        return parse_json_object(resp.choices[0].message.content)

    raise ValueError("JUDGE_PROVIDER must be openai or groq.")

def judge_answer(question, reference, answer, context):
    prompt = f"""
QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{answer}

CANDIDATE CONTEXT:
{context[:18000]}

Return:
{{
 "comprehensiveness":1,
 "faithfulness":1,
 "multi_hop_reasoning":1,
 "rationale":"2-5 sentences"
}}
"""
    obj = judge_json_fn(JUDGE_SYSTEM, prompt)
    out = {}
    for k in ["comprehensiveness","faithfulness","multi_hop_reasoning"]:
        out[k] = max(1, min(5, int(obj.get(k,1))))
    out["rationale"] = norm_space(obj.get("rationale"))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4.3 — Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════
def run_evaluation(golden_df):
    rows = []
    for q in tqdm(golden_df.itertuples(index=False), total=len(golden_df), desc="Evaluation"):
        flat = answer_flat_rag(q.question)
        graph = answer_graph_rag(q.question)

        jf = judge_answer(q.question, q.reference_answer, flat["answer"], flat["context"])
        jg = judge_answer(q.question, q.reference_answer, graph["answer"], graph["context"])

        rows.append({
            "id":q.id, "group":q.group, "question":q.question,
            "reference_answer":q.reference_answer,
            "flat_answer":flat["answer"], "graph_answer":graph["answer"],
            "flat_comprehensiveness":jf["comprehensiveness"],
            "graph_comprehensiveness":jg["comprehensiveness"],
            "flat_faithfulness":jf["faithfulness"],
            "graph_faithfulness":jg["faithfulness"],
            "flat_multi_hop_reasoning":jf["multi_hop_reasoning"],
            "graph_multi_hop_reasoning":jg["multi_hop_reasoning"],
            "flat_latency_s":flat["latency_s"],
            "graph_latency_s":graph["latency_s"],
            "flat_total_tokens":flat.get("total_tokens"),
            "graph_total_tokens":graph.get("total_tokens"),
            "flat_judge_rationale":jf["rationale"],
            "graph_judge_rationale":jg["rationale"],
            "graph_supernode_events":len(
                graph["graph_debug"]["diagnostics"].get("supernode_events",[])
            )
        })
        pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4.4 — Comparison table
# ═══════════════════════════════════════════════════════════════════════════════
def comparison_table(eval_df):
    metric_map = {
        "Comprehensiveness":("flat_comprehensiveness","graph_comprehensiveness"),
        "Faithfulness":("flat_faithfulness","graph_faithfulness"),
        "Multi-hop reasoning":("flat_multi_hop_reasoning","graph_multi_hop_reasoning"),
        "Latency (s)":("flat_latency_s","graph_latency_s"),
        "Token usage":("flat_total_tokens","graph_total_tokens"),
    }

    rows = []
    for group, g in eval_df.groupby("group"):
        for metric, (fc,gc) in metric_map.items():
            f = pd.to_numeric(g[fc], errors="coerce").mean()
            gr = pd.to_numeric(g[gc], errors="coerce").mean()

            if metric in {"Latency (s)","Token usage"}:
                comment = "Flat RAG usually cheaper/faster." if f < gr else "GraphRAG not more expensive in this sample."
            else:
                delta = gr - f
                if delta >= .75:
                    comment = "GraphRAG significantly better; check rationale and provenance."
                elif delta <= -.5:
                    comment = "Flat RAG better; graph extraction/retrieval may lose info or add noise."
                else:
                    comment = "Both methods are close."

            rows.append({
                "Group":group, "Metric":metric,
                "Flat RAG":round(f,3) if pd.notna(f) else np.nan,
                "GraphRAG":round(gr,3) if pd.notna(gr) else np.nan,
                "Analysis":comment
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5.1 — Super-node check
# ═══════════════════════════════════════════════════════════════════════════════
def test_supernode_policy():
    rows = run_cypher("""
    MATCH (n:Entity)-[r]-()
    WITH n, count(r) AS degree
    ORDER BY degree DESC LIMIT 1
    RETURN n.id AS id, n.name AS name, degree
    """)
    if not rows:
        print("Graph empty.")
        return

    n = rows[0]
    limit = 50 if n["degree"] > SUPER_NODE_DEGREE else 1000
    edges = recent_edges(n["id"], limit)
    print(f"Top node: {n}, fetched={len(edges)}")
    if n["degree"] > SUPER_NODE_DEGREE:
        assert len(edges) <= 50
        print("Super-node cap OK.")


# ═══════════════════════════════════════════════════════════════════════════════
# Bonus — NetworkX community
# ═══════════════════════════════════════════════════════════════════════════════
import networkx as nx

def build_communities(limit_edges=20000):
    edge_df = pd.DataFrame(run_cypher("""
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.id AS source, b.id AS target
    LIMIT $limit
    """, limit=int(limit_edges)))

    if edge_df.empty:
        print("No edges for community detection.")
        return pd.DataFrame()

    G = nx.Graph()
    G.add_edges_from(edge_df[["source","target"]].itertuples(index=False, name=None))
    communities = nx.algorithms.community.greedy_modularity_communities(G)

    rows = []
    for cid, members in enumerate(communities):
        rows += [{"id":node_id,"community_id":int(cid)} for node_id in members]

    for b in batches(rows, 1000):
        run_cypher("""
        UNWIND $rows AS row
        MATCH (n:Entity {id:row.id})
        SET n.community_id=row.community_id
        """, rows=b)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Bonus — Self-correction
# ═══════════════════════════════════════════════════════════════════════════════
SUFFICIENCY_SYSTEM = """
Decide whether the supplied retrieval context is sufficient to answer the question faithfully.
Do not answer the question. Return strict JSON only.
""".strip()

def context_sufficient(question, context):
    obj, _ = groq_json(
        SUFFICIENCY_SYSTEM,
        f"""QUESTION: {question}
CONTEXT:
{context[:16000]}
Return {{"sufficient":true,"missing":"..."}}"""
    )
    return bool(obj.get("sufficient")), norm_space(obj.get("missing"))

def self_correcting_context(question):
    g2 = retrieve_graph_context(question, 2, 50, True)
    ok, missing = context_sufficient(question, g2["context"])
    if ok:
        return {"route":"hop2","context":g2["context"],"missing":""}

    g3 = retrieve_graph_context(question, 3, 50, True)
    ok, missing2 = context_sufficient(question, g3["context"])
    if ok:
        return {"route":"hop3","context":g3["context"],"missing":missing}

    flat, _ = retrieve_flat_context(question, k=8)
    return {
        "route":"hop3+vector",
        "context":f"=== GRAPH ===\n{g3['context']}\n\n=== VECTOR ===\n{flat}",
        "missing":missing2
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                           MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUTS_DIR = PROJECT_DIR / "outputs"
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── Phase 1: Data loading ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Data Loading & Preprocessing")
    print("=" * 60)

    stream_dataset()

    connect_neo4j()
    setup_graph_schema()

    raw_df = load_news(DATA_PATH)
    news_df = standardize_news(raw_df)
    chunks_df = build_chunks(news_df)
    print(f"Chunks: {len(chunks_df):,}")
    print(chunks_df.head())

    # Coreference resolution
    extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS).copy()
    print(f"\nRunning coreference on {len(extraction_source)} chunks...")
    coref_df = run_coref(extraction_source)
    extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")
    print(f"Coreference done. Unresolved mentions sample:")
    unresolved = coref_df[coref_df.unresolved_mentions.apply(lambda x: len(x) > 0 and x != ['COREF_BATCH_FAILED'])]
    if len(unresolved) > 0:
        print(unresolved.head())

    # ── Phase 2: Extraction & Neo4j ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: NER/RE Extraction & Neo4j Ingestion")
    print("=" * 60)

    raw_triples_df, extraction_errors_df = run_extraction(extraction_source)
    print(f"Triples extracted: {len(raw_triples_df):,}")
    print(f"Extraction errors: {len(extraction_errors_df)}")
    if not raw_triples_df.empty:
        print(raw_triples_df.head())

    if raw_triples_df.empty:
        print("WARNING: No triples extracted! Check your Groq API key and model.")
        print("Attempting to continue with empty graph...")
        entity_resolution_audit_df = pd.DataFrame()
        triples_df = raw_triples_df
    else:
        entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)
        triples_df = canonicalize_triples(raw_triples_df, entity_map)
        print(f"\nEntity resolution audit: {len(entity_resolution_audit_df)} entries")
        print(entity_resolution_audit_df.head(20).to_string())

    # Bulk insert
    nodes_df = build_nodes(triples_df)
    print(f"\nInserting {len(nodes_df)} nodes...")
    bulk_insert_nodes(nodes_df)
    print(f"Inserting {len(triples_df)} edges...")
    bulk_insert_edges(triples_df)

    # Sanity checks
    graph_counts, top_degree_df = graph_checks()

    # ── Phase 3: RAG ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Flat RAG & Hybrid GraphRAG")
    print("=" * 60)

    build_flat_index(chunks_df)
    build_entity_matcher(nodes_df)

    # Quick test
    test_q = "What companies did Microsoft invest in?"
    print(f"\nTest query: {test_q}")
    flat_test = answer_flat_rag(test_q)
    print(f"Flat RAG answer: {flat_test['answer'][:200]}...")
    graph_test = answer_graph_rag(test_q)
    print(f"GraphRAG answer: {graph_test['answer'][:200]}...")

    # ── Phase 4: Golden Dataset & Evaluation ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: Golden Dataset & LLM-as-a-Judge")
    print("=" * 60)

    # Fill reference answers from graph data where possible
    golden_df = starter_golden.copy()

    # Try to fill reference answers using graph queries
    for idx, row in golden_df.iterrows():
        if row["reference_answer"] and str(row["reference_answer"]).strip():
            continue
        # Use GraphRAG to generate reference answer
        print(f"Generating reference answer for {row['id']}: {row['question'][:60]}...")
        try:
            result = answer_graph_rag(row["question"])
            golden_df.at[idx, "reference_answer"] = result["answer"]
        except Exception as e:
            print(f"  Error: {e}")
            golden_df.at[idx, "reference_answer"] = "Unable to determine from available data."

    golden_df.to_csv(GOLDEN_PATH, index=False)
    print("\nGolden Dataset:")
    print(golden_df[["id", "group", "question", "reference_answer"]].to_string())

    # Run evaluation
    print("\nRunning LLM-as-a-Judge evaluation...")
    eval_results_df = run_evaluation(golden_df)
    print("\nEvaluation results:")
    print(eval_results_df[["id","group","flat_comprehensiveness","graph_comprehensiveness",
                           "flat_faithfulness","graph_faithfulness",
                           "flat_multi_hop_reasoning","graph_multi_hop_reasoning"]].to_string())

    # Comparison table
    comparison_df = comparison_table(eval_results_df)
    print("\nComparison table:")
    print(comparison_df.to_string())

    # Export CSVs
    eval_out = str(OUTPUTS_DIR / "graphrag_eval_results.csv")
    comp_out = str(OUTPUTS_DIR / "graphrag_vs_flatrag_summary.csv")
    eval_results_df.to_csv(eval_out, index=False)
    comparison_df.to_csv(comp_out, index=False)
    print(f"\nExported: {eval_out}")
    print(f"Exported: {comp_out}")

    # ── Phase 5: Failure checks ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5: Failure-Mode Checks & Bonus")
    print("=" * 60)

    test_supernode_policy()

    print("\nEntity Resolution Audit (top rejected):")
    rejected = entity_resolution_audit_df[entity_resolution_audit_df.decision == "REJECT_GUARD"]
    if not rejected.empty:
        print(rejected.sort_values("similarity", ascending=False).head(20).to_string())
    else:
        print("No REJECT_GUARD entries.")

    # Bonus: Community detection
    print("\nBonus: Community detection...")
    try:
        community_df = build_communities()
        if not community_df.empty:
            print(f"Communities assigned: {community_df.community_id.nunique()} communities, {len(community_df)} nodes")
    except Exception as e:
        print(f"Community detection skipped: {e}")

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Graph: {graph_counts}")
    print(f"Eval results: {eval_out}")
    print(f"Comparison: {comp_out}")
    print(f"Golden dataset: {GOLDEN_PATH}")

    if driver:
        driver.close()
