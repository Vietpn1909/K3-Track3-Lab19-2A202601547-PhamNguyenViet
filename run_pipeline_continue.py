#!/usr/bin/env python3
"""
run_pipeline_continue.py — Resume from Phase 3 using a different model
Neo4j data is already inserted. Just need RAG + Evaluation.
"""
import os, re, json, time, random, hashlib, unicodedata, sys
from pathlib import Path
from collections import defaultdict, Counter, deque
from difflib import SequenceMatcher

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "groq").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Use models with separate rate limits for remaining work
GROQ_MODEL = "qwen/qwen3.6-27b"       # For answer generation & seed extraction
JUDGE_MODEL = "openai/gpt-oss-20b"    # For LLM-as-a-Judge (separate rate limit)

# Fallback chain if primary model hits rate limit
MODEL_FALLBACK = ["qwen/qwen3.6-27b", "openai/gpt-oss-20b", "allam-2-7b"]

DATA_PATH = str(DATA_DIR / "hackernoon_subset.csv")
GOLDEN_PATH = str(DATA_DIR / "golden_dataset.csv")
CHECKPOINT_PATH = str(DATA_DIR / "checkpoint.csv")

CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
LAB_MAX_ARTICLES = 1500
LAB_MAX_CHUNKS = 3000
EXTRACTION_MAX_CHUNKS = 400

ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
    "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS"
}

SUPER_NODE_DEGREE = 100
SUPER_NODE_EDGE_CAP = 50
GLOBAL_EDGE_CAP = 250
MAX_GRAPH_CONTEXT_CHARS = 14000

print("=" * 60)
print("Lab 19: RESUME from Phase 3 — RAG + Evaluation")
print(f"GROQ_MODEL: {GROQ_MODEL}")
print(f"JUDGE_MODEL: {JUDGE_MODEL}")
print("=" * 60)

# ─── Utilities ────────────────────────────────────────────────────────────────
def norm_space(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def sha1(x):
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()

def norm_entity(name):
    import unicodedata
    s = unicodedata.normalize("NFKC", norm_space(name)).lower()
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

MANUAL_ALIASES = {
    "msft": "Microsoft", "microsoft corp": "Microsoft", "microsoft corporation": "Microsoft",
    "goog": "Google", "googl": "Google", "google llc": "Google",
    "meta platforms": "Meta", "meta platforms inc": "Meta",
    "aapl": "Apple", "apple inc": "Apple",
}

# ─── Neo4j ────────────────────────────────────────────────────────────────────
driver = None

def connect_neo4j():
    global driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Neo4j connected.")

def run_cypher(query, **params):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, **params)
        rows = [r.data() for r in result]
        result.consume()
    return rows

# ─── LLM ──────────────────────────────────────────────────────────────────────
from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY)

def parse_json_object(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("No JSON object found.")
    return json.loads(text[a:b+1])

def groq_chat(messages, model=None, json_mode=False, max_retries=6):
    model = model or GROQ_MODEL
    last = None
    models_to_try = [model] + [m for m in MODEL_FALLBACK if m != model]

    for current_model in models_to_try:
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": current_model,
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
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str:
                    # Parse wait time or use exponential backoff
                    wait = min(120, 10 * (attempt + 1) + random.random() * 5)
                    print(f"  Rate limit on {current_model}, wait {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                elif "404" in err_str or "model_not_found" in err_str:
                    print(f"  Model {current_model} not found, trying next...")
                    break  # Try next model
                else:
                    wait = min(30, 2**attempt + random.random())
                    print(f"  Retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
                    time.sleep(wait)
    raise RuntimeError(last)

def groq_json(system, user, model=None):
    text, usage = groq_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        model=model, json_mode=True,
    )
    return parse_json_object(text), usage

# ─── Embedder ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)
    return embedder

# ─── Chunking (rebuild from CSV) ─────────────────────────────────────────────
def pick_col(df, candidates, required=True):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise KeyError(f"Missing one of columns: {candidates}")
    return None

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

# ─── Flat RAG ─────────────────────────────────────────────────────────────────
flat_index = None
flat_store = None

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
        rows.append({"score":float(score),"chunk_id":r.chunk_id,
                      "published_date":r.published_date,"text":r.text})
    df = pd.DataFrame(rows)
    context = "\n\n".join(
        f"[chunk_id={r.chunk_id} | date={r.published_date} | score={r.score:.3f}]\n{r.text}"
        for r in df.itertuples(index=False)
    )
    return context, df

# ─── Seed matching ────────────────────────────────────────────────────────────
entity_match_vectors = None
entity_match_store = None

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

# ─── Graph traversal ─────────────────────────────────────────────────────────
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
      startNode(r).id AS source_id, startNode(r).name AS source_name,
      startNode(r).entity_type AS source_type, type(r) AS relation,
      endNode(r).id AS target_id, endNode(r).name AS target_name,
      endNode(r).entity_type AS target_type,
      r.source_chunk_id AS source_chunk_id, r.published_date AS published_date,
      r.evidence AS evidence, m.id AS neighbor_id
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

# ─── Answer functions ─────────────────────────────────────────────────────────
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

# ─── Judge ────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """
You are a strict evaluator of RAG answers.
Score 1-5:
- comprehensiveness
- faithfulness to supplied candidate context
- multi_hop_reasoning accuracy
Use the reference answer as correctness anchor.
Return strict JSON only.
""".strip()

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
    obj, _ = groq_json(JUDGE_SYSTEM, prompt, model=JUDGE_MODEL)
    out = {}
    for k in ["comprehensiveness","faithfulness","multi_hop_reasoning"]:
        out[k] = max(1, min(5, int(obj.get(k,1))))
    out["rationale"] = norm_space(obj.get("rationale"))
    return out

# ─── Evaluation runner ────────────────────────────────────────────────────────
def run_evaluation(golden_df):
    rows = []
    for q in tqdm(golden_df.itertuples(index=False), total=len(golden_df), desc="Evaluation"):
        print(f"\n  Evaluating: {q.question[:60]}...")
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
        print(f"  Flat: comp={jf['comprehensiveness']} faith={jf['faithfulness']} mhop={jf['multi_hop_reasoning']}")
        print(f"  Graph: comp={jg['comprehensiveness']} faith={jg['faithfulness']} mhop={jg['multi_hop_reasoning']}")
    return pd.DataFrame(rows)

# ─── Comparison table ─────────────────────────────────────────────────────────
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
                comment = "Flat RAG usually cheaper/faster." if f < gr else "GraphRAG not more expensive."
            else:
                delta = gr - f
                if delta >= .75:
                    comment = "GraphRAG significantly better."
                elif delta <= -.5:
                    comment = "Flat RAG better; graph may lose info."
                else:
                    comment = "Both methods are close."
            rows.append({"Group":group,"Metric":metric,
                         "Flat RAG":round(f,3) if pd.notna(f) else np.nan,
                         "GraphRAG":round(gr,3) if pd.notna(gr) else np.nan,
                         "Analysis":comment})
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
#                           MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Reconnect Neo4j ──────────────────────────────────────────────────────
    connect_neo4j()

    # ── Verify existing graph data ───────────────────────────────────────────
    counts = {
        "nodes": run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"],
        "edges": run_cypher("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"],
    }
    print(f"Existing graph: {counts}")

    if counts["nodes"] == 0:
        print("ERROR: No nodes in graph. Run full pipeline first.")
        sys.exit(1)

    # ── Rebuild chunks_df from CSV ───────────────────────────────────────────
    print("\nRebuilding chunks from CSV...")
    raw_df = pd.read_csv(DATA_PATH)
    text_col = pick_col(raw_df, ["text","content","article","body","story","description"])
    title_col = pick_col(raw_df, ["title","headline"], required=False)
    date_col = pick_col(raw_df, ["published_date","date","published_at","created_at"], required=False)
    id_col = pick_col(raw_df, ["id","article_id","story_id","uuid"], required=False)

    news_df = pd.DataFrame()
    news_df["text"] = raw_df[text_col].fillna("").map(norm_space)
    news_df["title"] = raw_df[title_col].fillna("").map(norm_space) if title_col else ""
    if date_col:
        news_df["published_date"] = pd.to_datetime(raw_df[date_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")
    else:
        news_df["published_date"] = ""
    if id_col:
        news_df["article_id"] = raw_df[id_col].astype(str)
    else:
        news_df["article_id"] = [sha1(f"{t}\n{x}")[:20] for t,x in zip(news_df["title"], news_df["text"])]

    news_df = news_df[news_df["text"].str.len() >= 80].copy()
    news_df["dedup_key"] = [sha1(norm_space(f"{t}\n{x}").lower()) for t,x in zip(news_df["title"], news_df["text"])]
    news_df = news_df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
    if len(news_df) > LAB_MAX_ARTICLES:
        news_df = news_df.sample(LAB_MAX_ARTICLES, random_state=SEED).sort_index().reset_index(drop=True)

    chunks_rows = []
    for r in news_df.itertuples(index=False):
        for i, text in enumerate(chunk_text(r.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)):
            chunks_rows.append({
                "chunk_id": f"{r.article_id}::c{i:04d}",
                "article_id": r.article_id,
                "title": r.title,
                "published_date": r.published_date,
                "text": text,
            })
            if LAB_MAX_CHUNKS and len(chunks_rows) >= LAB_MAX_CHUNKS:
                break
    chunks_df = pd.DataFrame(chunks_rows)
    print(f"Chunks rebuilt: {len(chunks_df):,}")

    # ── Rebuild nodes_df from Neo4j ──────────────────────────────────────────
    nodes_data = run_cypher("""
    MATCH (n:Entity)
    RETURN n.id AS id, n.name AS name, n.name_norm AS name_norm, n.entity_type AS type
    """)
    nodes_df = pd.DataFrame(nodes_data)
    print(f"Nodes from Neo4j: {len(nodes_df)}")

    # ── Phase 3: RAG ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Flat RAG & Hybrid GraphRAG")
    print("=" * 60)

    build_flat_index(chunks_df)
    build_entity_matcher(nodes_df)

    test_q = "What companies did Microsoft invest in?"
    print(f"\nTest query: {test_q}")
    flat_test = answer_flat_rag(test_q)
    print(f"Flat RAG answer: {flat_test['answer'][:300]}")
    graph_test = answer_graph_rag(test_q)
    print(f"GraphRAG answer: {graph_test['answer'][:300]}")

    # ── Phase 4: Golden Dataset & Evaluation ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: Golden Dataset & LLM-as-a-Judge")
    print("=" * 60)

    starter_golden = pd.DataFrame([
        {"id":"G01","group":"factoid",
         "question":"Who was the CEO of Hugging Face in 2023?",
         "reference_answer":"Clement Delangue",
         "reference_evidence":"Validate against instructor dump."},
        {"id":"G02","group":"multi-hop",
         "question":"Which startups were founded by former Microsoft employees and later received investment from Google?",
         "reference_answer":"","reference_evidence":"TO_BE_FILLED_FROM_DATASET"},
        {"id":"G03","group":"cross-doc",
         "question":"Compare the direction of AI-related investments by Meta and Apple during 2023 using evidence from multiple articles.",
         "reference_answer":"","reference_evidence":"TO_BE_FILLED_FROM_DATASET"},
        {"id":"G04","group":"multi-hop",
         "question":"Find a company invested in by a major technology company that also developed a named AI technology; identify both relations and dates.",
         "reference_answer":"","reference_evidence":"TO_BE_FILLED_FROM_DATASET"},
        {"id":"G05","group":"cross-doc",
         "question":"Identify one technology connected to the same company in at least two news chunks and summarize how the relationship changed over time.",
         "reference_answer":"","reference_evidence":"TO_BE_FILLED_FROM_DATASET"},
    ])

    golden_df = starter_golden.copy()
    for idx, row in golden_df.iterrows():
        if row["reference_answer"] and str(row["reference_answer"]).strip():
            continue
        print(f"Generating reference answer for {row['id']}...")
        try:
            result = answer_graph_rag(row["question"])
            golden_df.at[idx, "reference_answer"] = result["answer"]
        except Exception as e:
            print(f"  Error: {e}")
            golden_df.at[idx, "reference_answer"] = "Unable to determine from available data."

    golden_df.to_csv(GOLDEN_PATH, index=False)
    print("\nGolden Dataset:")
    for _, r in golden_df.iterrows():
        print(f"  {r['id']} [{r['group']}]: {r['question'][:60]}...")
        print(f"    Ref: {str(r['reference_answer'])[:100]}...")

    print("\nRunning LLM-as-a-Judge evaluation...")
    eval_results_df = run_evaluation(golden_df)

    comparison_df = comparison_table(eval_results_df)
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string())

    eval_out = str(OUTPUTS_DIR / "graphrag_eval_results.csv")
    comp_out = str(OUTPUTS_DIR / "graphrag_vs_flatrag_summary.csv")
    eval_results_df.to_csv(eval_out, index=False)
    comparison_df.to_csv(comp_out, index=False)
    print(f"\nExported: {eval_out}")
    print(f"Exported: {comp_out}")

    # ── Phase 5: Failure-mode checks ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5: Failure-Mode Checks & Bonus")
    print("=" * 60)

    # Super-node policy
    top_rows = run_cypher("""
    MATCH (n:Entity)-[r]-()
    WITH n, count(r) AS degree
    ORDER BY degree DESC LIMIT 1
    RETURN n.id AS id, n.name AS name, degree
    """)
    if top_rows:
        n = top_rows[0]
        limit = 50 if n["degree"] > SUPER_NODE_DEGREE else 1000
        edges = recent_edges(n["id"], limit)
        print(f"Top node: {n}, fetched={len(edges)}")
        if n["degree"] > SUPER_NODE_DEGREE:
            assert len(edges) <= 50
            print("Super-node cap OK.")
    else:
        print("No nodes for super-node check.")

    # Community detection
    import networkx as nx
    print("\nBonus: Community detection...")
    try:
        edge_df = pd.DataFrame(run_cypher("""
        MATCH (a:Entity)-[r]->(b:Entity)
        RETURN a.id AS source, b.id AS target LIMIT 20000
        """))
        if not edge_df.empty:
            G = nx.Graph()
            G.add_edges_from(edge_df[["source","target"]].itertuples(index=False, name=None))
            communities = nx.algorithms.community.greedy_modularity_communities(G)
            rows = []
            for cid, members in enumerate(communities):
                rows += [{"id":nid,"community_id":int(cid)} for nid in members]
            for i in range(0, len(rows), 1000):
                run_cypher("UNWIND $rows AS row MATCH (n:Entity {id:row.id}) SET n.community_id=row.community_id",
                           rows=rows[i:i+1000])
            print(f"Communities: {len(communities)} communities, {len(rows)} nodes assigned")
    except Exception as e:
        print(f"Community detection skipped: {e}")

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Graph: {counts}")
    print(f"Eval results: {eval_out}")
    print(f"Comparison: {comp_out}")
    print(f"Golden dataset: {GOLDEN_PATH}")

    if driver:
        driver.close()
