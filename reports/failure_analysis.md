# Phân Tích Ca Lỗi — Flat RAG vs GraphRAG

**Học viên:** Phạm Nguyên Việt — 2A202601547  
**Ngày:** 19/08/2026  

---

## Ca lỗi 1: Flat RAG thất bại, GraphRAG thành công

### Question G02 (multi-hop):
> "Which startups were founded by former Microsoft employees and later received investment from Google?"

### Kết quả:
| Metric | Flat RAG | GraphRAG |
|--------|----------|----------|
| Comprehensiveness | 1 | **5** |
| Faithfulness | 1 | **5** |
| Multi-hop Reasoning | 1 | **5** |
| Latency (s) | 12.11 | 12.44 |

### Root-cause Analysis:

**Tại sao Flat RAG thất bại?**
1. Vector search (FAISS IndexFlatIP, k=6) tìm 6 chunks có embedding gần nhất với query.
2. Các chunks trả về chỉ chứa thông tin về Microsoft HOẶC Google riêng lẻ — không chunk nào đồng thời chứa cả chuỗi logic "Microsoft employee → founded startup → Google invested".
3. Câu hỏi multi-hop đòi hỏi kết nối ≥2 quan hệ (WORKED_AT + FOUNDED + INVESTED_IN), mà Flat RAG dựa trên similarity search không capture được structural relationships.
4. Kết quả: câu trả lời thiếu dữ kiện, không truy vết được path từ Microsoft → startup → Google.

**GraphRAG đã giải quyết như thế nào?**
1. **Seed Extraction:** LLM trích xuất seeds: `Microsoft (Company)`, `Google (Company)`.
2. **Seed Matching:** Khớp với nodes trong Neo4j qua `name_norm` matching.
3. **BFS Traversal (max_hops=2):** Từ node Microsoft, traverse qua edges WORKED_AT, FOUNDED → tìm startups liên quan. Từ node Google, traverse INVESTED_IN → tìm targets.
4. **Intersection:** Giao điểm giữa 2 subgraph cho ra startups thỏa cả 2 điều kiện.
5. **Hybrid Context:** Graph context + vector context (k=4) kết hợp cho LLM đủ evidence.

### Bài học:
Multi-hop queries là điểm mạnh tự nhiên của GraphRAG — graph edges encode structural relationships mà vector similarity bỏ qua.

---

## Ca lỗi 2: GraphRAG thất bại, Flat RAG thành công

### Question G04 (multi-hop):
> "Find a company invested in by a major technology company that also developed a named AI technology; identify both relations and dates."

### Kết quả:
| Metric | Flat RAG | GraphRAG |
|--------|----------|----------|
| Comprehensiveness | **5** | 1 |
| Faithfulness | **5** | 1 |
| Multi-hop Reasoning | **5** | 1 |
| Latency (s) | 7.56 | 18.08 |

### Root-cause Analysis:

**Tại sao GraphRAG thất bại?**
1. **Graph quá thưa:** Chỉ có 63 edges từ 400 chunks extraction (do rate limit API, ~50% batches fail). Nhiều relationships INVESTED_IN + DEVELOPED không được trích xuất.
2. **Seed extraction vague:** Câu hỏi nói "a major technology company" và "a named AI technology" — không có entity cụ thể. LLM seed extraction phải đoán → có thể chọn sai seed hoặc seed không tồn tại trong graph.
3. **Missing paths:** Ngay cả khi seed đúng, path INVESTED_IN → DEVELOPED có thể không tồn tại trong graph thưa → context trả về rỗng hoặc không đủ.
4. **Latency cao:** 18s do seed extraction + graph traversal + multiple Cypher queries, nhưng kết quả vẫn kém.

**Tại sao Flat RAG thành công?**
1. Vector search tìm được chunks chứa trực tiếp cả 2 thông tin (investment + AI technology development) trong text liền kề.
2. Không cần structural path — semantic similarity đủ để tìm relevant chunks.
3. LLM tổng hợp thông tin từ multiple chunks thành câu trả lời coherent.

### Đề xuất khắc phục:
1. **Tăng EXTRACTION_MAX_CHUNKS** từ 400 lên 1000+ để graph dày hơn.
2. **Fallback strategy:** Khi graph context rỗng hoặc quá ít edges, tự động tăng vector retrieval k từ 4 lên 8.
3. **Cải thiện seed extraction prompt:** Xử lý tốt hơn trường hợp entity vague/generic.
4. **Self-correction loop:** Detect context insufficient → expand hops (2→3) → vector fallback.

---

## Tổng kết

| Scenario | Flat RAG | GraphRAG | Lý do |
|----------|----------|----------|-------|
| Factoid (G01) | Tie | Tie | Cả 2 đều tìm được fact đơn giản |
| Multi-hop explicit entities (G02) | Fail | **Win** | Graph edges kết nối cross-entity |
| Cross-doc comparison (G03) | Fail | **Win** | Graph traversal tổng hợp multi-doc |
| Multi-hop vague entities (G04) | **Win** | Fail | Graph quá thưa + seed vague |
| Cross-doc single entity (G05) | Tie | Tie | Cả 2 đều tìm được evidence |

**Kết luận:** GraphRAG vượt trội khi entities rõ ràng và graph đủ dày. Flat RAG mạnh khi entities vague và thông tin nằm trong text liền kề.
