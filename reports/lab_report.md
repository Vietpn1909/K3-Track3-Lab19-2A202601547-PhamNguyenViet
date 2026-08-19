# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Nguyên Việt  
**Mã số:** 2A202601547  
**Khóa học:** AICB-K3 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026  

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu:** Trong các chunk nói về "onsemi and Sineng Electric Spearhead the Development...", đại từ "the company" và "it" xuất hiện nhiều lần khi nhắc đến cả onsemi lẫn Sineng Electric trong cùng một đoạn.
- **Hiện tượng:** Coreference resolution gặp khó khăn khi hai công ty được nhắc trong cùng một câu, dẫn đến "the company" có thể bị gán nhầm cho công ty sai — ví dụ gán hành động "developed solar technology" cho Sineng Electric thay vì onsemi.
- **Hậu quả đối với Graph:** Tạo ra False Edge gán nhầm quan hệ DEVELOPED cho công ty không phải chủ thể thật sự. Điều này làm ô nhiễm Knowledge Graph và ảnh hưởng đến kết quả truy vấn multi-hop downstream. Trong pipeline, một số batch coreference đã fail (COREF_BATCH_FAILED) và fallback về text gốc, tránh tệ hơn nhưng mất cơ hội phân giải chính xác.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` (mặc định trong notebook)
- **Cơ chế Lexical Guard:** Hàm `merge_guard()` sử dụng `SequenceMatcher.ratio() >= 0.72` sau khi strip corporate suffixes (Inc, Corp, LLC...). Ngay cả khi vector similarity > 0.90, nếu lexical ratio < 0.72 thì REJECT.
- **Ví dụ cặp bị Guard chặn:** Với dataset 5000 dòng, entity resolution audit có 0 REJECT_GUARD entries vì số lượng entities ít (113 nodes, 63 triples). Trong thực tế scale lớn hơn, các cặp như `Apple` vs `Apple Music`, hay `DI` vs `DI Wire` sẽ bị chặn vì `strip_suffix` cho ra tên khác nhau và `SequenceMatcher` ratio thấp hơn 0.72.
- **Lý do thiết kế:** Lexical Guard là lớp bảo vệ chống lại việc gộp nhầm các thực thể có embedding tương tự nhưng ngữ nghĩa khác nhau (e.g., công ty con vs công ty mẹ, sản phẩm vs hãng).

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | Synopsys | Company | 2 |
| 2 | Aqara | Company | 2 |
| 3 | ServiceNow | Company | 2 |

> **Ghi chú:** Với dataset giới hạn 5000 dòng và 400 chunks extraction, đồ thị có 113 nodes và 63 edges. Bậc tối đa chỉ là 2, nên không có super-node thực sự (degree > 100). Super-node mitigation policy vẫn được implement và test thành công — nếu degree > 100, chỉ lấy tối đa 50 cạnh mới nhất.

- **Ưu điểm & Rủi ro của Temporal Mitigation:**
  - *Ưu điểm:* Giảm thiểu context explosion khi traversal qua hub node (Google, Microsoft...). Giữ lại thông tin cập nhật nhất, phù hợp với câu hỏi real-time. Giới hạn `GLOBAL_EDGE_CAP=250` ngăn BFS bùng nổ.
  - *Rủi ro:* Nếu câu hỏi liên quan đến sự kiện lịch sử xa (ví dụ: "ai đầu tư vào công ty X năm 2018?"), các cạnh cũ có thể bị cắt mất. Giải pháp: cho phép hint temporal range trong query.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 2.80 | 3.40 | +0.60 | GraphRAG tốt hơn ở cross-doc, nhưng mixed ở multi-hop |
| **Faithfulness (1–5)** | 2.80 | 3.40 | +0.60 | GraphRAG tốt hơn nhờ evidence tracing qua graph edges |
| **Multi-hop Reasoning (1–5)** | 2.60 | 3.40 | +0.80 | GraphRAG vượt trội khi cần kết nối thông tin rời rạc |
| **Latency trung bình (s)** | 7.61 | 11.43 | +3.82 | GraphRAG chậm hơn do graph traversal + seed matching |
| **Token usage trung bình** | 2194.8 | 2030.0 | -164.8 | GraphRAG hiệu quả token hơn nhờ context có cấu trúc |

#### Phân tích 2 Ca lỗi Điển hình:
1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công) — G02 multi-hop:**
   - *Question:* "Which startups were founded by former Microsoft employees and later received investment from Google?"
   - *Flat RAG: comp=1, faith=1, mhop=1 | GraphRAG: comp=5, faith=5, mhop=5*
   - *Tại sao Flat RAG thất bại?* Vector search chỉ tìm được chunks liên quan đến Microsoft HOẶC Google, nhưng không kết nối được chuỗi logic "Microsoft employee → founded startup → Google invested". Câu hỏi multi-hop cần traversal qua ≥2 edges.
   - *GraphRAG đã giải quyết như thế nào?* Seed matching tìm ra node Microsoft và Google, graph traversal (BFS 2-hop) khám phá các edges FOUNDED, WORKED_AT, INVESTED_IN để tìm path kết nối 2 entities qua startup trung gian.

2. **Ca lỗi GraphRAG thất bại — G04 multi-hop:**
   - *Question:* "Find a company invested in by a major technology company that also developed a named AI technology"
   - *Flat RAG: comp=5, faith=5, mhop=5 | GraphRAG: comp=1, faith=1, mhop=1*
   - *Nguyên nhân:* Graph quá thưa (chỉ 63 edges từ 400 chunks extraction) → seed matching có thể không tìm được entities đúng, hoặc path INVESTED_IN → DEVELOPED không tồn tại trong graph. Flat RAG thành công vì vector search tìm được chunks đề cập trực tiếp cả 2 quan hệ trong text liền kề.
   - *Đề xuất khắc phục:* Tăng EXTRACTION_MAX_CHUNKS (>400), cải thiện NER/RE prompt để tăng recall, hoặc bổ sung edge types (e.g., COMPETES_WITH).

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:**

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:**
  - *GraphRAG*: Tốt hơn cho cross-doc và multi-hop (+0.6~0.8 điểm), nhưng latency cao hơn 50% (11.4s vs 7.6s) do graph traversal và seed extraction call thêm API. Token usage thực tế tương đương hoặc thấp hơn nhờ context graph có cấu trúc.
  - *Flat RAG*: Nhanh hơn, đơn giản hơn, phù hợp với factoid questions. Nhưng thiếu khả năng reasoning multi-hop.
  - *Kết luận*: Hybrid approach (graph + vector) cho kết quả tốt nhất tổng thể.

- **Quyết định từ chối AI Coding Agent:** Trong quá trình làm, Agent đề xuất dùng model `llama-3.3-70b-versatile` nhưng model này đã bị Groq gỡ bỏ. Phải chuyển sang `openai/gpt-oss-120b` rồi `qwen/qwen3.6-27b` do rate limit 200K tokens/day. Cần phải kiểm tra model availability trước khi chạy pipeline dài.

- **Giải pháp scale 350MB (~100K bài báo):**
  - **Bottleneck đầu tiên:** NER/RE Extraction qua LLM API — với 100K bài → ~30K chunks → 7,500 API calls → rate limit và chi phí.
  - **Giải pháp:**
    1. Async batch extraction với worker queue + multiple API keys
    2. HNSW index (thay IndexFlatIP) cho Entity Resolution trên triệu entities
    3. Community Partitioning trước traversal để giới hạn BFS scope
    4. Streaming ingestion vào Neo4j thay vì bulk UNWIND
    5. Caching embeddings vào disk (FAISS index serialization)

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | LLM-based coref hoạt động tốt cho đa số chunks, nhưng batch_size=5 đôi khi gây nhầm cross-chunk. Fallback text gốc khi fail là thiết kế tốt. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | 3 node types + 8 relations đủ cho tech news. Guard ngăn hallucinated relation types từ LLM. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | UNWIND + MERGE pattern hiệu quả, batch_size=1000 phù hợp. Edge provenance check (source_chunk_id + published_date) đảm bảo traceability. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | Union-Find O(α(n)) rất nhanh. 2-layer guard (vector threshold 0.90 + lexical ratio 0.72) hiệu quả. MANUAL_ALIASES cho tech giants cần thiết. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | `SUPER_NODE_DEGREE=100`, `SUPER_NODE_EDGE_CAP=50`, `GLOBAL_EDGE_CAP=250`. Thiết kế BFS + temporal sort + cap là pattern chuẩn cho production. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | 3 metrics (comprehensiveness, faithfulness, multi_hop_reasoning) trên 5-point scale. Golden dataset 5 câu hỏi (factoid, multi-hop, cross-doc). |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Model `llama-3.3-70b-versatile` bị Groq deprecate giữa chừng → toàn bộ 100 NER/RE batches fail → 0 triples. Phải debug bằng cách list available models, test JSON mode, switch sang `openai/gpt-oss-120b`, rồi lại hit rate limit 200K TPD → phải tạo continuation script dùng `qwen/qwen3.6-27b` (model khác = rate limit riêng).
- **Cách xử lý thành công:** 
  1. Thêm model fallback chain trong `groq_chat()` 
  2. Handle empty DataFrame gracefully (guard cho `canonicalize_triples`) 
  3. Tách pipeline thành resumable phases
  4. Test model compatibility (JSON mode) trước khi chạy full pipeline

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Hệ thống Knowledge Base cho VinGroup Tech News Intelligence
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán cần tổng hợp thông tin từ nhiều nguồn tin công nghệ (cross-doc) và trả lời câu hỏi về quan hệ giữa các công ty, nhân sự, công nghệ (multi-hop). GraphRAG + Hybrid RAG là phù hợp nhất.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `Technology`, `Product`, `Event`
  - Relations: `ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`, `COMPETED_WITH`, `LAUNCHED`
- **Chiến lược xử lý Super-node & Entity Resolution:** 
  - HNSW index cho entity matching (thay IndexFlatIP) 
  - Community partitioning bằng Leiden algorithm
  - Temporal sliding window cho super-node edge selection
  - Async extraction pipeline với Redis queue

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Hiểu rõ pipeline, cần thêm thực hành scale lớn |
| Khả năng kiểm soát AI Coding Agent | 4 | Debug model issues, resume pipeline thành công |
| Chất lượng đồ thị tri thức xây dựng | 3 | Đồ thị thưa (113 nodes/63 edges) do rate limit, cần more extraction |
| Khả năng phân tích và debug hệ thống | 4 | Xử lý 3 loại lỗi: column mapping, model deprecation, rate limit |
