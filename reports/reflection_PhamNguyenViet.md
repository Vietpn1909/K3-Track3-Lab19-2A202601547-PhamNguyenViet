# Reflection & Action Plan — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Nguyên Việt — 2A202601547  
**Ngày:** 19/08/2026  

---

## 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|----------------------------|
| **Conservative Coreference** | Module 1 (Cell 1.7) | `resolve_coref_batch()`, `run_coref()` | LLM-based coref hoạt động tốt cho đa số chunks nhưng batch_size=5 đôi khi gây nhầm cross-chunk. Fallback text gốc khi batch fail là thiết kế defensive tốt. |
| **Schema & Allowlist Guard** | Module 2 (Cell 2.1) | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `EXTRACT_SYSTEM` | 3 node types (Company, Person, Technology) + 8 relations là schema đủ gọn cho tech news. Guard ngăn LLM hallucinate relation types ngoài allowlist. |
| **Bulk Cypher Ingestion** | Module 2 (Cell 2.3) | `bulk_insert_nodes()`, `bulk_insert_edges()` | UNWIND + MERGE pattern hiệu quả, batch_size=1000 phù hợp. Edge provenance (source_chunk_id + published_date) đảm bảo traceability 100%. |
| **Entity Resolution & Union-Find** | Module 3 (Cell 2.2) | `build_resolution_map()`, `UF`, `merge_guard()` | Union-Find O(α(n)) rất nhanh. 2-layer guard (vector 0.90 + lexical 0.72) hiệu quả. MANUAL_ALIASES cho tech giants (MSFT→Microsoft) cần thiết. |
| **Super-node Degree Cap** | Module 4 (Cell 3.3) | `retrieve_graph_context()`, `node_degree()`, `recent_edges()` | Constants: SUPER_NODE_DEGREE=100, SUPER_NODE_EDGE_CAP=50, GLOBAL_EDGE_CAP=250. BFS + temporal sort + cap là pattern chuẩn production. |
| **LLM-as-a-Judge Evaluation** | Module 5 (Cell 4.2-4.4) | `judge_answer()`, `run_evaluation()`, `comparison_table()` | 3 metrics (comprehensiveness, faithfulness, multi_hop_reasoning) trên 5-point scale. Checkpoint mỗi câu hỏi cho fault tolerance. |

---

## 2. Quá trình Debugging & Bài học

### Lỗi kỹ thuật phức tạp nhất gặp phải:
**Chuỗi 3 lỗi liên tiếp về Groq API:**

1. **Lỗi 1 — Column mapping:** Dataset HackerNoon có cột `description` thay vì `text`. Pipeline crash ở `pick_col()`.
   - *Giải pháp:* Thêm `"description"` vào candidates list.

2. **Lỗi 2 — Model deprecation:** Model `llama-3.3-70b-versatile` bị Groq gỡ bỏ (trả về 404). Toàn bộ 100 NER/RE batches fail → 0 triples → empty DataFrame crash.
   - *Giải pháp:* List available models qua API (`c.models.list()`), test JSON mode compatibility, chuyển sang `openai/gpt-oss-120b`.

3. **Lỗi 3 — Rate limit TPD:** Model `openai/gpt-oss-120b` có limit 200K tokens/day. Sau Phase 1-2 (~200K tokens), Phase 3 bị block hoàn toàn.
   - *Giải pháp:* Nhận ra rate limit là per-model → tạo continuation script dùng `qwen/qwen3.6-27b` (model khác = rate limit riêng). Thêm model fallback chain + longer retry waits.

### Bài học rút ra:
1. **Defensive programming:** Luôn handle empty DataFrames, add guards cho edge cases.
2. **Model availability:** Không hard-code model names, sử dụng model fallback chain.
3. **Rate limit awareness:** Ước tính token budget trước khi chạy pipeline dài.
4. **Resumable pipelines:** Thiết kế pipeline có checkpoint/resume để tránh mất progress khi crash.

---

## 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** Hệ thống Knowledge Base cho VinGroup Tech News Intelligence
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán cần tổng hợp thông tin từ nhiều nguồn tin công nghệ (cross-doc) và trả lời câu hỏi về quan hệ giữa các công ty, nhân sự, công nghệ (multi-hop). Kết quả Lab 19 cho thấy GraphRAG vượt trội ở cross-doc (+2.0 điểm) → Hybrid RAG (Graph + Vector) là lựa chọn phù hợp nhất.

- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `Technology`, `Product`, `Event`
  - Relations: `ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`, `COMPETED_WITH`, `LAUNCHED`

- **Chiến lược xử lý Super-node & Entity Resolution:**
  - HNSW index cho entity matching (thay IndexFlatIP) — O(log n) vs O(n)
  - Community partitioning bằng Leiden algorithm (thay Greedy Modularity)
  - Temporal sliding window cho super-node edge selection
  - Async extraction pipeline với Redis/Celery queue cho scale 100K+ articles
  - Near-dedup bằng MinHash/LSH trước exact dedup

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Hiểu rõ pipeline end-to-end, cần thêm thực hành scale lớn |
| Khả năng kiểm soát AI Coding Agent | 4 | Debug model issues, resume pipeline, verify outputs thành công |
| Chất lượng đồ thị tri thức xây dựng | 3 | Đồ thị thưa (113 nodes/63 edges) do rate limit, cần more extraction |
| Khả năng phân tích và debug hệ thống | 4 | Xử lý thành công 3 loại lỗi: column mapping, model deprecation, rate limit |
