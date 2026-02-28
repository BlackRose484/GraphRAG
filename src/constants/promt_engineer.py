"""
Prompt templates for all LLM calls in the GraphRAG pipeline.

All prompt strings live here — no f-string templates with inline prose
anywhere else in the codebase. Each template uses {placeholder} format
so callers do: PROMPT.format(query=query, context=context, ...).
"""

from __future__ import annotations


# ── G-Retrieval: Query Processing ─────────────────────────────────────────────

QUERY_EXPAND = """\
Bạn là chuyên gia về Chèo Việt Nam.
Mở rộng câu hỏi sau bằng cách thêm các thuật ngữ liên quan, ngữ cảnh:

Câu hỏi: {query}

Trả về câu hỏi mở rộng (chỉ text, không giải thích):"""

QUERY_DECOMPOSE = """\
Phân tách câu hỏi phức tạp thành các câu hỏi con đơn giản hơn:

Câu hỏi: {query}

Trả về MỖI câu hỏi con trên MỘT dòng (không đánh số, không giải thích):"""


# ── G-Retrieval: Entity Extraction ────────────────────────────────────────────

ENTITY_EXTRACT = """\
Trích xuất TÊN RIÊNG từ câu hỏi về Chèo. CHỈ trả về JSON:
"{query}"

{{"characters": ["tên nhân vật"], "actors": ["tên diễn viên"], "plays": ["tên vở"], "scenes": ["tên cảnh"]}}"""


# ── G-Generation: Pre-generation ─────────────────────────────────────────────

PRE_GENERATION = """\
{context}

# Câu hỏi
{query}

# Yêu cầu
Trả lời dựa trên thông tin graph ở trên. Nhớ cite cụ thể tên từ graph:"""


# ── G-Generation: Mid-generation ─────────────────────────────────────────────

MID_GENERATION = """\
{context}

# Hướng dẫn trả lời có cấu trúc

Câu hỏi: {query}

Yêu cầu trả lời theo format:

1. **Thông tin từ graph**: (trích dẫn facts quan trọng từ dữ liệu)
{key_facts}

2. **Phân tích**: (giải thích dựa trên facts trên)

3. **Kết luận**: (tóm tắt ngắn gọn)

Trả lời:"""


# ── G-Generation: Post-generation ────────────────────────────────────────────

POST_INITIAL = """\
Trả lời ngắn gọn câu hỏi về Chèo (nếu không biết, nói "Không chắc chắn"):

Câu hỏi: {query}

Trả lời ngắn (1-2 câu):"""

POST_REFINE = """\
# Nhiệm vụ: Cải thiện câu trả lời dựa trên dữ liệu chính xác

## Câu hỏi gốc
{query}

## Câu trả lời ban đầu (có thể chưa chính xác)
{initial_answer}

## Dữ liệu chính xác từ Knowledge Graph
{verification_data}

## Hướng dẫn cải thiện
1. Kiểm tra câu trả lời ban đầu với dữ liệu thực tế
2. Sửa lỗi nếu có, thêm chi tiết cụ thể từ graph
3. Cite tên riêng chính xác từ dữ liệu

Ví dụ cải thiện:
- Ban đầu: "Bích Vân là diễn viên nổi tiếng"
- Cải thiện: "Theo dữ liệu, diễn viên Bích Vân thể hiện nhân vật Thúy Kiều trong vở Kim Vân Kiều"

===== CÂU TRẢ LỜI CUỐI CÙNG =====
(CHỈ viết câu trả lời, không giải thích quá trình cải thiện)"""


# ── Context builder header ────────────────────────────────────────────────────

CONTEXT_HEADER = "# Thông tin từ Knowledge Graph về Chèo\n"
CONTEXT_KEY_FACTS_HEADER = "## Tóm tắt quan trọng:\n{key_facts}\n"
CONTEXT_SECTION = "## {title}:\n{content}\n"
