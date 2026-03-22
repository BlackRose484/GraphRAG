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

QUERY_EXPAND_AND_EXTRACT = """\
Bạn là chuyên gia về Chèo Việt Nam. Với câu hỏi:
"{query}"

Thực hiện đồng thời 2 nhiệm vụ:
1. Mở rộng câu hỏi bằng cách thêm các thuật ngữ Chèo liên quan, ngữ cảnh nghệ thuật
2. Trích xuất thực thể từ câu hỏi — CHỈ sử dụng danh sách hợp lệ dưới đây

=== DANH SÁCH THỰC THỂ HỢP LỆ ===

VỞ CHÈO:
Chu Mãi Thần, Kim Nham, Lưu Bình - Dương Lễ, Quan Âm Thị Kính, Trinh Nguyên, Trương Viên, Từ Thức

TRÍCH ĐOẠN:
Cắt râu, Đưa bạn đi thi, Dương Lễ tiễn Châu Long đi nuôi bạn, Hề theo Thầy,
Lớp "Tiên Nữ - Đoàn tụ", Lý trưởng - Mẹ Mõ, Mầu - Nô - Phú Ông,
Mụ quán - Trần Phương, Phù thủy sợ ma, Súy Vân giả dại, Thầy đồ dạy học,
Thị Mầu lên chùa, Trần Phương vào chùa, Tuần Ty - Đào Huế, Vu quy

NHÂN VẬT:
Châu Long, Đào Huế, Dương Lễ, Hề (Đưa bạn đi thi), Hề áo đỏ (Dương Lễ tiễn Châu Long đi nuôi bạn),
Hề áo xanh (Dương Lễ tiễn Châu Long đi nuôi bạn), Hề (Lớp "Tiên Nữ - Đoàn tụ),
Hề (Mụ Quán Trần Phương), Hề (Trần Phương vào chùa), Hề gậy (Hề Theo Thầy),
Hỷ đồng, Khoèo, Lưu Bình, Lý Trưởng, Mãng Ông (bố Thị Kính), Mẹ Mõ (Đốp),
Mụ Quán, Nô, Phú Ông, Phù thủy, Sùng Bà, Sùng Ông, Súy Vân, Thầy Đồ,
Thị Kính, Thị Màu, Thị Phương, Thiện Sỹ, Thiệt Thê, Tiên Nữ,
Tôn Mạnh, Tôn Trọng, Trần Phương, Trinh Nguyên, Trương Mẫu, Trương Viên, Từ Thức, Tuần Ty

DIỄN VIÊN:
An Chinh, Bá Dũng, Bích Vân, Đăng Toàn, Đào Dũng, Hồng Nam, Hồng Thắm,
Huy Toàn, Huyền Trang, Hương Dịu, Khắc Huy, Kiều Oanh, Kim Liên, Lê Tuấn,
Mạnh Phóng, Minh Nhan, Ngọc Ánh, Ngọc Minh, Nguyễn Duy, Phú Kiên, Phương Mây,
Tạ Thị Kim Liên, Thanh Hương, Thanh Mai, Thanh Mạn, Thanh Ngoan, Thanh Tùng,
Thảo Hiền, Thu Hòa, Thu Huyền, Thúy Ngần, Trần Hải, Trần Thị Thân, Trần Vinh,
Trần Xuân Tài, Tuấn Cường, Tuấn Kha, Tuấn Nghĩa, Tử Dương, Văn Quân, Vân Quyền

=== QUY TẮC TRÍCH XUẤT ===
- CHỈ extract thực thể có trong danh sách trên
- Nếu câu hỏi dùng tên viết tắt/biến thể → map về tên chuẩn (VD: "Thị Mầu" → "Thị Màu", "Quan Âm" → "Quan Âm Thị Kính")
- Nếu không có thực thể nào khớp → trả về mảng rỗng []
- KHÔNG tự tạo thêm thực thể không có trong danh sách

CHỈ trả về JSON duy nhất (không giải thích, không markdown):
{{"expanded": "câu hỏi mở rộng đặt ở đây", "entities": {{"characters": [], "actors": [], "plays": [], "scenes": []}}}}"""


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
