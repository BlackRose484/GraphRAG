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
Mở rộng câu hỏi sau bằng cách thêm các thuật ngữ liên quan, ngữ cảnh.

Lưu ý: hệ thống có dữ liệu về vở chèo, trích đoạn, phiên bản trình diễn, nhân vật
(loại Đào / Hề / Kép / Mụ / Lão), diễn viên, trang phục theo loại vai, và cảm xúc
(emotion) của nhân vật trong từng lần xuất hiện — nên bạn có thể bổ sung từ khóa
liên quan tới trang phục hoặc cảm xúc nếu phù hợp.

Câu hỏi: {query}

Trả về câu hỏi mở rộng (chỉ text, không giải thích):"""

QUERY_DECOMPOSE = """\
Phân tách câu hỏi phức tạp thành các câu hỏi con đơn giản hơn:

Câu hỏi: {query}

Trả về MỖI câu hỏi con trên MỘT dòng (không đánh số, không giải thích):"""


QUERY_DECOMPOSE_AND_EXTRACT = """\
Bạn là chuyên gia về Chèo Việt Nam. Với câu hỏi:
"{query}"

Thực hiện đồng thời 2 nhiệm vụ:
1. Phân rã câu hỏi gốc thành 2-4 câu hỏi con đơn lẻ, mỗi câu hỏi con tập trung vào một khía cạnh duy nhất của câu hỏi gốc.
2. Với MỖI câu hỏi con, trích xuất các thực thể có trong câu hỏi đó — CHỈ sử dụng danh sách hợp lệ dưới đây.

=== DANH SÁCH THỰC THỂ HỢP LỆ ===

{entity_catalog}

=== QUY TẮC TRÍCH XUẤT ===
- CHỈ extract thực thể có trong danh sách trên
- Map biến thể tên về tên chuẩn (VD: "Thị Mầu" → "Thị Màu", "Quan Âm" → "Quan Âm Thị Kính")
- KHÔNG tự tạo thực thể không có trong danh sách
- Nếu câu hỏi con không nhắc đến thực thể cụ thể nào, trả về mảng rỗng []

CHỈ trả về JSON duy nhất (không giải thích, không markdown):
{{"decomposed": [
    {{"question": "câu hỏi con 1",
      "entities": {{"characters": [], "actors": [], "plays": [], "scenes": []}}}},
    {{"question": "câu hỏi con 2",
      "entities": {{"characters": [], "actors": [], "plays": [], "scenes": []}}}}
  ]}}"""

QUERY_EXPAND_AND_EXTRACT = """\
Bạn là chuyên gia về Chèo Việt Nam. Với câu hỏi:
"{query}"

Đồ thị tri thức hiện có các loại thông tin sau:
- Vở chèo (Play), trích đoạn (Scene), phiên bản trình diễn (Version)
- Nhân vật (Character) phân loại theo Đào / Hề / Kép / Mụ / Lão (property roleType) và subType (Chín / Pha / Áo dài / ...)
- Diễn viên (Actor) và ai đóng vai gì trong phiên bản nào
- Trang phục (Costume) theo loại vai, biết diễn viên nào mặc trang phục nào
- Cảm xúc (Mood/emotion) nhân vật biểu lộ trong từng lần xuất hiện
- Cử chỉ khuôn mặt (FaceGesture) tương ứng từng cảm xúc

Thực hiện đồng thời 3 nhiệm vụ:
1. Mở rộng câu hỏi bằng cách thêm các thuật ngữ Chèo liên quan, ngữ cảnh nghệ thuật
2. Trích xuất thực thể từ câu hỏi — CHỈ sử dụng danh sách hợp lệ dưới đây
3. Phân loại câu hỏi vào MỘT trong ba mức:
   - "Local"     : tra cứu trực tiếp 1-2 thực thể, ít bước suy luận
                   (VD: "Ai đóng vai Thị Màu?", "Vở Quan Âm Thị Kính có những trích đoạn nào?")
   - "Community" : tổng hợp/giao tập trên cụm thực thể có liên kết chặt
                   (VD: "Diễn viên nào đóng cả 2 vở Quan Âm Thị Kính và Kim Nham?")
   - "Global"    : so sánh/tổng hợp xuyên nhiều vở hoặc trên toàn bộ KG
                   (VD: "So sánh số phận Súy Vân và Thị Kính", "Liệt kê tất cả vở chèo")

=== DANH SÁCH THỰC THỂ HỢP LỆ ===

{entity_catalog}

=== QUY TẮC TRÍCH XUẤT ===
- CHỈ extract thực thể có trong danh sách trên
- Nếu câu hỏi dùng tên viết tắt/biến thể → map về tên chuẩn (VD: "Thị Mầu" → "Thị Màu", "Quan Âm" → "Quan Âm Thị Kính")
- KHÔNG tự tạo thêm thực thể không có trong danh sách

=== QUY TẮC ĐẶC BIỆT: CÂU HỎI TOÀN CỤC ===
- Nếu câu hỏi yêu cầu tổng hợp, so sánh, liệt kê, hoặc thống kê trên TOÀN BỘ KG (ví dụ: "diễn viên nào xuất hiện nhiều nhất", "so sánh nhân vật nữ chính các vở", "có bao nhiêu vở", "tổng kết toàn bộ")
- → query_type = "Global", liệt kê TẤT CẢ tên vở chèo vào "plays"
- Nếu không có thực thể cụ thể nào khớp VÀ câu hỏi không phải dạng toàn cục → trả về mảng rỗng [] và query_type phù hợp

CHỈ trả về JSON duy nhất (không giải thích, không markdown):
{{"expanded": "câu hỏi mở rộng đặt ở đây", "entities": {{"characters": [], "actors": [], "plays": [], "scenes": []}}, "query_type": "Local"}}"""


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
Trả lời dựa trên thông tin đồ thị ở trên. Nhớ trích dẫn cụ thể tên từ đồ thị:"""


# ── G-Generation: Mid-generation ─────────────────────────────────────────────

MID_GENERATION = """\
{context}

# Hướng dẫn trả lời có cấu trúc

Câu hỏi: {query}

Yêu cầu trả lời theo format:

1. **Thông tin từ đồ thị**: (trích dẫn dữ kiện quan trọng từ dữ liệu)
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

## Dữ liệu chính xác từ đồ thị tri thức
{verification_data}

## Hướng dẫn cải thiện
1. Kiểm tra câu trả lời ban đầu với dữ liệu thực tế
2. Sửa lỗi nếu có, thêm chi tiết cụ thể từ đồ thị
3. Cite tên riêng chính xác từ dữ liệu

Ví dụ cải thiện:
- Ban đầu: "Bích Vân là diễn viên nổi tiếng"
- Cải thiện: "Theo dữ liệu, diễn viên Bích Vân thể hiện nhân vật Thúy Kiều trong vở Kim Vân Kiều"

===== CÂU TRẢ LỜI CUỐI CÙNG =====
(CHỈ viết câu trả lời, không giải thích quá trình cải thiện)"""


# ── Context builder header ────────────────────────────────────────────────────

CONTEXT_HEADER = "# Thông tin từ đồ thị tri thức về Chèo\n"
CONTEXT_KEY_FACTS_HEADER = "## Tóm tắt quan trọng:\n{key_facts}\n"
CONTEXT_SECTION = "## {title}:\n{content}\n"
