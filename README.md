# 🎭 GraphRAGv2 — Hệ thống Hỏi đáp Nghệ thuật Chèo

> **Đề tài đồ án tốt nghiệp**: _XÂY DỰNG GRAPHRAG CHO LLM PHỤC VỤ NGHIÊN CỨU NGHỆ THUẬT CHÈO_
> Kết hợp **đồ thị tri thức (Neo4j)** với **mô hình ngôn ngữ lớn (LLM)** để trả lời câu hỏi về nghệ thuật Chèo truyền thống Việt Nam.

---

## 📋 Mục lục

1. [Tính năng chính](#-tính-năng-chính)
2. [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
3. [Cài đặt nhanh (5 bước)](#-cài-đặt-nhanh-5-bước)
4. [Hướng dẫn cài đặt chi tiết](#-hướng-dẫn-cài-đặt-chi-tiết)
   - [Bước 1: Clone & chuẩn bị môi trường Python](#bước-1-clone--chuẩn-bị-môi-trường-python)
   - [Bước 2: Cài đặt thư viện](#bước-2-cài-đặt-thư-viện)
   - [Bước 3: Cài đặt Neo4j](#bước-3-cài-đặt-neo4j)
   - [Bước 4: Cấu hình `.env`](#bước-4-cấu-hình-env)
   - [Bước 5: Nạp ontology Chèo vào Neo4j](#bước-5-nạp-ontology-chèo-vào-neo4j)
   - [Bước 6: Chạy ứng dụng](#bước-6-chạy-ứng-dụng)
5. [Cấu hình nâng cao](#-cấu-hình-nâng-cao)
6. [Triển khai bằng Docker](#-triển-khai-bằng-docker)
7. [Triển khai lên Google Cloud Run](#-triển-khai-lên-google-cloud-run)
8. [Cấu trúc dự án](#-cấu-trúc-dự-án)
9. [Câu lệnh thường dùng](#-câu-lệnh-thường-dùng)
10. [Xử lý sự cố thường gặp](#-xử-lý-sự-cố-thường-gặp)

---

## ✨ Tính năng chính

- **4 chiến lược truy xuất** đa mức trên đồ thị: `nodes`, `triplets`, `paths`, `subgraph`
- **3 chiến lược sinh câu trả lời**: `pre-generation`, `mid-generation`, `post-generation`
- **Auto-routing**: tự động chọn chiến lược phù hợp với từng loại câu hỏi
- **Đa nhà cung cấp LLM** qua LiteLLM: Google Gemini, Anthropic Claude, OpenAI, Ollama (local)
- **Vector RAG baseline** để đối chứng với GraphRAG
- **UI Streamlit 11 trang**: demo, so sánh, benchmark, trực quan hóa đồ thị, user study
- **Bộ benchmark CheoBench_v2** (100 câu hỏi: 35 Local / 32 Community / 33 Global) + 9 độ đo (IR: MAP, NDCG@10, Precision, Recall; RAGAS: Context Precision/Recall/EntitiesRecall, Faithfulness, Answer Relevance)
- **Chế độ Demo & Guest** cho khảo sát người dùng và quay video demo

---

## 💻 Yêu cầu hệ thống

| Thành phần       | Phiên bản tối thiểu     | Ghi chú                       |
| ---------------- | ----------------------- | ----------------------------- |
| **Python**       | 3.11                    | Khuyến nghị dùng `conda`      |
| **Neo4j**        | 5.14 trở lên            | Community Edition là đủ       |
| **RAM**          | 4 GB                    | 8 GB nếu chạy embedding local |
| **Dung lượng**   | ~2 GB                   | Bao gồm thư viện + dữ liệu    |
| **Hệ điều hành** | Windows / Linux / macOS | Đã test trên Windows 10       |

**Bắt buộc phải có ít nhất 1 trong các API key sau** (hoặc Ollama local):

- Google Gemini API Key — [lấy tại đây](https://aistudio.google.com/app/apikey) _(khuyến nghị, có gói miễn phí)_
- Anthropic Claude API Key
- OpenAI API Key

---

## ⚡ Cài đặt nhanh (5 bước)

Dành cho người đã quen với Python + Neo4j. Người mới nên xem [phần chi tiết](#-hướng-dẫn-cài-đặt-chi-tiết) bên dưới.

```bash
# 1. Clone
git clone <repo-url> GraphRAGv2 && cd GraphRAGv2

# 2. Tạo môi trường conda + cài thư viện
conda create -n graphrag python=3.11 -y
conda activate graphrag
pip install -r requirements.txt

# 3. Cấu hình
cp .env.example .env
# → Mở .env và điền GEMINI_API_KEY + NEO4J_PASSWORD

# 4. Nạp ontology Chèo vào Neo4j (Neo4j phải đang chạy)
python scripts/reload_neo4j.py

# 5. Chạy ứng dụng
streamlit run main.py
```

Mở trình duyệt tại **http://localhost:8501** để sử dụng.

---

## 📚 Hướng dẫn cài đặt chi tiết

### Bước 1: Clone & chuẩn bị môi trường Python

```bash
git clone <repo-url> GraphRAGv2
cd GraphRAGv2
```

**Tạo môi trường ảo** (khuyến nghị dùng `conda`):

```bash
conda create -n graphrag python=3.11 -y
conda activate graphrag
```

> 💡 Mỗi lần mở terminal mới, nhớ chạy lại `conda activate graphrag` trước khi làm việc với dự án.

Nếu không dùng conda, có thể dùng `venv`:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

---

### Bước 2: Cài đặt thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Quá trình này sẽ cài các thư viện chính:

| Nhóm    | Thư viện                                            |
| ------- | --------------------------------------------------- |
| Đồ thị  | `neo4j==5.14.1`, `rdflib==7.0.0`                    |
| LLM     | `litellm>=1.56.0`                                   |
| Vector  | `chromadb==0.4.18`, `sentence-transformers==2.2.2`  |
| UI      | `streamlit==1.29.0`, `pyvis==0.3.2`                 |
| Dữ liệu | `pandas`, `numpy`, `networkx`, `plotly`, `pydantic` |

**Cài thêm SDK của nhà cung cấp LLM bạn dùng** (tùy chọn — LiteLLM gọi qua HTTP, nên SDK chỉ giúp tăng tốc một số tính năng):

```bash
# Nếu dùng Gemini (khuyến nghị)
pip install google-generativeai

# Nếu dùng Claude
pip install anthropic

# Nếu dùng OpenAI
pip install openai
```

---

### Bước 3: Cài đặt Neo4j

Có **3 lựa chọn**, hãy chọn 1:

#### Cách A — Neo4j Desktop (dễ nhất cho người mới)

1. Tải tại **https://neo4j.com/download/**
2. Cài đặt và mở **Neo4j Desktop**
3. Tạo project mới → tạo database mới với version **5.14+**
4. Đặt mật khẩu (ghi nhớ — sẽ dùng cho `.env`)
5. Bấm **Start** để chạy database (mặc định cổng `bolt://localhost:7687`)

#### Cách B — Neo4j chạy bằng Docker

```bash
docker run -d \
  --name neo4j-cheo \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password_here \
  -v neo4j_data:/data \
  neo4j:5.14
```

Mở browser admin tại **http://localhost:7474** để kiểm tra.

#### Cách C — Neo4j AuraDB (cloud, miễn phí)

1. Đăng ký tại **https://neo4j.com/cloud/aura/**
2. Tạo **AuraDB Free** instance
3. Tải file credentials → lưu URI/user/password để dán vào `.env`

---

### Bước 4: Cấu hình `.env`

Sao chép file mẫu:

```bash
cp .env.example .env
```

Mở `.env` và **điền tối thiểu 2 thông tin**:

```bash
# ── LLM (mặc định dùng Gemini) ───────────────────────────────────────
GEMINI_API_KEY=AIza...your_real_key_here
LLM_MODEL=gemini/gemini-2.0-flash
LLM_EMBEDDING_MODEL=gemini/text-embedding-004

# ── Neo4j ────────────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

**Nếu dùng LLM khác**, mở comment khối tương ứng trong `.env` và comment khối Gemini lại.

> 💡 File `.env` đã có sẵn trong `.gitignore` — sẽ không bị push lên Git.

---

### Bước 5: Nạp ontology Chèo vào Neo4j

Đảm bảo Neo4j đang chạy, sau đó:

```bash
python scripts/reload_neo4j.py
```

Lệnh này sẽ:

1. **Xóa** toàn bộ dữ liệu cũ trong Neo4j
2. **Nạp lại** từ file [`data/CheoOntology_v4.ttl`](data/CheoOntology_v4.ttl) (~300 KB RDF/Turtle)
3. In ra summary: số node, số quan hệ, danh sách label & relation type

Kết quả thành công sẽ trông như:

```
[info] Reloading Neo4j from data/CheoOntology_v4.ttl
[done] LoadResult(nodes_created=..., relationships_created=...)

Total nodes:         ...
Total relationships: ...
Node labels (N): ['Actor', 'Character', 'Play', 'Scene', 'Mood', ...]
Rel types  (M): ['HAS_CHARACTER', 'HAS_SCENE', 'PERFORMED_BY', ...]
```

**Muốn nạp file ontology khác?**

```bash
python scripts/reload_neo4j.py data/CheoOntology.ttl
```

---

### Bước 6: Chạy ứng dụng

```bash
streamlit run main.py
```

Streamlit sẽ mở trình duyệt tại **http://localhost:8501**.

Sidebar bên trái có **11 trang**:

| Trang               | Mục đích                                          |
| ------------------- | ------------------------------------------------- |
| 🏠 Giới thiệu       | Tổng quan dự án                                   |
| 👋 Chào mừng        | Hướng dẫn nhanh cho người mới                     |
| ⚖️ So sánh          | Đặt câu hỏi → xem GraphRAG vs Vector RAG cùng lúc |
| 🔍 GraphRAG         | Demo GraphRAG + xem chi tiết quá trình retrieval  |
| 📚 RAG              | Demo Vector RAG baseline                          |
| 💬 Chat             | Giao diện hội thoại                               |
| 🔗 Neo4j            | Trực quan hóa đồ thị bằng Pyvis                   |
| 📊 Benchmark        | Chạy đánh giá hàng loạt trên CheoBench            |
| 🧪 Thử nghiệm       | Pha Experiment (cho user study)                   |
| 📋 Đánh giá ưu tiên | Pha Preference (cho user study)                   |
| 📈 Thống kê         | Dashboard phân tích kết quả                       |

---

## 🔧 Cấu hình nâng cao

### Chỉnh LLM tuning

```bash
# Trong .env
LLM_TIMEOUT=300         # giây
LLM_MAX_RETRIES=3
LLM_RETRY_DELAY=2.0     # giây
LLM_TEMPERATURE=0.7     # 0.0 = deterministic, 1.0 = sáng tạo
```

### Cho phép nhiều model trong dropdown Benchmark

```bash
LLM_MODELS_AVAILABLE=openai/gpt-5-mini,anthropic/claude-3-5-haiku-20241022,gemini/gemini-2.0-flash,ollama/llama3.2
```

Model nào không có API key tương ứng sẽ tự động bị ẩn khỏi dropdown.

### Logging

```bash
LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_TO_FILE=true            # Ghi log ra thư mục logs/
LOG_TO_STDOUT=true          # In log ra console
```

### Guest mode (cho khảo sát công khai)

Khi triển khai cho người dùng làm khảo sát, chỉ hiển thị 3 trang an toàn:

```bash
GUEST_MODE=1
ADMIN_PASSWORD=change-me     # Cho phép admin mở khóa toàn bộ app qua sidebar
```

- Người dùng thông thường chỉ thấy: **Chào mừng**, **Thử nghiệm**, **Đánh giá ưu tiên**
- Admin nhập mật khẩu vào ô **🔑 Admin** ở sidebar để xem toàn bộ trang

### Demo mode (cho quay video)

```bash
DEMO_MODE=1
```

Khi bật, các câu hỏi đã được pre-generate trong [`benchmark/datasets/demo_cache.json`](benchmark/datasets/demo_cache.json) sẽ trả lời từ cache thay vì gọi LLM thật — tiết kiệm chi phí và đảm bảo tính nhất quán khi demo.

---

## 🐳 Triển khai bằng Docker

### Chạy local bằng Docker Compose

1. Tạo file `.env.production` (giống `.env` nhưng cho production):

```bash
cp .env .env.production
# Sửa NEO4J_URI thành địa chỉ Neo4j thực tế
```

2. Build & chạy:

```bash
docker-compose up -d --build
```

Ứng dụng chạy tại **http://localhost:8080**.

3. Kiểm tra log:

```bash
docker-compose logs -f app
```

4. Dừng:

```bash
docker-compose down
```

> ⚠️ `docker-compose.yml` hiện chỉ định nghĩa service `app`, **không bao gồm Neo4j**. Bạn cần chạy Neo4j riêng (xem [Bước 3 — Cách B](#cách-b--neo4j-chạy-bằng-docker)) và đảm bảo `NEO4J_URI` trong `.env.production` trỏ đúng.

---

## ☁️ Triển khai lên Google Cloud Run

Dự án có sẵn script [`deploy.sh`](deploy.sh) để deploy lên **Cloud Run** (Singapore region).

### Yêu cầu

- Đã cài [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- Đã đăng nhập: `gcloud auth login`
- Có project GCP và đã enable **Cloud Build** + **Cloud Run** API

### Cách chạy

```bash
export GCP_PROJECT_ID=your-gcp-project-id
bash deploy.sh
```

Script sẽ:

1. Build Docker image bằng Cloud Build
2. Push lên `gcr.io/<project>/cheo-graphrag`
3. Deploy lên Cloud Run với **2 GB RAM, 1 CPU, timeout 300s, max 3 instances**
4. Nạp toàn bộ biến môi trường từ `.env.production`
5. In ra URL public

> ⚠️ **Neo4j không được host trên Cloud Run** — bạn cần Neo4j AuraDB hoặc Neo4j tự host ngoài Cloud Run, rồi cấu hình `NEO4J_URI` trong `.env.production`.

---

## 📁 Cấu trúc dự án

```
GraphRAGv2/
├── main.py                       # Streamlit entry point
├── requirements.txt
├── Dockerfile / docker-compose.yml / deploy.sh
├── .env.example                  # Template cấu hình
│
├── src/
│   ├── core/                     # Settings, base abstractions
│   ├── pipeline/                 # GraphRAGPipeline (điều phối tầng)
│   ├── g_retrieval/              # 5 chiến lược truy xuất đồ thị
│   ├── g_generation/             # 3 chiến lược sinh câu trả lời
│   ├── graph_loader/             # Neo4j loader & client
│   ├── rag/                      # Vector RAG baseline
│   ├── constants/                # Enums, prompts (zero magic strings)
│   └── utils/                    # Logger, cache, format converter, email
│
├── ui/                           # 11 trang Streamlit
│   ├── page_home.py / page_intro.py / page_compare.py
│   ├── page_graphrag.py / page_rag.py / page_chat.py
│   ├── page_neo4j.py / page_benchmark.py
│   ├── page_experiment.py / page_preference.py / page_analytics.py
│   ├── model_selector.py
│   └── components.py / graph_visualizer.py
│
├── benchmark/
│   ├── runner.py                 # BenchmarkRunner
│   ├── generate_answers.py
│   ├── datasets/                 # CheoBench_v2.json, demo_cache.json, ...
│   └── metrics/                  # IR, RAGAS, exact, generation, retrieval
│
├── scripts/
│   ├── reload_neo4j.py           # Nạp ontology vào Neo4j
│   ├── build_demo_cache.py       # Build cache cho DEMO_MODE
│   └── fetch_results.py          # Thu thập kết quả user study
│
├── data/
│   ├── CheoOntology_v4.ttl       # Ontology chính (RDF/Turtle)
│   ├── CheoOntology.ttl          # Phiên bản cũ
│   ├── cheo_entities.txt         # Danh mục thực thể (fallback cho EntityCatalog)
│   └── vector_store.pkl          # Vector store cho Vector RAG baseline
│
├── thesis/                       # Luận văn LaTeX (5 chương)
│   ├── chapters/                 # chapter01 → chapter05 + appendix
│   ├── figures/                  # Hình minh họa
│   └── references.bib            # ~150 citations
│
└── logs/                         # File log runtime
```

---

## 🛠 Câu lệnh thường dùng

```bash
# Kích hoạt môi trường
conda activate graphrag

# Chạy ứng dụng
streamlit run main.py

# Nạp lại Neo4j (xóa + nạp ontology mới)
python scripts/reload_neo4j.py

# Build cache cho DEMO_MODE (sau khi sửa pipeline)
python scripts/build_demo_cache.py

# Thu thập kết quả user study
python scripts/fetch_results.py

# Chạy với cấu hình tùy chỉnh
streamlit run main.py --server.port 8888 --server.address 0.0.0.0
```

---

## 📖 Tài liệu thêm

- **Luận văn**: [`thesis/`](thesis/) — LaTeX, 5 chương (Giới thiệu → Cơ sở lý thuyết → Phương pháp đề xuất → Thực nghiệm → Kết luận) + 3 phụ lục
- **Ontology**: [`data/cheo_entities_summary.md`](data/cheo_entities_summary.md) — mô tả các loại thực thể trong ontology Chèo
- **Benchmark dataset**: [`benchmark/datasets/CheoBench_v2.json`](benchmark/datasets/CheoBench_v2.json) — 100 câu hỏi đánh giá (phân loại Local/Community/Global)

---

## 👤 Tác giả

**Nguyễn Ngọc Hưng** — Đồ án tốt nghiệp.
