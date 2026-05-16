"""Validate CheoBench v2 — quantitative evidence for thesis §3.4 / §4.1.

Runs five experiments on existing data (no extra benchmark runs):
  E1  Coverage analysis              — entity-type + instance + subcategory coverage
  E2  Integrity audit                — related_entities trace to KG inventory
  E3  Difficulty gradient            — intrinsic + system-level difficulty signals
  E4  Discrimination power           — per-case |S_GraphRAG - S_RAG| distribution
  E5  Internal consistency           — Cronbach's alpha on 9 metrics

Outputs LaTeX tables and summary statistics to stdout.

Usage:
    python -m benchmark.validity.dataset_quality
"""
from __future__ import annotations

import math
import statistics
import sys
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from benchmark.validity._common import (
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    FINAL_PATH,
    load_cheobench,
    load_final,
    load_partial,
)


# KG inventory — extracted from data/cheo_entities_summary.md
# (Kept hard-coded so the validator does not need an RDF parser.)
KG_PLAYS = {
    "Chu Mãi Thần",
    "Kim Nham",
    "Lưu Bình - Dương Lễ",
    "Quan Âm Thị Kính",
    "Trinh Nguyên",
    "Trương Viên",
    "Từ Thức",
}

KG_CHARACTERS = {
    # Extracted directly from data/CheoOntology.ttl (charName values) — 38 individuals
    "Châu Long", "Dương Lễ", "Hỷ đồng", "Khoèo", "Lý Trưởng", "Lưu Bình",
    "Mãng Ông (bố Thị Kính)", "Mẹ Mõ (Đốp)", "Mụ Quán", "Nô", "Phù thủy",
    "Phú Ông", "Sùng Bà", "Sùng Ông", "Súy Vân", "Thiện Sỹ", "Thiệt Thê",
    "Thầy Đồ", "Thị Kính", "Thị Màu", "Thị Phương", "Tiên Nữ",
    "Trinh Nguyên", "Trương Mẫu", "Trương Viên", "Trần Phương", "Tuần Ty",
    "Tôn Mạnh", "Tôn Trọng", "Từ Thức", "Đào Huế",
    # Hề characters (seven distinct individuals in the KG, each bound to a specific scene)
    'Hề (Lớp "Tiên Nữ - Đoàn tụ")',
    "Hề (Mụ Quán Trần Phương)", "Hề (Trần Phương vào chùa)",
    "Hề (Đưa bạn đi thi)", "Hề gậy (Hề Theo Thầy)",
    "Hề áo xanh (Dương Lễ tiễn Châu Long đi nuôi bạn)",
    "Hề áo đỏ (Dương Lễ tiễn Châu Long đi nuôi bạn)",
}

KG_ACTORS = {
    "An Chinh", "Bá Dũng", "Bích Vân", "Đăng Toàn", "Đào Dũng", "Hồng Nam",
    "Hồng Thắm", "Huyền Trang", "Huy Toàn", "Hương Dịu", "Khắc Huy",
    "Kiều Oanh", "Kim Liên", "Lê Tuấn", "Mạnh Phóng", "Minh Nhan",
    "Ngọc Ánh", "Ngọc Minh", "Nguyễn Duy", "Phú Kiên", "Phương Mây",
    "Tạ Thị Kim Liên", "Thanh Hương", "Thanh Mai", "Thanh Mạn",
    "Thanh Ngoan", "Thanh Tùng", "Thảo Hiền", "Thu Hòa", "Thu Huyền",
    "Thúy Ngần", "Trần Hải", "Trần Thị Thân", "Trần Vinh", "Trần Xuân Tài",
    "Tử Dương", "Tuấn Cường", "Tuấn Kha", "Tuấn Nghĩa",
    "Vân Quyền", "Văn Quân",
}

KG_SCENES = {
    "Cắt râu", "Đưa bạn đi thi", "Dương Lễ tiễn Châu Long đi nuôi bạn",
    "Hề theo Thầy", 'Lớp "Tiên Nữ - Đoàn tụ"', "Lý trưởng - Mẹ Mõ",
    "Mầu - Nô - Phú Ông", "Mụ quán - Trần Phương", "Phù thủy sợ ma",
    "Súy Vân giả dại", "Thầy đồ dạy học", "Thị Mầu lên chùa",
    "Trần Phương vào chùa", "Tuần Ty - Đào Huế", "Vu quy",
}

KG_OBJECT_PROPERTIES = {
    "forCharacter", "hasAppearance", "hasCharacter", "hasScene",
    "hasVersion", "inVersion", "performedBy",
}


METRICS = [
    "Precision", "Recall", "MAP", "NDCG@10",
    "Faithfulness", "AnswerRelevance",
    "ContextPrecision", "ContextRecall", "ContextEntitiesRecall",
]


def _norm(s: str) -> str:
    """Normalise entity name for matching: lowercase, strip, drop parentheticals."""
    s = s.strip().lower()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return s


def _match_entity(name: str, inventory: set[str]) -> bool:
    n = _norm(name)
    for e in inventory:
        if _norm(e) == n:
            return True
        # allow partial match when KG name contains a parenthetical
        if _norm(e).startswith(n) or n.startswith(_norm(e)):
            return True
    return False


# ────────────────────────────────────────────────────────────────────────────
# E1  Coverage analysis
# ────────────────────────────────────────────────────────────────────────────

def experiment_1_coverage(cases: list[dict]) -> None:
    print("=" * 78)
    print("E1  COVERAGE ANALYSIS")
    print("=" * 78)

    all_entities: set[str] = set()
    for c in cases:
        for e in c["ground_truth"].get("related_entities", []):
            all_entities.add(e)

    def coverage(inventory: set[str]) -> tuple[int, int, set[str]]:
        hit: set[str] = set()
        for e in all_entities:
            for kg in inventory:
                if _norm(kg) == _norm(e) or _norm(kg).startswith(_norm(e)) \
                        or _norm(e).startswith(_norm(kg)):
                    hit.add(kg)
                    break
        return len(hit), len(inventory), hit

    rows = []
    for name, inv in [
        ("Vở chèo",     KG_PLAYS),
        ("Nhân vật",    KG_CHARACTERS),
        ("Diễn viên",   KG_ACTORS),
        ("Trích đoạn",  KG_SCENES),
    ]:
        h, t, _ = coverage(inv)
        pct = 100.0 * h / t
        rows.append((name, h, t, pct))

    print("\nTable: Coverage của các loại thực thể KG")
    print(f"{'Loại thực thể':<18}{'Đã phủ':>10}{'Tổng':>8}{'Tỷ lệ':>10}")
    for name, h, t, pct in rows:
        print(f"{name:<18}{h:>10}{t:>8}{pct:>9.1f}%")

    # Subcategory diversity
    subcats = Counter(c.get("subcategory", "unknown") for c in cases)
    print(f"\nSố lượng subcategory phân biệt: {len(subcats)}")
    for sub, n in sorted(subcats.items(), key=lambda x: -x[1]):
        print(f"  {sub:<30} {n}")

    # LaTeX
    print("\n--- LaTeX ---")
    print(r"\begin{table}[h]\centering")
    print(r"\caption{Mức độ bao phủ của CheoBench v2 trên Knowledge Graph Chèo.}")
    print(r"\label{tab:coverage_entity}")
    print(r"\begin{tabular}{lrrr}\hline")
    print(r"\textbf{Loại thực thể} & \textbf{Đã phủ} & \textbf{Tổng} & \textbf{Tỷ lệ} \\ \hline")
    for name, h, t, pct in rows:
        print(f"{name} & {h} & {t} & {pct:.1f}\\% \\\\")
    print(r"\hline\end{tabular}\end{table}")


# ────────────────────────────────────────────────────────────────────────────
# E2  Integrity audit
# ────────────────────────────────────────────────────────────────────────────

def experiment_2_integrity(cases: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("E2  INTEGRITY AUDIT — related_entities → KG")
    print("=" * 78)

    all_kg = KG_PLAYS | KG_CHARACTERS | KG_ACTORS | KG_SCENES

    total_entities = 0
    missing: list[tuple[str, str]] = []
    for c in cases:
        for e in c["ground_truth"].get("related_entities", []):
            total_entities += 1
            if not _match_entity(e, all_kg):
                missing.append((c["id"], e))

    n_cases_clean = sum(
        1 for c in cases
        if all(_match_entity(e, all_kg) for e in c["ground_truth"].get("related_entities", []))
    )

    print(f"Tổng số related_entities được trích dẫn: {total_entities}")
    print(f"Số thực thể không tìm thấy trong KG       : {len(missing)}")
    print(f"Tỷ lệ câu hỏi có ground truth hợp lệ 100% : "
          f"{n_cases_clean}/{len(cases)} = {100*n_cases_clean/len(cases):.1f}%")

    if missing:
        print("\nCác thực thể bị thiếu (để rà soát thủ công):")
        for cid, e in missing[:20]:
            print(f"  {cid}: {e!r}")
        if len(missing) > 20:
            print(f"  ... và {len(missing) - 20} mục khác")


# ────────────────────────────────────────────────────────────────────────────
# E3  Difficulty gradient
# ────────────────────────────────────────────────────────────────────────────

def experiment_3_difficulty(cases: list[dict], final: dict) -> None:
    print("\n" + "=" * 78)
    print("E3  DIFFICULTY GRADIENT")
    print("=" * 78)

    # (a) Intrinsic signal: #related_entities per category
    cases_by_cat: dict[str, list[int]] = defaultdict(list)
    for c in cases:
        n = len(c["ground_truth"].get("related_entities", []))
        cases_by_cat[c["category"]].append(n)

    print("\n(a) Số thực thể liên quan trung bình trên câu hỏi:")
    print(f"{'Category':<12}{'N câu':>8}{'Avg ents':>12}{'Max':>6}")
    intrinsic = []
    for cat in CATEGORY_ORDER:
        xs = cases_by_cat[cat]
        avg = statistics.mean(xs) if xs else 0.0
        mx = max(xs) if xs else 0
        intrinsic.append((CATEGORY_LABEL[cat], len(xs), avg, mx))
        print(f"{CATEGORY_LABEL[cat]:<12}{len(xs):>8}{avg:>12.2f}{mx:>6}")

    # (b) System signal: GraphRAG & RAG per category, and gap
    by_pipe = {r["pipeline"]: r for r in final["results"]}
    print("\n(b) Điểm hệ thống theo nhóm truy vấn và khoảng cách GraphRAG - RAG:")
    print(f"{'Category':<12}{'GraphRAG':>10}{'RAG':>10}{'Δ':>10}")
    sys_rows = []
    for cat in CATEGORY_ORDER:
        g = by_pipe["GraphRAG"]["by_category"][cat]["S_overall"]
        r = by_pipe["RAG"]["by_category"][cat]["S_overall"]
        sys_rows.append((CATEGORY_LABEL[cat], g, r, g - r))
        print(f"{CATEGORY_LABEL[cat]:<12}{g:>10.3f}{r:>10.3f}{g-r:>+10.3f}")

    print("\n--- LaTeX ---")
    print(r"\begin{table}[h]\centering")
    print(r"\caption{Chỉ số độ khó nội tại và độ khó hệ thống theo nhóm truy vấn.}")
    print(r"\label{tab:difficulty_gradient}")
    print(r"\begin{tabular}{lcccc}\hline")
    print(r"\textbf{Nhóm} & \textbf{\#ents TB} & \textbf{GraphRAG} & \textbf{RAG} & \textbf{Gap} \\ \hline")
    for (lab, n_cases, avg, _), (_, g, r, d) in zip(intrinsic, sys_rows):
        print(f"{lab} & {avg:.2f} & {g:.3f} & {r:.3f} & +{d:.3f} \\\\")
    print(r"\hline\end{tabular}\end{table}")


# ────────────────────────────────────────────────────────────────────────────
# E4  Discrimination power
# ────────────────────────────────────────────────────────────────────────────

def experiment_4_discrimination(partial: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("E4  DISCRIMINATION POWER — per-case |S_GraphRAG - S_RAG|")
    print("=" * 78)

    by_id: dict[str, dict[str, float]] = defaultdict(dict)
    for row in partial:
        s = (row.get("scores") or {}).get("S_overall")
        if s is None:
            continue
        by_id[row["case_id"]][row["pipeline"]] = s

    gaps: list[float] = []
    for cid, d in by_id.items():
        if "GraphRAG" in d and "RAG" in d:
            gaps.append(d["GraphRAG"] - d["RAG"])

    n = len(gaps)
    if not n:
        print("Không có cặp (GraphRAG, RAG) nào khớp được.")
        return

    abs_gaps = [abs(g) for g in gaps]
    avg = statistics.mean(gaps)
    sd = statistics.pstdev(gaps)
    pos = sum(1 for g in gaps if g > 0)
    tie = sum(1 for g in gaps if abs(g) < 1e-6)

    thresholds = [0.05, 0.10, 0.20, 0.30]
    print(f"\nSố case khớp cặp         : {n}")
    print(f"Gap trung bình (G - R)   : {avg:+.3f}")
    print(f"Độ lệch chuẩn             : {sd:.3f}")
    print(f"Số case GraphRAG ≥ RAG   : {pos}/{n} ({100*pos/n:.1f}%)")
    print(f"Số case hoà               : {tie}/{n}")
    print("\nPhân phối |gap|:")
    print(f"{'ngưỡng':<10}{'số case':>10}{'tỷ lệ':>10}")
    for t in thresholds:
        k = sum(1 for x in abs_gaps if x >= t)
        print(f">= {t:<7.2f}{k:>10}{100*k/n:>9.1f}%")

    # Histogram bins
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    counts = [0] * (len(bins) - 1)
    for x in abs_gaps:
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if x >= bins[-1]:
                counts[-1] += 1
    print("\nHistogram |gap|:")
    for i, c in enumerate(counts):
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f})  {c:>3}  {'#'*c}")


# ────────────────────────────────────────────────────────────────────────────
# E5  Internal consistency (Cronbach's alpha)
# ────────────────────────────────────────────────────────────────────────────

def cronbach_alpha(matrix: list[list[float]]) -> float:
    """matrix: n_items × n_subjects  → returns α.

    Following Cronbach (1951):
        α = (k / (k-1)) * (1 - Σ var(item_i) / var(total_score))
    """
    k = len(matrix)
    if k < 2:
        return float("nan")
    n = len(matrix[0])
    var_items = [statistics.pvariance(row) for row in matrix]
    totals = [sum(matrix[i][j] for i in range(k)) for j in range(n)]
    var_total = statistics.pvariance(totals)
    if var_total == 0:
        return float("nan")
    return (k / (k - 1)) * (1 - sum(var_items) / var_total)


def experiment_5_consistency(partial: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("E5  INTERNAL CONSISTENCY — Cronbach's α")
    print("=" * 78)

    # GraphRAG scores matrix: metric → list of per-case scores
    cols: dict[str, list[float]] = {m: [] for m in METRICS}
    for row in partial:
        if row["pipeline"] != "GraphRAG":
            continue
        s = row.get("scores") or {}
        if any(s.get(m) is None for m in METRICS):
            continue
        for m in METRICS:
            cols[m].append(float(s[m]))

    n_subjects = min(len(v) for v in cols.values())
    print(f"Số câu đầy đủ 9 độ đo (GraphRAG): {n_subjects}")

    # Full 9-metric alpha
    mat_all = [cols[m][:n_subjects] for m in METRICS]
    alpha_all = cronbach_alpha(mat_all)

    # Block-wise alpha
    ir_block = ["MAP", "NDCG@10", "Precision", "Recall", "ContextEntitiesRecall"]
    rag_block = ["ContextPrecision", "ContextRecall"]
    gen_block = ["Faithfulness", "AnswerRelevance"]

    def block_alpha(block: list[str]) -> float:
        return cronbach_alpha([cols[m][:n_subjects] for m in block])

    print(f"\nCronbach's α (toàn bộ 9 độ đo)   : {alpha_all:+.3f}")
    print(f"Cronbach's α (IR block, k=5)     : {block_alpha(ir_block):+.3f}")
    print(f"Cronbach's α (Context block, k=2): {block_alpha(rag_block):+.3f}")
    print(f"Cronbach's α (Generation, k=2)   : {block_alpha(gen_block):+.3f}")

    # Pearson correlation matrix (for discussion)
    def pearson(xs: list[float], ys: list[float]) -> float:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
        dy = math.sqrt(sum((b - my) ** 2 for b in ys))
        return num / (dx * dy) if dx * dy else float("nan")

    print("\nMa trận tương quan Pearson giữa các độ đo (GraphRAG):")
    short = [m[:6] for m in METRICS]
    print(" " * 8 + "".join(f"{s:>8}" for s in short))
    for m, row in zip(METRICS, mat_all):
        cells = []
        for m2, row2 in zip(METRICS, mat_all):
            r = pearson(row, row2)
            cells.append(f"{r:>8.2f}")
        print(f"{m[:6]:<8}" + "".join(cells))


# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cases = load_cheobench()
    final = load_final()
    partial = load_partial()

    print(f"CheoBench v2: {len(cases)} câu hỏi")
    print(f"Kết quả baseline: {FINAL_PATH}")
    print(f"Per-case records: {len(partial)} dòng\n")

    experiment_1_coverage(cases)
    experiment_2_integrity(cases)
    experiment_3_difficulty(cases, final)
    experiment_4_discrimination(partial)
    experiment_5_consistency(partial)


if __name__ == "__main__":
    main()
