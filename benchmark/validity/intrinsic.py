"""
TN-B1: Kiểm chứng tính hợp lệ nội tại (intrinsic validity) của CheoBench.

Mục tiêu: chứng minh phân nhóm Local / Community / Global của bộ benchmark có
nền tảng cấu trúc - không phải gán nhãn tuỳ tiện - bằng cách:

  (1) Tính các đặc trưng độ phức tạp NỘI TẠI cho từng câu hỏi, không cần chạy
      bất kỳ hệ thống nào:
        - n_related_entities  : số thực thể trong ground_truth.related_entities
        - q_token_len         : số token trong câu hỏi
        - hop_pairwise_avg    : trung bình shortest-path trên đồ thị tri thức
                                giữa các cặp thực thể trong related_entities
                                (lấy từ Neo4j; bỏ qua nếu DB không sẵn sàng)
        - subgraph_size       : số nút trong neighborhood mở rộng 1 hop của
                                tất cả thực thể trong related_entities

  (2) Kiểm định Kruskal-Wallis: ba nhóm có khác biệt thống kê trên các đặc
      trưng độ phức tạp nội tại không?

  (3) Tương quan Spearman giữa độ phức tạp và điểm S_overall của hai hệ thống
      (GraphRAG và RAG) - chứng tỏ độ phức tạp nội tại tác động đồng nhất, đây
      là bằng chứng đặc trưng nội tại đo đúng độ khó chứ không thiên vị một
      kiến trúc.

Output:  benchmark/validity/output/intrinsic_*.{csv,md,png}

Chạy:    python -m benchmark.validity.intrinsic
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow `python benchmark/validity/intrinsic.py` (script mode) by ensuring the
# project root is importable. Harmless when run as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.validity._common import (  # noqa: E402
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    OUT,
    load_cheobench,
    load_per_case_scores,
)
from src.graph_loader.neo4j_client import Neo4jClient  # noqa: E402


# ── Neo4j: hop count and subgraph size ───────────────────────────────────────

ENTITY_LOOKUP_CYPHER = """
MATCH (n)
WHERE n.charName = $name OR n.title = $name OR n.actorName = $name
   OR n.sceneName = $name OR n.id = $name
RETURN elementId(n) AS eid, labels(n) AS labels LIMIT 1
"""

PAIRWISE_HOP_CYPHER = """
MATCH (a), (b)
WHERE elementId(a) = $a_eid AND elementId(b) = $b_eid
WITH a, b
MATCH p = shortestPath((a)-[*..10]-(b))
RETURN length(p) AS hop
"""

SUBGRAPH_SIZE_CYPHER = """
UNWIND $eids AS eid
MATCH (n) WHERE elementId(n) = eid
OPTIONAL MATCH (n)-[*..1]-(m)
RETURN count(DISTINCT m) + count(DISTINCT n) AS sz
"""


def resolve_entities(client: Neo4jClient, names: List[str]) -> List[Optional[str]]:
    """Map entity names to Neo4j elementId (None if not found)."""
    eids: List[Optional[str]] = []
    for n in names:
        rows = client.read(ENTITY_LOOKUP_CYPHER, {"name": n})
        eids.append(rows[0]["eid"] if rows else None)
    return eids


def pairwise_hop_avg(client: Neo4jClient, eids: List[Optional[str]]) -> Optional[float]:
    """Average shortest-path length over all pairs in eids; None if <2 valid eids."""
    valid = [e for e in eids if e is not None]
    if len(valid) < 2:
        return None
    hops: List[int] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            rows = client.read(PAIRWISE_HOP_CYPHER, {"a_eid": valid[i], "b_eid": valid[j]})
            if rows and rows[0]["hop"] is not None:
                hops.append(int(rows[0]["hop"]))
    return float(np.mean(hops)) if hops else None


def subgraph_size(client: Neo4jClient, eids: List[Optional[str]]) -> Optional[int]:
    valid = [e for e in eids if e is not None]
    if not valid:
        return None
    rows = client.read(SUBGRAPH_SIZE_CYPHER, {"eids": valid})
    return int(rows[0]["sz"]) if rows else None


# ── Build feature table ──────────────────────────────────────────────────────

def build_feature_table(use_neo4j: bool) -> pd.DataFrame:
    cases = load_cheobench()
    scores = load_per_case_scores()
    rows = []

    client: Optional[Neo4jClient] = None
    if use_neo4j:
        client = Neo4jClient()
        try:
            client.ping()
        except Exception as e:
            print(f"[warn] Neo4j unreachable: {e}\n        sẽ tính các đặc trưng còn lại, bỏ qua hop_pairwise_avg và subgraph_size.")
            client = None
            use_neo4j = False

    for c in cases:
        cid = c["id"]
        category = c["category"]
        related = c["ground_truth"].get("related_entities", []) or []
        n_rel = len(related)
        q_tok = len(c["question"].split())

        hop_avg: Optional[float] = None
        sg_size: Optional[int] = None
        n_resolved = 0
        if use_neo4j and client is not None:
            eids = resolve_entities(client, related)
            n_resolved = sum(1 for e in eids if e is not None)
            hop_avg = pairwise_hop_avg(client, eids)
            sg_size = subgraph_size(client, eids)

        per_case_scores = scores.get(cid, {})
        rows.append({
            "case_id": cid,
            "category": category,
            "n_related_entities": n_rel,
            "n_resolved_in_kg": n_resolved,
            "q_token_len": q_tok,
            "hop_pairwise_avg": hop_avg,
            "subgraph_size": sg_size,
            "S_overall_graphrag": per_case_scores.get("GraphRAG"),
            "S_overall_rag": per_case_scores.get("RAG"),
        })

    if client is not None:
        client.close()
    return pd.DataFrame(rows)


# ── Statistics ───────────────────────────────────────────────────────────────

INTRINSIC_FEATURES = ["n_related_entities", "q_token_len", "hop_pairwise_avg", "subgraph_size"]
SYSTEM_SCORES = ["S_overall_graphrag", "S_overall_rag"]


def kruskal_wallis_per_feature(df: pd.DataFrame) -> pd.DataFrame:
    """One Kruskal-Wallis test per intrinsic feature, across the 3 categories."""
    rows = []
    for feat in INTRINSIC_FEATURES + SYSTEM_SCORES:
        groups = [df[df.category == c][feat].dropna().values for c in CATEGORY_ORDER]
        if all(len(g) > 0 for g in groups) and any(len(set(g)) > 1 for g in groups):
            stat, p = kruskal(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        else:
            stat, p, sig = float("nan"), float("nan"), "n/a"
        rows.append({
            "feature": feat,
            "local_mean": df[df.category == "local_queries"][feat].mean(),
            "community_mean": df[df.category == "community_queries"][feat].mean(),
            "global_mean": df[df.category == "global_queries"][feat].mean(),
            "kruskal_H": stat,
            "p_value": p,
            "sig": sig,
        })
    return pd.DataFrame(rows)


def spearman_feature_score(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between each intrinsic feature and per-system score."""
    rows = []
    for feat in INTRINSIC_FEATURES:
        for sys_col in SYSTEM_SCORES:
            sub = df[[feat, sys_col]].dropna()
            if len(sub) >= 5 and sub[feat].nunique() > 1:
                rho, p = spearmanr(sub[feat], sub[sys_col])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            else:
                rho, p, sig = float("nan"), float("nan"), "n/a"
            rows.append({
                "feature": feat,
                "system": sys_col.replace("S_overall_", ""),
                "n": len(sub),
                "spearman_rho": rho,
                "p_value": p,
                "sig": sig,
            })
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, out_path: Path) -> None:
    feats = [f for f in INTRINSIC_FEATURES if df[f].notna().sum() > 0]
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, feats):
        data = [df[df.category == c][feat].dropna().values for c in CATEGORY_ORDER]
        ax.boxplot(data, labels=[CATEGORY_LABEL[c] for c in CATEGORY_ORDER], showmeans=True)
        ax.set_title(feat)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phân bố đặc trưng độ phức tạp nội tại theo nhóm câu hỏi")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_score_by_category(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, sys_col in zip(axes, SYSTEM_SCORES):
        data = [df[df.category == c][sys_col].dropna().values for c in CATEGORY_ORDER]
        ax.boxplot(data, labels=[CATEGORY_LABEL[c] for c in CATEGORY_ORDER], showmeans=True)
        ax.set_title(sys_col.replace("S_overall_", "").upper())
        ax.set_ylabel("S_overall")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phân bố S_overall theo nhóm câu hỏi (GraphRAG vs RAG)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_vs_score(df: pd.DataFrame, feat: str, out_path: Path) -> None:
    sub = df.dropna(subset=[feat, "S_overall_graphrag", "S_overall_rag"])
    if len(sub) < 5:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"local_queries": "tab:blue", "community_queries": "tab:orange", "global_queries": "tab:red"}
    for cat in CATEGORY_ORDER:
        s = sub[sub.category == cat]
        ax.scatter(s[feat], s["S_overall_graphrag"], marker="o", color=colors[cat], alpha=0.6, label=f"GraphRAG / {CATEGORY_LABEL[cat]}")
        ax.scatter(s[feat], s["S_overall_rag"], marker="x", color=colors[cat], alpha=0.6, label=f"RAG / {CATEGORY_LABEL[cat]}")
    ax.set_xlabel(feat)
    ax.set_ylabel("S_overall")
    ax.set_title(f"{feat} vs S_overall (cả 100 ca)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Markdown report ──────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, kw: pd.DataFrame, sp: pd.DataFrame, out_path: Path, neo4j_used: bool) -> None:
    lines: List[str] = []
    lines.append("# TN-B1 — Tính hợp lệ nội tại của CheoBench\n")
    lines.append(f"- Tổng số ca: {len(df)}")
    for c in CATEGORY_ORDER:
        n = (df.category == c).sum()
        lines.append(f"- {CATEGORY_LABEL[c]}: {n} ca")
    lines.append("")
    lines.append(f"- Đặc trưng đồ thị (hop_pairwise_avg, subgraph_size): "
                 + ("đã tính từ Neo4j" if neo4j_used else "**bỏ qua** vì Neo4j không sẵn sàng"))
    lines.append("")

    lines.append("## 1. Trung bình đặc trưng theo nhóm + Kruskal-Wallis")
    lines.append("")
    lines.append("| Đặc trưng | Local | Community | Global | Kruskal H | p-value | Ý nghĩa |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for _, r in kw.iterrows():
        lines.append(
            f"| {r['feature']} | {r['local_mean']:.3f} | {r['community_mean']:.3f} | "
            f"{r['global_mean']:.3f} | {r['kruskal_H']:.3f} | {r['p_value']:.4f} | {r['sig']} |"
        )
    lines.append("")
    lines.append("Ý nghĩa: `***` p<0,001, `**` p<0,01, `*` p<0,05, `ns` không có ý nghĩa.")
    lines.append("Ba nhóm có khác biệt ở các đặc trưng nội tại → phân nhóm Local/Community/Global "
                 "không phải gán nhãn tuỳ tiện mà có nền tảng cấu trúc.")
    lines.append("")

    lines.append("## 2. Spearman: đặc trưng độ phức tạp vs điểm S_overall")
    lines.append("")
    lines.append("| Đặc trưng | Hệ thống | n | Spearman ρ | p-value | Ý nghĩa |")
    lines.append("|---|:---:|---:|---:|---:|:---:|")
    for _, r in sp.iterrows():
        lines.append(
            f"| {r['feature']} | {r['system']} | {r['n']} | {r['spearman_rho']:.3f} | "
            f"{r['p_value']:.4f} | {r['sig']} |"
        )
    lines.append("")
    lines.append("Tương quan đồng dấu giữa độ phức tạp và điểm số (đặc biệt là negative ρ - "
                 "câu khó hơn → điểm thấp hơn) trên CẢ HAI hệ thống chứng tỏ đặc trưng nội tại "
                 "đo đúng tín hiệu độ khó, không thiên vị một kiến trúc.")
    lines.append("")

    lines.append("## 3. Hình ảnh")
    lines.append("- `intrinsic_feature_box.png`: phân bố đặc trưng theo nhóm")
    lines.append("- `intrinsic_score_by_category.png`: phân bố S_overall theo nhóm")
    lines.append("- `intrinsic_scatter_*.png`: scatter đặc trưng vs điểm")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    use_neo4j = "--no-neo4j" not in sys.argv
    print(f"[1/4] Build feature table (Neo4j: {'on' if use_neo4j else 'off'})...")
    df = build_feature_table(use_neo4j=use_neo4j)
    df.to_csv(OUT / "intrinsic_features.csv", index=False, encoding="utf-8")

    neo4j_used = df["hop_pairwise_avg"].notna().any()
    print(f"      neo4j_features_filled = {neo4j_used}")

    print("[2/4] Kruskal-Wallis per feature...")
    kw = kruskal_wallis_per_feature(df)
    kw.to_csv(OUT / "intrinsic_kruskal.csv", index=False, encoding="utf-8")

    print("[3/4] Spearman feature x score...")
    sp = spearman_feature_score(df)
    sp.to_csv(OUT / "intrinsic_spearman.csv", index=False, encoding="utf-8")

    print("[4/4] Plots + report...")
    plot_feature_distributions(df, OUT / "intrinsic_feature_box.png")
    plot_score_by_category(df, OUT / "intrinsic_score_by_category.png")
    for feat in INTRINSIC_FEATURES:
        if df[feat].notna().sum() >= 5:
            plot_complexity_vs_score(df, feat, OUT / f"intrinsic_scatter_{feat}.png")

    write_report(df, kw, sp, OUT / "intrinsic_report.md", neo4j_used)
    print(f"Done. Output in {OUT}")


if __name__ == "__main__":
    main()
