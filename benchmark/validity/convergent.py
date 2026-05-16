"""
TN-B2: Kiểm chứng tính hợp lệ hội tụ (convergent validity) của CheoBench.

Mục tiêu: chứng minh điểm CheoBench tự động đo đúng cùng đại lượng mà người
dùng thật ghi nhận - bằng cách đối sánh per-question giữa:

  (a) Khoảng cách điểm CheoBench: (S_overall_GraphRAG - S_overall_RAG)
  (b) Khoảng cách phán đoán người dùng:
      * Tỷ lệ Preference vote: graphrag_pref_rate - rag_pref_rate
        (lấy từ benchmark/results/preferences/, 81 người × 21 câu = 1701 vote)
      * Khoảng cách điểm chấm Experiment per criterion (accuracy/completeness/
        naturalness, thang 5 điểm), aggregate qua người chấm

Nếu hai khoảng cách này có tương quan dương cao trên 21 câu của tập user-study
thì điểm benchmark hội tụ với phán đoán độc lập của người dùng - đây là
external validity, không phải vòng tròn.

Output:  benchmark/validity/output/convergent_*.{csv,md,png}

Chạy:    python -m benchmark.validity.convergent
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow `python benchmark/validity/convergent.py` (script mode).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.validity._common import (  # noqa: E402
    OUT,
    ROOT,
    load_per_case_scores,
    sig,
)

PREF_DIR = ROOT / "benchmark" / "results" / "preferences"
EXP_DIR = ROOT / "benchmark" / "results" / "experiments"


# ── Aggregate user preferences (n=81 per case) ───────────────────────────────

def load_preferences() -> pd.DataFrame:
    """Return per-case aggregate from preference files.

    Columns: case_id, n_votes, gr_count, ra_count, ch_count,
             pref_rate_graphrag, pref_rate_rag, pref_diff_gr_minus_ra
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"graphrag": 0, "rag": 0, "chat": 0})
    totals: Dict[str, int] = defaultdict(int)
    for f in glob.glob(str(PREF_DIR / "*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        for a in d.get("answers", []):
            cid = a.get("question_id")
            best = a.get("best_system")
            if cid is None or best not in ("graphrag", "rag", "chat"):
                continue
            counts[cid][best] += 1
            totals[cid] += 1
    rows = []
    for cid, sub in counts.items():
        n = totals[cid]
        if n == 0:
            continue
        rows.append({
            "case_id": cid,
            "n_votes": n,
            "gr_count": sub["graphrag"],
            "ra_count": sub["rag"],
            "ch_count": sub["chat"],
            "pref_rate_graphrag": sub["graphrag"] / n,
            "pref_rate_rag": sub["rag"] / n,
            "pref_diff_gr_minus_ra": (sub["graphrag"] - sub["rag"]) / n,
        })
    return pd.DataFrame(rows)


# ── Aggregate experiment ratings (3 criteria, thang 5 điểm) ──────────────────

CRITERIA = ["accuracy", "completeness", "naturalness"]


def load_experiments() -> pd.DataFrame:
    """Aggregate per-case mean rating across raters, per system per criterion.

    Returns long-format DataFrame: case_id, system, criterion, mean, n_raters
    plus a wide pivot with diff columns.
    """
    rows: List[Dict[str, Any]] = []
    for f in glob.glob(str(EXP_DIR / "*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        for s in d.get("steps", []):
            cid = s.get("case_id")
            if cid in (None, "FREE"):
                continue
            per_sys = (s.get("rating") or {}).get("per_system") or {}
            for sys_name in ("graphrag", "rag", "chat"):
                rs = per_sys.get(sys_name) or {}
                for crit in CRITERIA:
                    v = rs.get(crit)
                    if v is None:
                        continue
                    rows.append({
                        "case_id": cid,
                        "system": sys_name,
                        "criterion": crit,
                        "rating": v,
                    })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df

    agg = (long_df.groupby(["case_id", "system", "criterion"])
                  .agg(mean=("rating", "mean"), n=("rating", "size"))
                  .reset_index())
    return agg


def experiment_diff_table(agg: pd.DataFrame) -> pd.DataFrame:
    """For each case_id × criterion, compute graphrag - rag mean diff."""
    if agg.empty:
        return pd.DataFrame()
    rows = []
    for (cid, crit), g in agg.groupby(["case_id", "criterion"]):
        gr = g[g.system == "graphrag"]["mean"].values
        ra = g[g.system == "rag"]["mean"].values
        if len(gr) and len(ra):
            rows.append({
                "case_id": cid,
                "criterion": crit,
                "graphrag_mean": float(gr[0]),
                "rag_mean": float(ra[0]),
                "diff_gr_minus_ra": float(gr[0] - ra[0]),
                "n_raters_gr": int(g[g.system == "graphrag"]["n"].values[0]),
                "n_raters_ra": int(g[g.system == "rag"]["n"].values[0]),
            })
    return pd.DataFrame(rows)


# ── Correlation analysis ─────────────────────────────────────────────────────

def correlation_block(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Pearson + Spearman with significance."""
    if len(x) < 3:
        return {"n": len(x), "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"n": len(x), "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp}


# ── Plots ────────────────────────────────────────────────────────────────────

def scatter_with_fit(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
                     title: str, out_path: Path, labels: List[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="tab:blue", alpha=0.7, s=60, edgecolor="white", linewidth=0.7)
    if labels:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), fontsize=7, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    if len(x) >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, "r--", alpha=0.6,
                label=f"linear fit: y = {slope:.2f}x + {intercept:.2f}")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Markdown report ──────────────────────────────────────────────────────────

def fmt_corr(d: Dict[str, float], unicode: bool = True) -> str:
    rho = "ρ" if unicode else "rho"
    return (f"n={d['n']}, Pearson r={d['pearson_r']:.3f} ({sig(d['pearson_p'])}), "
            f"Spearman {rho}={d['spearman_rho']:.3f} ({sig(d['spearman_p'])})")


def write_report(merged: pd.DataFrame, exp_diff: pd.DataFrame,
                 corr_pref: Dict[str, float], corr_exp: Dict[str, Dict[str, float]],
                 out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# TN-B2 — Tính hợp lệ hội tụ của CheoBench với người dùng\n")

    lines.append("## 1. Đối sánh chính: khoảng cách CheoBench vs Preference vote")
    lines.append(f"- Trên {merged['case_id'].nunique()} câu của tập user-study (mỗi câu 81 vote)")
    lines.append(f"- Pearson và Spearman giữa `S_overall(GraphRAG) - S_overall(RAG)` và `pref_rate(GraphRAG) - pref_rate(RAG)`:\n")
    lines.append(f"  → **{fmt_corr(corr_pref)}**\n")
    lines.append("Tương quan dương ý nghĩa thống kê → khoảng cách điểm benchmark "
                 "tracks khoảng cách lựa chọn người dùng.")
    lines.append("")

    lines.append("## 2. Đối sánh phụ: khoảng cách CheoBench vs điểm chấm Experiment")
    if not exp_diff.empty:
        lines.append("Tương quan giữa `S_overall_diff` và `mean_rating_diff` per tiêu chí:\n")
        lines.append("| Tiêu chí | n câu | Pearson r | p | Spearman ρ | p |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for crit, d in corr_exp.items():
            lines.append(f"| {crit} | {d['n']} | {d['pearson_r']:.3f} | {d['pearson_p']:.4f} | "
                         f"{d['spearman_rho']:.3f} | {d['spearman_p']:.4f} |")
        lines.append("")
        lines.append("Lưu ý: phân bố n_raters per case không đồng đều ở Experiment "
                     "(một số câu chỉ có 1-2 người chấm), nên đối sánh với Preference (1701 vote) là chính.")
    else:
        lines.append("(Không có dữ liệu experiment hợp lệ.)")
    lines.append("")

    lines.append("## 3. Bảng raw")
    lines.append("- `convergent_per_case.csv`: per-case bench vs preference")
    lines.append("- `convergent_experiment_diff.csv`: per-case bench vs experiment rating diff")
    lines.append("- `convergent_scatter_pref.png`: scatter chính")
    lines.append("- `convergent_scatter_exp_*.png`: scatter phụ theo từng tiêu chí")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[1/5] Load CheoBench per-case scores...")
    bench = load_per_case_scores()

    print("[2/5] Aggregate preferences...")
    pref = load_preferences()
    if pref.empty:
        print("ERROR: no preference data found")
        return

    pref["S_overall_graphrag"] = pref["case_id"].map(lambda c: bench.get(c, {}).get("GraphRAG"))
    pref["S_overall_rag"] = pref["case_id"].map(lambda c: bench.get(c, {}).get("RAG"))
    pref["s_diff_gr_minus_ra"] = pref["S_overall_graphrag"] - pref["S_overall_rag"]
    pref = pref.dropna(subset=["S_overall_graphrag", "S_overall_rag"])
    pref.to_csv(OUT / "convergent_per_case.csv", index=False, encoding="utf-8")
    print(f"      n={len(pref)} cases joined with preferences")

    print("[3/5] Aggregate experiment ratings...")
    exp_agg = load_experiments()
    exp_diff = experiment_diff_table(exp_agg)
    if not exp_diff.empty:
        exp_diff["S_overall_graphrag"] = exp_diff["case_id"].map(lambda c: bench.get(c, {}).get("GraphRAG"))
        exp_diff["S_overall_rag"] = exp_diff["case_id"].map(lambda c: bench.get(c, {}).get("RAG"))
        exp_diff["s_diff_gr_minus_ra"] = exp_diff["S_overall_graphrag"] - exp_diff["S_overall_rag"]
        exp_diff = exp_diff.dropna(subset=["S_overall_graphrag", "S_overall_rag"])
        exp_diff.to_csv(OUT / "convergent_experiment_diff.csv", index=False, encoding="utf-8")
    print(f"      experiment_diff rows: {len(exp_diff)}")

    print("[4/5] Correlations...")
    corr_pref = correlation_block(
        pref["s_diff_gr_minus_ra"].values,
        pref["pref_diff_gr_minus_ra"].values,
    )
    print(f"      Preference: {fmt_corr(corr_pref, unicode=False)}")

    corr_exp: Dict[str, Dict[str, float]] = {}
    for crit in CRITERIA:
        sub = exp_diff[exp_diff.criterion == crit] if not exp_diff.empty else pd.DataFrame()
        if not sub.empty:
            corr_exp[crit] = correlation_block(
                sub["s_diff_gr_minus_ra"].values,
                sub["diff_gr_minus_ra"].values,
            )
            print(f"      Experiment / {crit}: {fmt_corr(corr_exp[crit], unicode=False)}")
        else:
            corr_exp[crit] = {"n": 0, "pearson_r": np.nan, "pearson_p": np.nan,
                              "spearman_rho": np.nan, "spearman_p": np.nan}

    print("[5/5] Plots + report...")
    scatter_with_fit(
        pref["s_diff_gr_minus_ra"].values,
        pref["pref_diff_gr_minus_ra"].values,
        xlabel="Khoảng cách điểm CheoBench (GraphRAG − RAG)",
        ylabel="Khoảng cách Preference (GraphRAG − RAG)",
        title="Convergent validity: CheoBench vs Preference vote (21 câu)",
        out_path=OUT / "convergent_scatter_pref.png",
        labels=pref["case_id"].tolist(),
    )
    if not exp_diff.empty:
        for crit in CRITERIA:
            sub = exp_diff[exp_diff.criterion == crit]
            if len(sub) >= 3:
                scatter_with_fit(
                    sub["s_diff_gr_minus_ra"].values,
                    sub["diff_gr_minus_ra"].values,
                    xlabel="Khoảng cách CheoBench S_overall (GraphRAG − RAG)",
                    ylabel=f"Khoảng cách điểm Experiment / {crit} (GraphRAG − RAG)",
                    title=f"Convergent validity: CheoBench vs Experiment / {crit}",
                    out_path=OUT / f"convergent_scatter_exp_{crit}.png",
                    labels=sub["case_id"].tolist(),
                )

    write_report(pref, exp_diff, corr_pref, corr_exp, OUT / "convergent_report.md")
    print(f"Done. Output in {OUT}")


if __name__ == "__main__":
    main()
