#!/usr/bin/env python3
"""Render the per-step experiment timeline for the autoresearch duel."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# (exp_id, short change label, val_bpb, kept) — in execution order
SIQ = [
    ("exp-0", "baseline\n9L·512d, GQA8/4\nMLP×2, vocab1024", 2.019, True),
    ("exp-1", "depth↔width\n12L / 448d", 2.259, False),
    ("exp-2", "depth\n12L / 512d", 2.333, False),
    ("exp-3", "warmdown\n1200→800", 1.866, True),
    ("exp-4", "arch tweak\n(layers/dim)", 1.915, False),
    ("exp-5", "MLP mult\nchange", 1.950, False),
    ("exp-0b", "re-verify\nchampion", 1.869, False),
    ("exp-6", "warmdown\n800→1200", 2.026, False),
    ("exp-7", "matrix_lr\n0.04→0.05", 1.806, True),
    ("exp-8", "mlp/layers\ntweak", 1.858, False),
    ("exp-9", "↑capacity\n(layers/heads)", 1.767, True),
    ("exp-10", "seq_len ↑\n(12 MB)", 1.935, False),
]
OPUS = [
    ("exp-0", "baseline", 1.969, True),
    ("exp-1", "matrix_lr\ntune", 1.899, True),
    ("exp-2", "matrix_lr\ntune", 1.840, True),
    ("exp-3", "lr / capacity", 1.760, True),
]
GLM = [
    ("exp-0", "baseline", 2.078, True),
    ("exp-1", "depth\n9→12", 2.409, False),
    ("exp-2", "depth\n9→10", 2.407, False),
]

LANES = [
    ("Opus 4.8",                       OPUS, "1.760*", "4 эксп · 4 kept · ещё идёт"),
    ("SIQ-1-35B  (Qwen3.6-A3B, тюн)", SIQ, "1.767", "12 эксп · 4 kept · полные 2ч"),
    ("GLM-5.2",                        GLM, "2.078", "3 эксп · 0 улучш · стоп @65м"),
]

KEEP = "#2e7d32"; KEEP_FILL = "#c8e6c9"
REJ = "#b71c1c"; REJ_FILL = "#ffcdd2"
CHAMP_EDGE = "#1b5e20"

fig, ax = plt.subplots(figsize=(17, 8.6))
ax.set_xlim(0, 13.2); ax.set_ylim(0, 9.6)
ax.axis("off")

bw, bh = 0.98, 1.55          # box width/height
x0 = 1.9                       # first box x
row_y = {0: 7.2, 1: 4.5, 2: 1.9}
row_gapnote = 0.0

fig.suptitle("AUTORESEARCH-дуэль (karpathy/autoresearch) на openai/parameter-golf — что каждая модель меняла на каждом шаге",
             fontsize=14.5, fontweight="bold", y=0.978)
ax.text(6.6, 9.12, "autoresearch-петля: edit train_gpt.py → train (300с) → eval val_bpb → keep / revert.  Каждая модель ведёт её сама как Pi-Agent.",
        ha="center", fontsize=10, color="#333")
ax.text(6.6, 8.74, "Метрика: val_bpb (bits-per-byte на FineWeb), НИЖЕ = лучше · 1×A6000 на модель · одинаковая обвязка, разный «мозг»",
        ha="center", fontsize=9.2, color="#666")
# autoresearch badge
ax.add_patch(FancyBboxPatch((0.15, 8.95), 1.55, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                            edgecolor="#1565c0", facecolor="#e3f2fd", lw=1.6, zorder=4))
ax.text(0.92, 9.2, "AUTORESEARCH", ha="center", va="center", fontsize=9.5,
        fontweight="bold", color="#1565c0", zorder=5)

# track champion (best-so-far) staircase per lane
for ri, (name, data, final, sub) in enumerate(LANES):
    y = row_y[ri]
    # lane label
    ax.text(1.65, y + bh/2, name, ha="right", va="center", fontsize=12.5, fontweight="bold")
    ax.text(1.65, y + bh/2 - 0.42, sub, ha="right", va="center", fontsize=8.5, color="#555")
    ax.text(1.65, y + bh/2 + 0.42, f"итог {final}", ha="right", va="center",
            fontsize=11, color=KEEP_EDGE if False else "#1b5e20", fontweight="bold")

    best = None
    for i, (eid, label, bpb, kept) in enumerate(data):
        x = x0 + i * (bw + 0.12)
        fill = KEEP_FILL if kept else REJ_FILL
        edge = CHAMP_EDGE if kept else REJ
        lw = 2.4 if kept else 1.1
        box = FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.015,rounding_size=0.08",
                             linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
        ax.add_patch(box)
        # champion star
        is_new_champ = kept and (best is None or bpb < best)
        if is_new_champ:
            best = bpb
            ax.text(x + bw - 0.12, y + bh - 0.16, "★", ha="right", va="top",
                    fontsize=13, color="#f9a825", zorder=5)
        ax.text(x + bw/2, y + bh - 0.18, eid, ha="center", va="top",
                fontsize=8.2, fontweight="bold", color="#222")
        ax.text(x + bw/2, y + bh/2 + 0.04, label, ha="center", va="center",
                fontsize=7.0, color="#333")
        ax.text(x + bw/2, y + 0.14, f"{bpb:.3f}", ha="center", va="bottom",
                fontsize=9.2, fontweight="bold",
                color=KEEP if kept else REJ)
        # arrow to next
        if i < len(data) - 1:
            ax.annotate("", xy=(x + bw + 0.12, y + bh/2), xytext=(x + bw, y + bh/2),
                        arrowprops=dict(arrowstyle="->", color="#999", lw=1.1), zorder=2)

# legend
ax.add_patch(FancyBboxPatch((1.9, 0.15), 0.5, 0.42, boxstyle="round,pad=0.01,rounding_size=0.06",
                            edgecolor=CHAMP_EDGE, facecolor=KEEP_FILL, lw=2.0))
ax.text(2.5, 0.36, "kept (новый чемпион ★)", va="center", fontsize=9)
ax.add_patch(FancyBboxPatch((6.0, 0.15), 0.5, 0.42, boxstyle="round,pad=0.01,rounding_size=0.06",
                            edgecolor=REJ, facecolor=REJ_FILL, lw=1.1))
ax.text(6.6, 0.36, "отклонён (хуже чемпиона или >16MB)", va="center", fontsize=9)
ax.text(12.9, 0.36, "* Opus ещё идёт", va="center", ha="right", fontsize=9, color="#555", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "autoresearch-duel/duel_steps.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", out)
