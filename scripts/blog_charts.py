"""Charts for the Codictate blog post about this spike.

Reads results/final/results.json and writes PNGs (dark, to match the blog's other charts).

    python3 scripts/blog_charts.py <out_dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#1a1a1a"
TEXT = "#f0efec"
TEXT_2 = "#c3c2b7"
MUTED = "#8a8984"
GRID = "#3a3a38"
ORIGINAL = "#d95926"  # categorical slot 2 (dark)
EDITED = "#3987e5"  # categorical slot 1 (dark)
CONTROL = "#5f5e5a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "text.color": TEXT,
    }
)

r = json.load(open(ROOT / "results/final/results.json"))["results"]


def loop_words(m):
    return r[m]["long"]["timestamps"]["loopWords"] + r[m]["long"]["nt"]["loopWords"]


def loops(m):
    return r[m]["long"]["timestamps"]["loops"] + r[m]["long"]["nt"]["loops"]


def finish(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# 1. Words written by Loops, 30 sealed recordings
rows = [
    ("Original large-v3", "baseline", ORIGINAL),
    ("Random edit A", "C1", CONTROL),
    ("Random edit B", "C4", CONTROL),
    ("Copy heads off", "E3", EDITED),
]
fig, ax = plt.subplots(figsize=(11, 4.6))
ys = list(range(len(rows)))[::-1]
for y, (label, m, color) in zip(ys, rows):
    v = loop_words(m)
    ax.barh(y, v, height=0.6, color=color)
    ax.text(v + 30, y, f"{v:,} words  ·  {loops(m)} Loops", va="center", color=TEXT, fontsize=13)
ax.set_yticks(ys, [x[0] for x in rows], fontsize=14, color=TEXT)
ax.set_xlim(0, 3000)
ax.xaxis.grid(True, color=GRID, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("Extra words written by Loops (lower is better)")
finish(ax)
fig.suptitle(
    "Loops on 30 Danish recordings the edit never saw (90 min)",
    x=0.02, ha="left", fontsize=16, fontweight="bold",
)
ax.set_title(
    "whisper.cpp long-form, with and without timestamps",
    loc="left", fontsize=12, color=MUTED, pad=10,
)
fig.tight_layout()
fig.savefig(OUT / "loop-words.png", dpi=200)
plt.close(fig)

# 2. Word error rate, original vs edited
conds = [
    ("Long-form,\nwith timestamps", r["baseline"]["long"]["timestamps"]["wer"], r["E3"]["long"]["timestamps"]["wer"]),
    ("Long-form,\nno timestamps", r["baseline"]["long"]["nt"]["wer"], r["E3"]["long"]["nt"]["wer"]),
    ("Cut into slices\n(Codictate)", r["baseline"]["long"]["nt-vad"]["wer"], r["E3"]["long"]["nt-vad"]["wer"]),
    ("FLEURS Danish test,\n930 recordings", r["baseline"]["fleurs"]["wer"], r["E3"]["fleurs"]["wer"]),
]
fig, ax = plt.subplots(figsize=(11, 5.2))
w = 0.36
for i, (label, a, b) in enumerate(conds):
    for dx, v, color in ((-w / 2 - 0.01, a, ORIGINAL), (w / 2 + 0.01, b, EDITED)):
        ax.bar(i + dx, v * 100, width=w, color=color)
        digits = 2 if i == len(conds) - 1 else 1
        ax.text(i + dx, v * 100 + 0.6, f"{v * 100:.{digits}f}%", ha="center", color=TEXT_2, fontsize=12)
ax.set_xticks(range(len(conds)), [c[0] for c in conds], fontsize=12.5, color=TEXT)
ax.set_ylim(0, 45)
ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_ylabel("Word error rate (lower is better)")
finish(ax)
ax.bar(0, 0, color=ORIGINAL, label="Original large-v3")
ax.bar(0, 0, color=EDITED, label="Copy heads off")
ax.legend(frameon=False, loc="upper right", labelcolor=TEXT, fontsize=12.5)
fig.suptitle(
    "The edit helps where Loops happen, and changes nothing where they don't",
    x=0.02, ha="left", fontsize=16, fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT / "word-error-rate.png", dpi=200)
plt.close(fig)

# 3. How the copy Loop happens
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(0, 110)
ax.set_ylim(2, 56)
ax.axis("off")


def box(x, y, w, h, title, body, edge, fill="#242423"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                       linewidth=1.6, edgecolor=edge, facecolor=fill)
    )
    ax.text(x + 1.5, y + h - 2.6, title, fontsize=13.5, fontweight="bold", color=TEXT, va="top")
    ax.text(x + 1.5, y + h - 7.2, body, fontsize=11.5, color=TEXT_2, va="top", linespacing=1.45)


def arrow(x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 color=color, linewidth=1.6))


box(1, 27, 33, 26, "Previous text",
    "The last ~220 tokens Whisper\nwrote, fed back in as a hint\nfor the next 30 s window.\n\n\"…oplyser ICI. Der er ikke\nrapporteret nogen…\"", MUTED)
box(39, 27, 32, 26, "Finder heads",
    "Spot that the words just\nwritten appeared before, and\npoint at what came next.\n\nL12H13, L13H7, L13H18,\nL15H17, L2H19, L18H19", ORIGINAL)
box(76, 27, 33, 26, "Writer heads",
    "Copy that next word out\nagain, even when the audio\nsays something else.\n\nL25H18, L25H19, L24H0,\nL26H6, L21H14", ORIGINAL)
arrow(34.8, 40, 38.2, 40)
arrow(71.8, 40, 75.2, 40)
box(1, 4, 108, 15, "The edit: switch the 11 heads off",
    "For each head, zero its 64 columns of the self-attention output projection.\n"
    "One line of arithmetic, no training. The other 629 self-attention heads are untouched.\n"
    "With them off, the model stops copying and goes back to listening.", EDITED, fill="#1d2633")
arrow(55, 25.6, 55, 20.2, color=EDITED)
fig.suptitle("Where the copy Loop lives in Whisper large-v3's decoder",
             x=0.02, ha="left", fontsize=16, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "copy-heads.png", dpi=200)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
