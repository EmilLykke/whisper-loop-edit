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
# One framed plot, bars sorted from most to fewest repeated words, all starting on a
# visible zero line, with an x axis and a light grid; the description sits under it.
base_words = loop_words("baseline")
rows = [
    ("Random edit B", "C4", CONTROL, TEXT_2),
    ("Original large-v3", "baseline", ORIGINAL, TEXT),
    ("Random edit A", "C1", CONTROL, TEXT_2),
    ("Edited (copy heads off)", "E3", EDITED, TEXT),
]
fig, ax = plt.subplots(figsize=(11, 5.8))
fig.subplots_adjust(left=0.28, right=0.975, top=0.80, bottom=0.2)
ax.set_xlim(0, 3800)
ax.set_ylim(0.4, 4.6)
ax.set_facecolor("#202020")
for y, (label, m, color, ink) in zip([4, 3, 2, 1], rows):
    v = loop_words(m)
    ax.barh(y, v, height=0.6, color=color, zorder=3)
    ax.text(-60, y, label, ha="right", va="center", color=ink, fontsize=14.5,
            fontweight="bold" if ink == TEXT else "normal")
    t = ax.text(v + 50, y + 0.02, f"{v:,}", va="center", color=ink, fontsize=17,
                fontweight="bold", zorder=4)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    ax.text(bb.x1 + 30, y, f"words   ·   {loops(m)} repeats", va="center", color=MUTED,
            fontsize=13, zorder=4)
drop = 1 - loop_words("E3") / base_words
ax.set_yticks([])
ax.set_xticks(range(0, 3501, 500), [f"{x:,}" for x in range(0, 3501, 500)], fontsize=12)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
for side in ("top", "right", "bottom"):
    ax.spines[side].set_color(GRID)
ax.spines["left"].set_color(TEXT_2)
ax.spines["left"].set_linewidth(1.6)
ax.spines["left"].set_zorder(5)
ax.tick_params(axis="x", length=4, color=GRID, pad=6)
ax.set_xlabel("Extra words written by repeats", fontsize=12.5, color=TEXT_2, labelpad=10)
fig.text(0.022, 0.9, f"The edit cut repeated words by {drop:.0%}", fontsize=19, fontweight="bold",
         color=TEXT, va="center")
fig.text((0.28 + 0.975) / 2, 0.04, "30 Danish recordings (90 min) the edit never saw. Lower is better.",
         fontsize=12.5, color=MUTED, ha="center", va="center")
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
    "The edit helps where repeats happen, and changes nothing where they don't",
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
