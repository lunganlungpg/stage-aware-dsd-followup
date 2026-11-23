"""Draw the README figures from the stored designs and the run-count trajectory.

Usage (from the repository root):  python python/examples/make_figures.py
Needs matplotlib in addition to the package dependencies. Writes SVG files to docs/figures/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stageaware import FollowUpProblem, dsd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "svg.fonttype": "path",
})

CASES = [(6, 3, 4), (8, 4, 6), (10, 4, 4)]
LABELS = {
    (6, 3, 4): "(6, 3, 4)\ntissue-culture media",
    (8, 4, 6): "(8, 4, 6)\nrice N-P-K-Zn",
    (10, 4, 4): "(10, 4, 4)\nseed priming",
}


def load(tag, which):
    return np.loadtxt(DATA / "selected_designs" / f"{tag}_{which}.csv", delimiter=",", skiprows=1)


def title(fig, main, sub):
    fig.text(0.02, 0.97, main, fontsize=12.5, fontweight="bold", va="top", color=INK)
    fig.text(0.02, 0.905, sub, fontsize=9.5, va="top", color=INK2)


# ---------------------------------------------------------------- Figure A: Table 1 (Psi)
rows = []
for m, k, n2 in CASES:
    P, tag = FollowUpProblem(dsd(m)[:, :k]), f"m{m}_k{k}_n{n2}"
    rows.append((P.psi(load(tag, "psi_aware")), P.psi(load(tag, "Iu_naive"))))

fig, ax = plt.subplots(figsize=(7.2, 3.6))
fig.subplots_adjust(left=0.23, right=0.95, top=0.7, bottom=0.15)
y, h = np.arange(len(CASES))[::-1], 0.34
aware, naive = np.array(rows).T
ax.barh(y + h / 2 + 0.01, naive, h, color=ORANGE, label="Chosen by unblocked $I_u$ (stage ignored)")
ax.barh(y - h / 2 - 0.01, aware, h, color=BLUE, label="Chosen by stage-aware Ψ")
for yi, a, n in zip(y, aware, naive):
    ax.text(n + 0.01, yi + h / 2, f"{n:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.text(a + 0.01, yi - h / 2, f"{a:.3f}  (−{100 * (n - a) / n:.1f}%)", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y, [LABELS[c] for c in CASES], fontsize=9, color=INK2)
ax.set_xlabel("Integrated excess prediction variance Ψ  (lower is better)")
ax.set_xlim(0, max(naive) * 1.32)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.85), ncol=2, frameon=False, fontsize=8.5)
title(fig, "Stage-aware selection lowers prediction variance",
      "Ψ of the follow-up chosen by each criterion, evaluated under the stage-adjusted analysis (Table 1)")
fig.savefig(OUT / "psi_reduction.svg")
plt.close(fig)

# ---------------------------------------------------------------- Figure B: pooled-criterion bias
fig, ax = plt.subplots(figsize=(7.2, 3.6))
fig.subplots_adjust(left=0.23, right=0.95, top=0.7, bottom=0.15)
for yi, (m, k, n2) in zip(y, CASES):
    P, tag = FollowUpProblem(dsd(m)[:, :k]), f"m{m}_k{k}_n{n2}"
    pooled = load(tag, "Cu_pooled")
    claim, actual, best = P.pooled_contrast(pooled), P.contrast(pooled), P.contrast(load(tag, "C_aware"))
    ax.plot([claim, actual], [yi, yi], color=AXIS, lw=2, zorder=1)
    ax.scatter([claim], [yi], s=70, facecolor=SURFACE, edgecolor=ORANGE, lw=2, zorder=3)
    ax.scatter([actual], [yi], s=70, color=ORANGE, edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.scatter([best], [yi - 0.22], s=70, marker="D", color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.text(actual, yi + 0.17, f"{actual:.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.text(best, yi - 0.47, f"{best:.3f}", ha="center", fontsize=8.5, color=INK)
    if (m, k, n2) == (10, 4, 4):
        ax.text(actual, yi - 0.3, f"+{100 * (actual - best) / best:.0f}% above\nstage-aware", ha="center",
                va="top", fontsize=8.5, color=INK2)
ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=ORANGE, lw=2, label="Pooled design: $C_u$ it reports")
ax.scatter([], [], s=60, color=ORANGE, label="Pooled design: true stage-adjusted $C$")
ax.scatter([], [], s=60, marker="D", color=BLUE, label="Stage-aware design: $C$")
ax.set_yticks(y, [LABELS[c] for c in CASES], fontsize=9, color=INK2)
ax.set_ylim(-0.8, len(CASES) - 0.45)
ax.set_xlabel("Average variance of pairwise contrasts C  (lower is better)")
ax.set_xlim(left=0)
ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.85), ncol=3, frameon=False, fontsize=8.5, columnspacing=1.2)
title(fig, "A pooled criterion overstates the precision it delivers",
      "The gap between the hollow and filled orange dots is the directional-leverage bias, eq. (10) (Table 2)")
fig.savefig(OUT / "pooled_bias.svg")
plt.close(fig)

# ---------------------------------------------------------------- Figure C: run-count rule (Fig. 1)
traj = np.genfromtxt(DATA / "run_count_trajectory.csv", delimiter=",", names=True)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
fig.subplots_adjust(left=0.1, right=0.97, top=0.72, bottom=0.17, wspace=0.12)
for ax, (m, k), color in zip(axes, [(6, 3), (8, 4)], [BLUE, AQUA]):
    t = traj[(traj["m"] == m) & (traj["k"] == k)]
    P = FollowUpProblem(dsd(m)[:, :k])
    qmin = P.q + 1
    ok = np.isfinite(t["excess"]) & (t["n2"] > 0)
    ax.plot(t["n2"][ok], t["excess"][ok], color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    if P.q > 0:
        ax.axvspan(-0.5, qmin - 0.5, color=GRID, alpha=0.7, lw=0)
        ax.text((qmin - 1) / 2, 0.08, f"rank-deficient\nn₂ < q + 1 = {qmin}", ha="center", fontsize=8.5, color=INK2)
    base = t[t["n2"] == 0]["excess"][0]
    if np.isfinite(base):
        ax.axhline(base, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(6.3, base - 0.05, "screening runs only", ha="right", fontsize=8, color=MUTED)
    ax.set_title(f"m = {m}, k = {k}   (q = {P.q})", fontsize=10, color=INK, loc="left")
    ax.set_xlim(-0.5, 6.5)
    ax.set_xticks(range(0, 7))
    ax.set_xlabel("Follow-up runs n₂")
axes[0].set_ylabel("Best-found Ψ")
axes[0].set_ylim(0, 0.8)
title(fig, "How many follow-up runs? At least q + 1 (Theorem 1)",
      "Best Ψ found by 200-start coordinate exchange; below q + 1 runs the model is not estimable")
fig.savefig(OUT / "run_count.svg")
plt.close(fig)

print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
