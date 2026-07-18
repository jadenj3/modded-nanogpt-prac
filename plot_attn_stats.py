"""
Visualize the attention-distance histograms saved by train_gpt.py
(logs/*_attn_stats_step*.pt) — the data behind the `attn_dist ...` log lines,
at full token resolution rather than log2 bins.

Run in a Jupyter notebook on the machine that holds the logs:

    %run plot_attn_stats.py                 # newest run in logs/

or, for control:

    from plot_attn_stats import plot_attn_stats
    fig = plot_attn_stats("logs")           # returns the matplotlib Figure

Each .pt file holds:
    mass   (L, H, T)  dense (causal + document masked) attention mass per
                      (attention layer, head, query-key distance d = i - j),
                      summed over the analysis query tokens
    count  (T,)       number of eligible (query, key) pairs at each distance
    layers list[int]  layer indices that have attention (layer 7 has none)
"""
import glob
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# palette (validated): ordinal blue ramp for training steps, categorical slots for series
STEP_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # light -> dark = early -> late
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE, GREEN = "#2a78d6", "#008300"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load_newest_run(logs_dir="logs"):
    files = glob.glob(f"{logs_dir}/*_attn_stats_step*.pt")
    if not files:
        raise FileNotFoundError(f"no *_attn_stats_step*.pt files under {logs_dir}/")
    runs = {}
    for f in files:
        m = re.match(r"(.*)_attn_stats_step(\d+)\.pt$", f)
        runs.setdefault(m.group(1), []).append((int(m.group(2)), f))
    run = max(runs, key=lambda r: max(f for _, f in runs[r]))  # newest run by filename
    snaps = {}
    for step, f in sorted(runs[run]):
        d = torch.load(f, map_location="cpu")
        snaps[step] = dict(mass=d["mass"].double().numpy(), count=d["count"].double().numpy(),
                           layers=list(d["layers"]))
    return run, snaps


def radius(p_cum, frac):
    """Smallest w with C(w) >= frac, given the cumulative distance distribution."""
    return int(np.searchsorted(p_cum, frac))


def log_bin_edges(T, per_octave=4):
    """Integer bin edges, ~per_octave bins per distance doubling, starting at d=1."""
    edges = np.unique(np.round(2 ** np.arange(0, np.log2(T), 1 / per_octave)).astype(int))
    return np.append(edges, T)


def binned_density(mass_lht, count, edges):
    """Mean attention probability received by one eligible token at each distance bin."""
    m, c = mass_lht.sum(axis=(0, 1)), count
    LH = mass_lht.shape[0] * mass_lht.shape[1]
    x, y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cs = c[lo:hi].sum()
        if cs > 0:
            x.append(np.sqrt(lo * max(hi - 1, 1)))  # geometric bin center
            y.append(m[lo:hi].sum() / (LH * cs))
    return np.array(x), np.array(y)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def mark_windows(ax, T, label=True):
    for w, name in [(128, "128"), (896, "short window 896"), (1792, "long window 1792")]:
        ax.axvline(w, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)))
        if label:
            ax.text(w, 0.02, name + " ", color=MUTED, fontsize=7, rotation=90,
                    ha="right", va="bottom", transform=ax.get_xaxis_transform())
    ax.axvspan(1792, T, color=GRID, alpha=0.35, zorder=0, lw=0)


def plot_attn_stats(logs_dir="logs", max_series=5):
    run, snaps = load_newest_run(logs_dir)
    steps = sorted(snaps)
    shown = sorted({steps[i] for i in np.round(np.linspace(0, len(steps) - 1, min(max_series, len(steps)))).astype(int)})
    colors = {s: STEP_RAMP[i] for i, s in enumerate(shown)}
    final = snaps[steps[-1]]
    L, H, T = final["mass"].shape
    layer_ids = final["layers"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    (axA, axB), (axC, axD) = axes

    # ---- A: per-token density vs distance (the `density` log lines) ----------------
    style_axes(axA)
    edges = log_bin_edges(T)
    for s in shown:
        x, y = binned_density(snaps[s]["mass"], snaps[s]["count"], edges)
        lbl = f"step {s}" + (" (init ≈ uniform)" if s == 0 else "")
        axA.plot(x, y, color=colors[s], linewidth=1.8, label=lbl)
    axA.set_xscale("log"); axA.set_yscale("log")
    mark_windows(axA, T)
    # slope guide: density proportional to 1/d
    xg = np.array([8, 320]); yg = 2.19e-2 * (xg / 4) ** -1.0
    axA.plot(xg, yg, color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)))
    axA.text(xg[1] * 1.15, yg[1], "~1/d", color=MUTED, fontsize=8, va="center")
    axA.set_title("Average attention on one token at distance d", color=INK, fontsize=11, loc="left")
    axA.set_xlabel("query–key distance d (tokens, log)", color=INK2, fontsize=9)
    axA.set_ylabel("mean prob. per eligible token (log)", color=INK2, fontsize=9)
    axA.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    axA.text(0, 1.06, "log line: density [...]   ·   d = 0 (self) omitted from log axis",
             transform=axA.transAxes, color=MUTED, fontsize=8)

    # ---- B: retained mass C(w) (the `mass`, C*, r* log fields) ---------------------
    style_axes(axB)
    for s in shown:
        p = snaps[s]["mass"].sum(axis=(0, 1)); c = np.cumsum(p / p.sum())
        axB.plot(np.arange(1, T + 1), c, color=colors[s], linewidth=1.8, label=f"step {s}")
    axB.set_xscale("log"); axB.set_ylim(0, 1.02)
    for frac, name in [(0.5, "r50"), (0.9, "r90"), (0.95, "r95")]:
        axB.axhline(frac, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)))
        axB.text(T * 0.92, frac + 0.012, name, color=MUTED, fontsize=7.5, ha="right")
    mark_windows(axB, T)
    pf = final["mass"].sum(axis=(0, 1)); cf = np.cumsum(pf / pf.sum())
    r90 = radius(cf, 0.9)
    axB.plot([r90], [0.9], "o", color=colors[steps[-1]], markersize=7)
    axB.annotate(f"r90 = {r90}", (r90, 0.9), xytext=(r90 * 0.06, 0.93), color=INK2, fontsize=8.5)
    axB.set_title("Fraction of dense attention a window of size w retains", color=INK, fontsize=11, loc="left")
    axB.set_xlabel("window size w (tokens, log)", color=INK2, fontsize=9)
    axB.set_ylabel("C(w) = cumulative mass within w", color=INK2, fontsize=9)
    axB.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    axB.text(0, 1.06, "log fields: C128 / C896 / C1792, r50 / r90 / r95   ·   mass bins = increments of this curve",
             transform=axB.transAxes, color=MUTED, fontsize=8)

    # ---- C: per-layer r50 / r90 at the final step (the per-layer log lines) --------
    style_axes(axC)
    for li, layer in enumerate(layer_ids):
        m = final["mass"][li].sum(axis=0); c = np.cumsum(m / m.sum())
        x50, x90 = max(radius(c, 0.5), 1), radius(c, 0.9)
        axC.plot([x50, x90], [layer, layer], color=AXIS, linewidth=1.0, zorder=2)
        axC.plot([x50], [layer], "o", color=GREEN, markersize=6.5, zorder=3)
        axC.plot([x90], [layer], "o", color=BLUE, markersize=6.5, zorder=3)
    missing = sorted(set(range(max(layer_ids) + 1)) - set(layer_ids))
    for layer in missing:
        axC.text(1.05, layer, f"layer {layer}: no attention", color=MUTED, fontsize=8, va="center")
    axC.set_xscale("log"); axC.set_xlim(0.9, T)
    axC.set_ylim(max(layer_ids) + 0.8, -0.8)
    axC.set_yticks(range(max(layer_ids) + 1))
    mark_windows(axC, T, label=False)
    axC.plot([], [], "o", color=GREEN, markersize=6.5, label="r50 (half the mass)")
    axC.plot([], [], "o", color=BLUE, markersize=6.5, label="r90 (90% of the mass)")
    axC.set_title(f"Effective attention radius by layer · step {steps[-1]}", color=INK, fontsize=11, loc="left")
    axC.set_xlabel("distance (tokens, log)", color=INK2, fontsize=9)
    axC.set_ylabel("layer", color=INK2, fontsize=9)
    axC.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    axC.text(0, 1.06, "log lines: layer:N r50 / r90   ·   dashed verticals: 128 / 896 / 1792-token windows",
             transform=axC.transAxes, color=MUTED, fontsize=8)

    # ---- D: per-head r90 heatmap at the final step (the `head_r90` log lines) ------
    axD.set_facecolor(SURFACE)
    r90_grid = np.zeros((L, H))
    for li in range(L):
        for h in range(H):
            m = final["mass"][li, h]; c = np.cumsum(m / m.sum())
            r90_grid[li, h] = radius(c, 0.9)
    cmap = LinearSegmentedColormap.from_list("blue_seq", SEQ_RAMP)
    im = axD.imshow(r90_grid, cmap=cmap.reversed(), aspect="auto", vmin=0, vmax=r90_grid.max())
    axD.set_xticks(range(H), [f"h{h}" for h in range(H)])
    axD.set_yticks(range(L), [str(l) for l in layer_ids])
    axD.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in axD.spines.values():
        side.set_visible(False)
    thresh = 0.55 * r90_grid.max()
    for li in range(L):
        for h in range(H):
            v = r90_grid[li, h]
            axD.text(h, li, f"{v:.0f}", ha="center", va="center", fontsize=6.8,
                     color="#ffffff" if v < thresh else INK)
    cb = fig.colorbar(im, ax=axD, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label("r90 (tokens) — darker = more local head", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    axD.set_title(f"Head specialization: r90 per (layer, head) · step {steps[-1]}", color=INK, fontsize=11, loc="left")
    axD.set_xlabel("head", color=INK2, fontsize=9)
    axD.set_ylabel("layer", color=INK2, fontsize=9)
    axD.text(0, 1.06, "log lines: layer:N head_r90 h0:... h7:...   ·   ≈2300 = broad/uniform, small = local specialist",
             transform=axD.transAxes, color=MUTED, fontsize=8)

    fig.suptitle("Where does attention go? — dense-attention mass vs. query–key distance",
                 color=INK, fontsize=14, x=0.02, y=0.99, ha="left")
    fig.text(0.02, 0.958, f"run {run.split('/')[-1]}   ·   {T:,}-token analysis context, {L} attention layers × {H} heads, "
                          f"averaged over queries   ·   shaded region: beyond the 1792-token training window",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    # cross-check table against the printed attn_dist log lines
    print(f"{'step':>6} {'r50':>6} {'r90':>6} {'r95':>6} {'C128':>7} {'C896':>7} {'C1792':>7}")
    for s in shown:
        p = snaps[s]["mass"].sum(axis=(0, 1)); c = np.cumsum(p / p.sum())
        print(f"{s:>6} {radius(c, 0.5):>6} {radius(c, 0.9):>6} {radius(c, 0.95):>6} "
              f"{c[128]:>7.4f} {c[896]:>7.4f} {c[min(1792, T - 1)]:>7.4f}")
    fig.savefig(f"{logs_dir}/attn_stats_summary.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {logs_dir}/attn_stats_summary.png")
    return fig


if __name__ == "__main__":
    plot_attn_stats()
    plt.show()
