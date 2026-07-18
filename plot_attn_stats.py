"""
Visualize the attention-distance histograms saved by train_gpt.py
(logs/*_attn_stats_step*.pt) as a self-explanatory, plain-language figure.

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
from matplotlib.patches import Rectangle

# palette (validated): ordinal blue ramp early->late training, categorical slots for series
STEP_RAMP3 = ["#86b6ef", "#2a78d6", "#0d366b"]  # light -> dark = early -> late
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE, GREEN = "#2a78d6", "#008300"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CALLOUT = dict(arrowstyle="-", color=MUTED, lw=0.9)


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
    """Smallest distance that contains `frac` of all attention."""
    return int(np.searchsorted(p_cum, frac))


def log_bin_edges(T, per_octave=4):
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
            x.append(np.sqrt(max(lo, 1) * max(hi - 1, 1)))
            y.append(m[lo:hi].sum() / (LH * cs))
    return np.array(x), np.array(y)


def cum_dist(mass_lht):
    p = mass_lht.sum(axis=(0, 1))
    return np.cumsum(p / p.sum())


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def mark_windows(ax, T, label=True):
    for w, name in [(896, "short training window: 896"), (1792, "long training window: 1,792")]:
        ax.axvline(w, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)))
        if label:
            ax.text(w, 0.02, name + " ", color=MUTED, fontsize=7.5, rotation=90,
                    ha="right", va="bottom", transform=ax.get_xaxis_transform())
    ax.axvspan(1792, T, color=GRID, alpha=0.35, zorder=0, lw=0)


def title_caption(ax, title, cap):
    ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=24)
    ax.text(0, 1.035, cap, transform=ax.transAxes, color=MUTED, fontsize=8.5)


def plot_attn_stats(logs_dir="logs"):
    run, snaps = load_newest_run(logs_dir)
    steps = sorted(snaps)
    first, last = steps[0], steps[-1]
    mid = steps[len(steps) // 2] if len(steps) > 2 else None
    shown = [s for s in (first, mid, last) if s is not None]
    names = {first: f"before training (step {first})", last: f"fully trained (step {last})"}
    if mid is not None:
        names[mid] = f"mid-training (step {mid})"
    colors = dict(zip(shown, STEP_RAMP3 if len(shown) == 3 else [STEP_RAMP3[0], STEP_RAMP3[2]]))
    final = snaps[last]
    L, H, T = final["mass"].shape
    layer_ids = final["layers"]
    edges = log_bin_edges(T)
    cf = cum_dist(final["mass"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    (axA, axB), (axC, axD) = axes

    # ---- 1 · how much attention one token gets, by how far back it sits ------------
    style_axes(axA)
    dens = {}
    for s in shown:
        x, y = binned_density(snaps[s]["mass"], snaps[s]["count"], edges)
        dens[s] = (x, y)
        axA.plot(x, y, color=colors[s], linewidth=2.0, label=names[s])
    axA.set_xscale("log"); axA.set_yscale("log")
    mark_windows(axA, T)
    xf, yf = dens[last]
    x0, y0 = dens[first]
    axA.annotate(f"after training, the immediately preceding token\ngets {yf[0] / y0[0]:.0f}× more attention than at random start",
                 (xf[0], yf[0]), xytext=(3.5, yf[0] * 0.7), color=INK2, fontsize=9, va="top", arrowprops=CALLOUT)
    i_tail = min(np.searchsorted(xf, 3500), len(yf) - 1)
    axA.annotate("in the shaded zone the trained model overlaps its\nuntrained self — nothing was learned this far back\n(training never let the model look past 1,792 tokens)",
                 (xf[i_tail], yf[i_tail]), xytext=(0.45, 0.62), textcoords="axes fraction",
                 color=INK2, fontsize=9, arrowprops=CALLOUT)
    title_caption(axA, "1 · Training teaches the model to focus on tokens close by",
                  "Average share of attention a single earlier token receives, by how far back it sits.")
    axA.set_xlabel("how far back the earlier token sits (tokens, log scale)", color=INK2, fontsize=9.5)
    axA.set_ylabel("avg. share of attention one token gets (log scale)", color=INK2, fontsize=9.5)
    axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")

    # ---- 2 · what a sliding window would keep --------------------------------------
    style_axes(axB)
    for s in (first, last):
        c = cum_dist(snaps[s]["mass"])
        axB.plot(np.arange(1, T + 1), c, color=colors[s], linewidth=2.0, label=names[s])
    axB.set_xscale("log"); axB.set_ylim(0, 1.05)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    mark_windows(axB, T)
    offsets = {128: (7, -6, "left", "top"), 896: (-8, 8, "right", "bottom"), 1792: (8, -12, "left", "top")}
    for w in (128, 896, 1792):
        dx, dy, ha, va = offsets[w]
        axB.plot([w], [cf[w]], "o", color=colors[last], markersize=7)
        axB.annotate(f"{cf[w] * 100:.0f}% within {w:,}", (w, cf[w]), xytext=(dx, dy),
                     textcoords="offset points", color=INK2, fontsize=8.5, ha=ha, va=va)
    r50 = radius(cf, 0.5)
    axB.annotate(f"half of all attention goes to\njust the last {r50} tokens", (max(r50, 1), 0.5),
                 xytext=(0.03, 0.68), textcoords="axes fraction", color=INK2, fontsize=9, arrowprops=CALLOUT)
    title_caption(axB, f"2 · The last {r50} tokens get half the attention — the rest trails far back",
                  "If the model could only see its last w tokens, how much of its attention would survive?")
    axB.set_xlabel("window size: how many recent tokens the model may see (log scale)", color=INK2, fontsize=9.5)
    axB.set_ylabel("share of attention that window keeps", color=INK2, fontsize=9.5)
    axB.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

    # ---- 3 · layers specialize ------------------------------------------------------
    style_axes(axC)
    r50s, r90s = {}, {}
    for li, layer in enumerate(layer_ids):
        m = final["mass"][li].sum(axis=0)
        c = np.cumsum(m / m.sum())
        r50s[layer], r90s[layer] = max(radius(c, 0.5), 1), radius(c, 0.9)
        axC.plot([r50s[layer], r90s[layer]], [layer, layer], color=AXIS, linewidth=1.0, zorder=2)
        axC.plot([r50s[layer]], [layer], "o", color=GREEN, markersize=7, zorder=3)
        axC.plot([r90s[layer]], [layer], "o", color=BLUE, markersize=7, zorder=3)
    for layer in sorted(set(range(max(layer_ids) + 1)) - set(layer_ids)):
        axC.text(1.05, layer, f"layer {layer} has no attention", color=MUTED, fontsize=8.5, va="center")
    lo_l = min(r50s, key=r50s.get)
    hi_l = max(r50s, key=r50s.get)
    axC.annotate(f"layer {lo_l}: half its attention within {r50s[lo_l]} token(s)\n— it tracks the word right before",
                 (r50s[lo_l], lo_l), xytext=(0.30, 0.85), textcoords="axes fraction",
                 color=INK2, fontsize=9, arrowprops=CALLOUT)
    if hi_l != lo_l:
        axC.annotate(f"layer {hi_l}: the widest reader (half its\nattention within {r50s[hi_l]} tokens) — it gathers\ncontext for the final prediction",
                     (r50s[hi_l], hi_l), xytext=(0.30, 0.18), textcoords="axes fraction",
                     color=INK2, fontsize=9, arrowprops=CALLOUT)
    axC.set_xscale("log"); axC.set_xlim(0.9, T)
    axC.set_ylim(max(layer_ids) + 0.8, -0.8)
    axC.set_yticks(range(max(layer_ids) + 1))
    mark_windows(axC, T, label=False)
    axC.plot([], [], "o", color=GREEN, markersize=7, label="half of the layer's attention")
    axC.plot([], [], "o", color=BLUE, markersize=7, label="90% of it")
    title_caption(axC, "3 · Early layers watch their neighbors; the last layer reads broadly",
                  "Fully-trained model, per layer: distance containing half (green) and 90% (blue) of its attention.")
    axC.set_xlabel("distance (tokens, log scale)", color=INK2, fontsize=9.5)
    axC.set_ylabel("layer (input → output)", color=INK2, fontsize=9.5)
    axC.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="center right")

    # ---- 4 · heads specialize -------------------------------------------------------
    axD.set_facecolor(SURFACE)
    r90_grid = np.zeros((L, H))
    for li in range(L):
        for h in range(H):
            m = final["mass"][li, h]
            r90_grid[li, h] = radius(np.cumsum(m / m.sum()), 0.9)
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
            axD.text(h, li, f"{v:.0f}", ha="center", va="center", fontsize=7,
                     color="#ffffff" if v < thresh else INK)
    li_min, h_min = np.unravel_index(np.argmin(r90_grid), r90_grid.shape)
    axD.add_patch(Rectangle((h_min - 0.5, li_min - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.8))
    cb = fig.colorbar(im, ax=axD, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label("tokens needed to cover 90% of the head's attention\ndark = focused nearby · light = reads broadly", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    title_caption(axD, "4 · A few heads specialize in nearby text; most scan widely",
                  f"Each layer has {H} independent 'heads'. Outlined cell — layer {layer_ids[li_min]}, head {h_min} — "
                  f"keeps 90% of its attention within {r90_grid[li_min, h_min]:.0f} tokens.")
    axD.set_xlabel("attention head", color=INK2, fontsize=9.5)
    axD.set_ylabel("layer", color=INK2, fontsize=9.5)

    fig.suptitle("Where does a GPT look? — attention vs. distance, before and after training",
                 color=INK, fontsize=15, x=0.02, y=0.995, ha="left")
    fig.text(0.02, 0.968,
             f"When predicting each next word, the model distributes 100% of its “attention” over earlier tokens of the same document. "
             f"Measured on {T:,} tokens of held-out text ·  run {run.split('/')[-1]}",
             color=INK2, fontsize=9.5)
    fig.text(0.02, 0.006,
             "Reading guide: axes are logarithmic — each labeled gridline is 10× farther. Shaded zone = distances beyond the 1,792-token sliding window used in "
             "training; attention there was measured by letting the trained model read the whole document once, without a window.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95), h_pad=3.5, w_pad=2.5)

    # cross-check table against the printed attn_dist log lines
    print(f"{'step':>6} {'r50':>6} {'r90':>6} {'r95':>6} {'C128':>7} {'C896':>7} {'C1792':>7}")
    for s in steps:
        c = cum_dist(snaps[s]["mass"])
        print(f"{s:>6} {radius(c, 0.5):>6} {radius(c, 0.9):>6} {radius(c, 0.95):>6} "
              f"{c[128]:>7.4f} {c[896]:>7.4f} {c[min(1792, T - 1)]:>7.4f}")
    fig.savefig(f"{logs_dir}/attn_stats_summary.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {logs_dir}/attn_stats_summary.png")
    return fig


if __name__ == "__main__":
    plot_attn_stats()
    plt.show()
