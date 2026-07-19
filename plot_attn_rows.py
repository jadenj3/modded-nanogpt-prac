"""
Visualize the middle- and last-token attention rows saved by train_gpt.py
(logs/*_attn_rows_step*.pt).

At each validation step train_gpt.py recomputes, for the middle token and the
last token of the validation sequence, the exact attention distribution (sums
to 1) that query places over the tokens behind it — per attention layer,
averaged over heads and over every sequence in the validation loop. This script
renders one figure per validation step: a panel per layer showing both
distributions against distance-from-query, with that layer's sliding-window
width marked, so you can watch the distributions track the window as it grows
128 -> 1,792 tokens over training.

Run in a Jupyter notebook on the machine that holds the logs:

    %run plot_attn_rows.py                  # most recently written run in logs/, all val steps
    %run plot_attn_rows.py 923bef0b         # a specific run, by any unique substring of its name

or, for control:

    from plot_attn_rows import plot_attn_rows
    figs = plot_attn_rows("logs", run="923bef0b")   # {step: matplotlib Figure}
"""
import glob
import os
import re
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# palette (validated): categorical pair for the two query tokens + neutral chrome tokens
BLUE, GREEN = "#2a78d6", "#008300"  # middle token, last token
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
QUERIES = [("middle token", BLUE), ("last token", GREEN)]


def load_run(logs_dir="logs", run=None):
    """Load one run's snapshots. run=None picks the most recently written run; otherwise
    run is matched as a substring of the run prefix (e.g. "000_923bef0b" or a full uuid)."""
    files = glob.glob(f"{logs_dir}/*_attn_rows_step*.pt")
    if not files:
        raise FileNotFoundError(f"no *_attn_rows_step*.pt files under {logs_dir}/")
    runs = {}
    for f in files:
        m = re.match(r"(.*)_attn_rows_step(\d+)\.pt$", f)
        if m is not None:  # the glob is looser than the regex (e.g. renamed/backup copies)
            runs.setdefault(m.group(1), []).append((int(m.group(2)), f))
    if not runs:
        raise FileNotFoundError(f"no parseable *_attn_rows_step<digits>.pt files under {logs_dir}/")
    if run is None:
        chosen = max(runs, key=lambda r: max(os.path.getmtime(f) for _, f in runs[r]))
    else:
        matches = [r for r in runs if run in r]
        if not matches:
            raise FileNotFoundError(f"no run matching {run!r}; available: {sorted(runs)}")
        if len(matches) > 1:
            raise ValueError(f"run {run!r} is ambiguous; matches: {sorted(matches)}")
        chosen = matches[0]
    snaps = {step: torch.load(f, map_location="cpu") for step, f in sorted(runs[chosen])}
    return chosen, snaps


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def plot_step(run, step, snap, ylim):
    rows = snap["rows"].double().numpy()  # (L, H, 2, D) attention weight vs distance-from-query
    layers, long_layers = list(snap["layers"]), set(snap["long_layers"])
    wb, H, D = snap["window_blocks"], rows.shape[1], rows.shape[-1]
    per_layer = rows.mean(axis=1)  # (L, 2, D) heads averaged
    w_long, w_short = 128 * wb, 128 * max(wb // 2, 1)
    xmax = D * 1.5

    ncols = 4
    nrows = -(-(len(layers) + 1) // ncols)  # layer panels + one summary cell
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.1 * nrows + 0.4), dpi=110,
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for idx, layer in enumerate(layers):
        ax = axes.flat[idx]
        style_axes(ax)
        is_long = layer in long_layers
        w_tokens = w_long if is_long else w_short
        for qi, (name, color) in enumerate(QUERIES):
            y = per_layer[idx, qi]
            m = y > 0
            # a 1-block window leaves the block-aligned middle token attending only to itself:
            # too few points for a line, so mark them individually
            marker = "o" if m.sum() <= 3 else None
            ax.plot(np.arange(D)[m], y[m], color=color, linewidth=1.6, label=name,
                    marker=marker, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.axvline(w_tokens, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.axvspan(w_tokens, xmax, color=GRID, alpha=0.4, zorder=0, lw=0)
        ax.text(w_tokens, 0.97, f"window {w_tokens:,} ", color=MUTED, fontsize=6.8,
                rotation=90, ha="right", va="top", transform=ax.get_xaxis_transform())
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, xmax)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(f"layer {layer} · {'long' if is_long else 'short'} window",
                     color=INK, fontsize=9.5, loc="left", pad=4)
        if idx == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left", handlelength=1.4)
    # unused grid cells hold no panel; re-enable tick labels on the lowest used panel of each column
    for j in range(axes.shape[1]):
        rows_used = [i for i in range(axes.shape[0]) if i * axes.shape[1] + j < len(layers)]
        if not rows_used:
            continue
        low = axes[rows_used[-1], j]
        low.tick_params(labelbottom=True)
        low.set_xlabel("how far back the key token sits (0 = the query itself)", color=INK2, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("attention weight (log)", color=INK2, fontsize=8.5)

    # summary cell: legend + reading guide
    ax = axes.flat[len(layers)]
    ax.axis("off")
    for qi, (name, color) in enumerate(QUERIES):
        y = 0.96 - 0.07 * qi
        ax.plot([0.03, 0.10], [y, y], color=color, linewidth=2.0, transform=ax.transAxes, clip_on=False)
        ax.text(0.13, y, name, transform=ax.transAxes, color=INK2, fontsize=8.8, va="center")
    no_attn = sorted(set(range(max(layers) + 1)) - set(layers))
    ax.text(0.02, 0.78, "\n".join([
        f"step {step} · sliding window {w_long:,} tokens",
        f"(short-window layers use {w_short:,})",
        "",
        f"each curve: attention distribution of one query",
        f"token over the keys behind it — sums to 1,",
        f"averaged over {H} heads × {snap['num_batches']} validation",
        f"sequences of {snap['seq_len']:,} tokens.",
        "",
        "shaded region = beyond this layer's window,",
        "unreachable by construction. the window is",
        "block-wise (128 tokens): the middle token sits",
        "at a block boundary so its reach ends up to one",
        "block short of the last token's.",
        ] + ([f"", f"layer {no_attn[0]} has no attention"] if no_attn else [])),
        transform=ax.transAxes, color=INK2, fontsize=8.8, va="top")
    for extra in axes.flat[len(layers) + 1:]:
        extra.axis("off")

    fig.suptitle(f"Where the middle vs last token looks — validation step {step}, "
                 f"sliding window {w_long:,} tokens", color=INK, fontsize=14, x=0.02, ha="left")
    fig.text(0.02, 0.955, "Attention weight each query places on the token d positions back, per layer "
             f"(heads averaged) · run {run.split('/')[-1]}", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_attn_rows(logs_dir="logs", run=None):
    run, snaps = load_run(logs_dir, run)
    # one shared y-range across all steps so figures are comparable step-to-step
    lo, hi = np.inf, 0
    for snap in snaps.values():
        w = snap["rows"].double().numpy().mean(axis=1)
        lo, hi = min(lo, w[w > 0].min()), max(hi, w.max())
    ylim = (10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi)))

    figs = {}
    print(f"{'step':>6} {'window':>8} {'row sums (min..max, should be 1)':>34}")
    for step, snap in snaps.items():
        sums = snap["rows"].double().sum(dim=-1)  # (L, H, 2), each should be ~1
        print(f"{step:>6} {128 * snap['window_blocks']:>8,} {f'{sums.min():.4f} .. {sums.max():.4f}':>34}")
        figs[step] = plot_step(run, step, snap, ylim)
        out = f"{run}_attn_rows_step{step:06d}.png"
        figs[step].savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print(f"saved {out}")
    return figs


if __name__ == "__main__":
    plot_attn_rows(run=sys.argv[1] if len(sys.argv) > 1 else None)
    plt.show()
