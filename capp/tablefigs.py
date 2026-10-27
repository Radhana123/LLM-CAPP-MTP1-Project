"""Figures of Table 4 (Step 22): the ablation read at a glance. Needs matplotlib (Agg: no window). Called by capp/tables.build().

Two figures, both from the same cells as table4.csv (which is the table view of them, and the thing to quote):
  fig_ablation.png   one row per configuration A0, A1 ...; three panels (M1 first-pass valid, M2 LLM-final share, M6 efficiency gap),
                     one dot per LLM with a whisker of +-1 SD over the repeats
  fig_cost.png       one panel per LLM: its ablation path A0 -> A1 -> ... in the plane API calls per part (x) against M6 gap (y), what each
                     added component buys and costs (lower left is better; rows with identical numbers share a label)

Colours follow the dataviz rules: the validated categorical palette in its fixed order (never cycled, so at most 3 LLMs: a fourth is
left out of the figures and the tables still have it), a different marker shape per LLM so that identity never rests on colour alone,
a legend whenever there are two or more LLMs, a surface-coloured ring around every marker, text in the ink tokens (never the series
colour), a recessive grid. The third hue (aqua) is below 3:1 contrast on the light surface, which the rules answer with a legend, the
marker shapes and the table view: all three are present.
"""
import os

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a")          # validated: node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
MARKERS = ("o", "s", "^")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MAX_LLMS = len(PALETTE)

PANELS = (("M1", "M1 first-pass valid", "share of parsed raw routes"),
          ("M2p", "M2 LLM-final share", "share of final routes from the LLM"),
          ("M6", "M6 efficiency gap", "points below the best Route Builder variant (lower is better)"))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=3)


def _legend(fig, models, handles_ax):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=MARKERS[i], linestyle="none", markersize=9, markerfacecolor=PALETTE[i], markeredgecolor=SURFACE,
                      markeredgewidth=1.5, label=m["label"]) for i, m in enumerate(models)]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.995, 0.995), handletextpad=0.4, columnspacing=1.6)


def _points(res, models):
    """{model id: {row id: cell}} for the cells that have data."""
    return {m["id"]: {r["id"]: res["usable"][(r["id"], m["id"])]["cell"] for r in res["cfg"]["rows"] if (r["id"], m["id"]) in res["usable"]}
            for m in models}


def make_figures(res: dict, out_dir: str):
    """Writes the figures that the data allows. Returns (file names made, a reason when something was left out or None)."""
    cfg = res["cfg"]
    models = cfg["models"][:MAX_LLMS]
    notes = []
    if len(cfg["models"]) > MAX_LLMS:
        notes.append(f"figures show the first {MAX_LLMS} LLMs only (the palette is validated for {MAX_LLMS} series; tables have all)")
    pts = _points(res, models)
    if not any(pts[m["id"]] for m in models):
        return [], "no figures: no usable run yet"
    try:
        plt = _plt()
    except ImportError:
        return [], "no figures: matplotlib is not installed (pip install matplotlib)"
    made = [_ablation(plt, res, models, pts, out_dir)]
    if any(c["M7"]["mean"] is not None and c["M6"]["mean"] is not None for m in models for c in pts[m["id"]].values()):
        made.append(_cost(plt, res, models, pts, out_dir))
    return made, "; ".join(notes) or None


def _ablation(plt, res, models, pts, out_dir):
    rows = [r for r in res["cfg"]["rows"] if r["built"] and any(r["id"] in pts[m["id"]] for m in models)]
    n = len(rows)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.0, 1.6 + 0.42 * n + 0.18 * n * len(models)), sharey=True, facecolor=SURFACE)
    offsets = [0.0] if len(models) == 1 else [(-0.5 + i / (len(models) - 1)) * 0.5 for i in range(len(models))]
    for ax, (key, title, sub) in zip(axes, PANELS):
        _style(ax)
        for i, m in enumerate(models):
            for y, r in enumerate(rows):
                cell = pts[m["id"]].get(r["id"])
                stat = None if cell is None else cell[key]
                if stat is None or stat["mean"] is None:
                    continue
                yy = y + offsets[i]
                if stat["sd"]:
                    ax.plot([stat["mean"] - stat["sd"], stat["mean"] + stat["sd"]], [yy, yy], color=PALETTE[i], linewidth=1.5, zorder=2,
                            solid_capstyle="round")
                ax.plot([stat["mean"]], [yy], linestyle="none", marker=MARKERS[i], markersize=9, markerfacecolor=PALETTE[i],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=25)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
        if key in ("M1", "M2p"):
            ax.set_xlim(-0.03, 1.03)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r["id"] for r in rows], color=INK2, fontsize=9)
        ax.set_ylim(n - 0.4, -0.6)
        if key == "M6":
            ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    if len(models) >= 2:
        _legend(fig, models, axes[0])
    fig.suptitle(f"{res['cfg']['name']}: ablation by configuration (mean, whisker = SD over repeats)", x=0.01, ha="left", y=0.995,
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, "fig_ablation.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return "fig_ablation.png"


def _cost(plt, res, models, pts, out_dir):
    """One panel per LLM (same axes): the ablation path A0 -> A1 -> ... of that LLM. Rows with the same coordinates share one label."""
    k = len(models)
    fig, axes = plt.subplots(1, k, figsize=(4.3 * k + 0.6, 4.6), sharex=True, sharey=True, facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    order = [r["id"] for r in res["cfg"]["rows"] if r["built"]]
    for i, (ax, m) in enumerate(zip(axes, models)):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        seq = []
        for rid in order:
            cell = pts[m["id"]].get(rid)
            if cell and cell["M7"]["mean"] is not None and cell["M6"]["mean"] is not None:
                seq.append((rid, cell["M7"]["mean"], cell["M6"]["mean"]))
        if seq:
            ax.plot([x for _, x, _ in seq], [y for _, _, y in seq], color=PALETTE[i], linewidth=2, zorder=2, solid_joinstyle="round")
            ax.plot([x for _, x, _ in seq], [y for _, _, y in seq], linestyle="none", marker=MARKERS[i], markersize=9,
                    markerfacecolor=PALETTE[i], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            groups = {}
            for rid, x, y in seq:
                groups.setdefault((round(x, 6), round(y, 6)), []).append(rid)
            for n, ((x, y), ids) in enumerate(groups.items()):
                up = n % 2 == 0
                ax.annotate("/".join(ids), (x, y), textcoords="offset points", xytext=(7, 7 if up else -14), fontsize=8.5, color=INK2)
        ax.set_title(m["label"], loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("API calls per part (all calls)", color=INK2, fontsize=9)
        ax.margins(x=0.12, y=0.18)
    axes[0].set_ylabel("M6 efficiency gap, points (lower is better)", color=INK2, fontsize=9)
    fig.suptitle(f"{res['cfg']['name']}: what each added component costs and buys", x=0.01, ha="left", y=0.995, fontsize=11, color=INK)
    if k >= 2:
        _legend(fig, models, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out_dir, "fig_cost.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return "fig_cost.png"
