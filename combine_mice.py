"""Pool per-mouse analysis results into combined tables and figures.

Each folder must contain analysis/neurons.csv and analysis/traces_stim<j>.npy,
written by the GUI's Analysis stage. Mice analysed before those files existed
need the Analysis stage re-run once (motion correction and CNMF unchecked);
that re-reads the saved CNMF results, not the movies.

Usage (Anaconda Prompt, from the project folder):

    python combine_mice.py D:/out/ZH537 D:/out/ZH539 D:/out/ZH541 --out D:/out/combined
    python combine_mice.py D:/out/ZH537 D:/out/ZH539 --out D:/out/combined --split-region

Outputs in --out:
    pooled_neurons.csv          every neuron from every mouse, one row each
    mouse_summary.csv           one row per mouse: neurons, responders, % per group,
                                repeated for AP and NTS when outlined
    mean_traces_<subset>.csv    per-mouse mean z-score trace of each stimulus's
                                responders, plus the across-mouse mean and SEM
    pooled_heatmap[_AP|_NTS].png
    mean_sem_traces.png
    responders_by_mouse.png

Mean traces for stimulus j average the neurons that respond to stimulus j
(for 2 stimuli that is "only" + "both"). Mean ± SEM is across mice, n = mice.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from analysis.responders import (classify_from_medians, group_names, group_labels,
                                 sort_key)
from analysis.results_io import load_neuron_results, REGIONS
from visualization.response_plots import _group_colors

NONRESP_COLOR = "#bdbdbd"


# ── loading and pooling ───────────────────────────────────────────────────────

def load_mice(folders):
    """Load every folder; mouse labels are made unique across folders."""
    runs, seen = [], set()
    for folder in folders:
        run = load_neuron_results(folder)
        d = run["dir"]
        src_name = d.parent.name if d.name == "analysis" else d.name
        table = run["table"]
        rename = {}
        for m in dict.fromkeys(table["mouse"]):
            new, i = m, 1
            while new in seen:
                new = f"{m} ({src_name})" if i == 1 else f"{m} ({src_name} {i})"
                i += 1
            rename[m] = new
            seen.add(new)
        table["mouse"] = table["mouse"].map(rename)
        table.insert(1, "source", str(d))
        runs.append(run)
    return runs


def pool(runs, threshold=None):
    """Concatenate tables and onset-aligned traces from every run.

    Traces are cropped to the window every run shares around stimulus onset.
    With `threshold`, responders and groups are re-derived from the saved medians.
    Returns (table, traces, info); traces[j] rows match table rows.
    """
    Ns = {len(r["traces"]) for r in runs}
    if len(Ns) != 1:
        raise ValueError(f"Mice have different numbers of stimuli: {sorted(Ns)}")
    N = Ns.pop()

    for r in runs:
        for key in ("frame_period", "stim_onset_idx", "threshold"):
            if key not in r["params"]:
                raise ValueError(f"{r['dir'] / 'params.yaml'} has no '{key}'; "
                                 "re-run the Analysis stage for this mouse.")
    fps = [float(r["params"]["frame_period"]) for r in runs]
    if max(fps) - min(fps) > 1e-6:
        raise ValueError(
            f"Frame periods differ between mice ({sorted(set(fps))} s/frame); "
            "traces cannot be aligned frame by frame.")

    notes = []
    onsets = [int(r["params"]["stim_onset_idx"]) for r in runs]
    posts = [r["traces"][0].shape[1] - o for r, o in zip(runs, onsets)]
    pre, post = min(onsets), min(posts)
    if len(set(onsets)) > 1 or len(set(posts)) > 1:
        notes.append(
            f"Traces cropped to the window all mice share: {pre} frames before "
            f"and {post} frames from stimulus onset.")
    stim_s = {r["params"].get("stim_s") for r in runs}
    if len(stim_s) > 1:
        notes.append(
            f"Stimulus durations differ ({sorted(stim_s)} s); each mouse's medians "
            "and responders were computed over its own stimulus window.")

    table = pd.concat([r["table"] for r in runs], ignore_index=True)
    traces = [np.vstack([r["traces"][j][:, o - pre:o + post]
                         for r, o in zip(runs, onsets)]).astype(np.float32)
              for j in range(N)]

    thresholds = sorted({float(r["params"]["threshold"]) for r in runs})
    if threshold is not None:
        medians = table[[f"median_z_stim{j + 1}" for j in range(N)]].to_numpy(float)
        responds, group = classify_from_medians(medians, threshold)
        names = group_names(N)
        for j in range(N):
            table[f"responds_stim{j + 1}"] = responds[:, j].astype(int)
        table["responder"] = (group >= 0).astype(int)
        table["group"] = [names[g] if g >= 0 else "none" for g in group]
        thresholds = [threshold]
    elif len(thresholds) > 1:
        notes.append(
            f"Mice were classified at different thresholds {thresholds}; "
            "pass --threshold to use one for all.")

    info = dict(n_stims=N, frame_period=fps[0], pre=pre, post=post,
                thresholds=thresholds, notes=notes)
    return table, traces, info


def group_codes(table, N):
    """Group name column → int code (-1 = none)."""
    lookup = {name: i for i, name in enumerate(group_names(N))}
    return table["group"].map(lookup).fillna(-1).astype(int).to_numpy()


def mouse_order(table):
    return list(dict.fromkeys(table["mouse"]))


def mouse_colors(mice):
    cmap = plt.get_cmap("tab10" if len(mice) <= 10 else "tab20")
    return {m: cmap(i % cmap.N) for i, m in enumerate(mice)}


# ── tables ────────────────────────────────────────────────────────────────────

def _pct(a, b):
    return round(100.0 * a / b, 2) if b else np.nan


def _counts(t, N, suffix):
    n = len(t)
    n_resp = int(t["responder"].sum())
    c = {f"n_neurons{suffix}": n,
         f"n_responders{suffix}": n_resp,
         f"pct_responders{suffix}": _pct(n_resp, n)}
    if N > 1:   # with one stimulus the single group is the responders themselves
        for g in group_names(N):
            k = int((t["group"] == g).sum())
            c[f"n_{g}{suffix}"] = k
            c[f"pct_{g}{suffix}"] = _pct(k, n)
            c[f"pct_of_responders_{g}{suffix}"] = _pct(k, n_resp)
        for j in range(N):
            k = int(t[f"responds_stim{j + 1}"].sum())
            c[f"n_resp_stim{j + 1}{suffix}"] = k
            c[f"pct_resp_stim{j + 1}{suffix}"] = _pct(k, n)
    return c


def mouse_summary(table, N):
    """One row per mouse; percentages are of all neurons unless named otherwise."""
    regions = [reg for reg in REGIONS if (table["region"] == reg).any()]
    rows = []
    for mouse, t in table.groupby("mouse", sort=False):
        row = {"mouse": mouse, "source": t["source"].iloc[0]}
        row.update(_counts(t, N, ""))
        for reg in regions:
            row.update(_counts(t[t["region"] == reg], N, f"_{reg}"))
        rows.append(row)
    return pd.DataFrame(rows)


def mean_traces(table, traces, j, rows):
    """Per-mouse mean trace of the neurons in `rows` that respond to stimulus j.

    Returns {mouse: (trace, n_neurons)}, skipping mice with no such neurons.
    """
    sel_resp = rows & (table[f"responds_stim{j + 1}"].to_numpy() == 1)
    mice = table["mouse"].to_numpy()
    out = {}
    for m in mouse_order(table):
        sel = sel_resp & (mice == m)
        if sel.any():
            out[m] = (traces[j][sel].mean(axis=0), int(sel.sum()))
    return out


def across_mice(per_mouse, T):
    """Mean and SEM across mice (SEM is NaN with fewer than two mice)."""
    if not per_mouse:
        return np.full(T, np.nan), np.full(T, np.nan)
    arr = np.array([tr for tr, _ in per_mouse.values()], dtype=float)
    mean = arr.mean(axis=0)
    if len(arr) < 2:
        return mean, np.full(T, np.nan)
    return mean, arr.std(axis=0, ddof=1) / np.sqrt(len(arr))


def time_axis(info, T):
    return (np.arange(T) - info["pre"]) * info["frame_period"]


def mean_traces_table(table, traces, info, rows):
    T = traces[0].shape[1]
    out = {"time_s": np.round(time_axis(info, T), 4)}
    for j in range(info["n_stims"]):
        per_mouse = mean_traces(table, traces, j, rows)
        for m, (tr, _) in per_mouse.items():
            # "mouse_" prefix keeps this distinct from the _mean/_sem columns
            # below even if a mouse is literally named "mean" or "sem"
            out[f"stim{j + 1}_mouse_{m}"] = tr
        mean, sem = across_mice(per_mouse, T)
        out[f"stim{j + 1}_mean"] = mean
        out[f"stim{j + 1}_sem"] = sem
    return pd.DataFrame(out)


# ── figures ───────────────────────────────────────────────────────────────────

def _time_ticks(ax, info, T):
    fp = info["frame_period"]
    ax.set_xticks([0, info["pre"], T - 1],
                  [f"-{round(info['pre'] * fp)}", "0", f"+{round(info['post'] * fp)}"])


def plot_heatmap(table, traces, info, rows, colors, stim_names, path, title,
                 sort="group"):
    """Pooled heatmap with a mouse colour bar and a response-group colour bar."""
    N = info["n_stims"]
    idx = np.flatnonzero(rows)
    if not len(idx):
        print(f"  {Path(path).name}: no neurons to plot, skipped")
        return False

    medians = table[[f"median_z_stim{j + 1}" for j in range(N)]].to_numpy(float)[idx]
    codes = group_codes(table, N)[idx]
    key = sort_key(medians, codes)
    n_groups = len(group_names(N))
    group_rank = np.where(codes < 0, n_groups, codes)
    mice = mouse_order(table)
    mouse_rank = table["mouse"].map({m: i for i, m in enumerate(mice)}).to_numpy()[idx]
    keys = (-key, group_rank) if sort == "group" else (-key, group_rank, mouse_rank)
    order = idx[np.lexsort(keys)]

    K, T = len(order), traces[0].shape[1]
    gcols = _group_colors(N, [0] * n_groups)
    code_sorted = group_codes(table, N)[order]
    group_rgb = np.array([to_rgb(gcols[c]) if c >= 0 else to_rgb(NONRESP_COLOR)
                          for c in code_sorted])[:, None, :]
    mouse_rgb = np.array([to_rgb(colors[m])
                          for m in table["mouse"].to_numpy()[order]])[:, None, :]

    fig = plt.figure(figsize=(2.5 + 6 * N, float(np.clip(3 + K / 80, 5, 14))),
                     constrained_layout=True)
    gs = fig.add_gridspec(1, 2 + N, width_ratios=[0.35, 0.35] + [6] * N)
    ax_m = fig.add_subplot(gs[0])
    ax_g = fig.add_subplot(gs[1], sharey=ax_m)
    ax_m.imshow(mouse_rgb, aspect="auto", interpolation="nearest")
    ax_g.imshow(group_rgb, aspect="auto", interpolation="nearest")
    ax_m.set_xticks([])
    ax_g.set_xticks([])
    ax_m.set_title("mouse", fontsize=8)
    ax_g.set_title("group", fontsize=8)
    ax_m.set_ylabel("Neuron #")
    ax_g.tick_params(labelleft=False)

    heat_axes, im = [], None
    for j in range(N):
        ax = fig.add_subplot(gs[2 + j], sharey=ax_m)
        im = ax.imshow(traces[j][order], aspect="auto", vmin=0, vmax=8,
                       interpolation="nearest")
        ax.axvline(info["pre"], color="w", lw=0.8, ls="--")
        ax.set_title(stim_names[j])
        ax.tick_params(labelleft=False)
        ax.set_xlabel("Time (s, stim onset = 0)")
        _time_ticks(ax, info, T)
        heat_axes.append(ax)
    fig.colorbar(im, ax=heat_axes, shrink=0.5, label="z-score")

    present = [m for m in mice if m in set(table["mouse"].to_numpy()[order])]
    handles = [Patch(color=colors[m], label=m) for m in present]
    labels = group_labels(N, stim_names)
    handles += [Patch(color=gcols[g], label=labels[g]) for g in range(n_groups)
                if (code_sorted == g).any()]
    if (code_sorted < 0).any():
        handles.append(Patch(color=NONRESP_COLOR, label="Non-responder"))
    fig.legend(handles=handles, loc="outside lower center",
               ncols=min(len(handles), 6), fontsize=8, frameon=False)

    fig.suptitle(f"{title}  ({K} neurons, {len(present)} mice)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_mean_traces(table, traces, info, subsets, colors, stim_names, path):
    """Rows = subsets (All / AP / NTS), columns = stimuli; thin lines are mice."""
    N = info["n_stims"]
    T = traces[0].shape[1]
    t = time_axis(info, T)
    fig, axes = plt.subplots(len(subsets), N, figsize=(5 * N, 3.2 * len(subsets)),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    for row, (name, rows) in enumerate(subsets):
        for j in range(N):
            ax = axes[row, j]
            per_mouse = mean_traces(table, traces, j, rows)
            for m, (tr, _) in per_mouse.items():
                ax.plot(t, tr, color=colors[m], lw=0.8, alpha=0.6)
            mean, sem = across_mice(per_mouse, T)
            if per_mouse:
                ax.plot(t, mean, color="k", lw=2)
                if len(per_mouse) > 1:
                    ax.fill_between(t, mean - sem, mean + sem, color="k", alpha=0.2, lw=0)
            n_neurons = sum(n for _, n in per_mouse.values())
            ax.axvline(0, color="gray", lw=0.8, ls="--")
            ax.axhline(0, color="gray", lw=0.5, ls=":")
            ax.set_title(f"{name} · {stim_names[j]} responders\n"
                         f"{len(per_mouse)} mice, {n_neurons} neurons", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("z-score")
            if row == len(subsets) - 1:
                ax.set_xlabel("Time (s, stim onset = 0)")
    handles = [Patch(color=colors[m], label=m) for m in mouse_order(table)]
    handles.append(Patch(color="k", label="mean ± SEM across mice"))
    fig.legend(handles=handles, loc="outside lower center",
               ncols=min(len(handles), 6), fontsize=8, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _percent_metrics(N, stim_names):
    if N == 1:
        return [("Responders", "pct_responders")]
    metrics = [("Any stimulus", "pct_responders")]
    if N == 2:
        labels = group_labels(N, stim_names)
        metrics += [(lbl, f"pct_{g}") for lbl, g in zip(labels, group_names(N))]
    else:
        metrics += [(f"{s} (any)", f"pct_resp_stim{j + 1}")
                    for j, s in enumerate(stim_names)]
    return metrics


def plot_percentages(summary, N, subset_names, colors, stim_names, path):
    """Bars = mean ± SEM across mice, dots = individual mice."""
    metrics = _percent_metrics(N, stim_names)
    names = [s for s in subset_names
             if f"n_neurons{'' if s == 'All' else '_' + s}" in summary.columns]
    fig, axes = plt.subplots(1, len(names), figsize=(1.3 * len(metrics) * len(names) + 1.5, 4),
                             squeeze=False, sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    for col, name in enumerate(names):
        ax = axes[0, col]
        suffix = "" if name == "All" else f"_{name}"
        for x, (label, column) in enumerate(metrics):
            vals = summary[column + suffix].to_numpy(float)
            ok = ~np.isnan(vals)
            if not ok.any():
                continue
            mean = vals[ok].mean()
            sem = vals[ok].std(ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else 0
            ax.bar(x, mean, color="#d9d9d9", width=0.7)
            ax.errorbar(x, mean, yerr=sem, color="k", capsize=4, lw=1)
            jitter = rng.uniform(-0.18, 0.18, ok.sum())
            ax.scatter(x + jitter, vals[ok], s=22, zorder=3,
                       color=[colors[m] for m in summary["mouse"][ok]])
        ax.set_xticks(range(len(metrics)), [lbl for lbl, _ in metrics],
                      rotation=30, ha="right")
        ax.set_title(name)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel("% of neurons")
    handles = [Patch(color=colors[m], label=m) for m in summary["mouse"]]
    fig.legend(handles=handles, loc="outside right center", fontsize=8, frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="See the docstring at the top of combine_mice.py for the outputs.")
    ap.add_argument("folders", nargs="+",
                    help="mouse output folders (or their analysis/ subfolders)")
    ap.add_argument("--out", required=True, help="folder for the combined results")
    ap.add_argument("--split-region", action="store_true",
                    help="also plot AP and NTS neurons separately")
    ap.add_argument("--sort", choices=["group", "mouse"], default="group",
                    help="heatmap rows: pooled by response group (default), "
                         "or grouped by mouse first")
    ap.add_argument("--include-nonresponders", action="store_true",
                    help="show non-responders in the heatmaps (grey group bar)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="re-classify every mouse at this z-score threshold "
                         "instead of each mouse's saved one")
    ap.add_argument("--stim-names", nargs="+", default=None,
                    help="labels for the stimuli, e.g. --stim-names Saline CCK")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_mice(args.folders)
    table, traces, info = pool(runs, args.threshold)
    for note in info["notes"]:
        print(f"Note: {note}")
    N = info["n_stims"]
    stim_names = args.stim_names or [f"Stimulus {j + 1}" for j in range(N)]
    if len(stim_names) != N:
        raise SystemExit(f"--stim-names needs {N} names, got {len(stim_names)}")

    mice = mouse_order(table)
    colors = mouse_colors(mice)

    table.to_csv(out / "pooled_neurons.csv", index=False, float_format="%.6f")
    summary = mouse_summary(table, N)
    summary.to_csv(out / "mouse_summary.csv", index=False)

    subsets = [("All", np.ones(len(table), dtype=bool))]
    if args.split_region:
        for reg in REGIONS:
            in_reg = (table["region"] == reg).to_numpy()
            if in_reg.any():
                subsets.append((reg, in_reg))
            else:
                print(f"No {reg} neurons in any mouse; {reg} plots skipped.")

    shown = (np.ones(len(table), dtype=bool) if args.include_nonresponders
             else table["responder"].to_numpy() == 1)
    what = "neurons" if args.include_nonresponders else "responders"
    thr = ", ".join(f"{t:g}" for t in info["thresholds"])
    for name, rows in subsets:
        suffix = "" if name == "All" else f"_{name}"
        label = f"Pooled {what}" + ("" if name == "All" else f" · {name}")
        plot_heatmap(table, traces, info, rows & shown, colors, stim_names,
                     out / f"pooled_heatmap{suffix}.png", f"{label} · z > {thr}",
                     sort=args.sort)
        mean_traces_table(table, traces, info, rows).to_csv(
            out / f"mean_traces_{name}.csv", index=False, float_format="%.6f")

    plot_mean_traces(table, traces, info, subsets, colors, stim_names,
                     out / "mean_sem_traces.png")
    plot_percentages(summary, N, [name for name, _ in subsets], colors, stim_names,
                     out / "responders_by_mouse.png")

    print(f"Combined {len(mice)} mice ({len(table)} neurons, "
          f"{int(table['responder'].sum())} responders) into {out}")


if __name__ == "__main__":
    sys.exit(main())
