from pathlib import Path
import numpy as np


# ── palette helpers ───────────────────────────────────────────────────────────

# N=2 keeps the original three-group colours for backward visual compatibility.
_COLORS_2STIM = ["#4fa1ca", "#bb70b6", "#110979"]
# N=1 or N≥3: tab10 cycle
_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _group_colors(N, group_sizes):
    """Return one color per display group."""
    if N == 2:
        return _COLORS_2STIM          # stim1only / both / stim2only
    return [_TAB10[j % len(_TAB10)] for j in range(len(group_sizes))]


def _stim_names(N, names=None):
    if names:
        return list(names)
    return [f"Stimulus {i + 1}" for i in range(N)]


# ── show_plots ────────────────────────────────────────────────────────────────

def show_plots(resp_n, nums, group_sizes,
               stim_onset_idx: int, ses_f: int,
               fp: float, pre_s: float, stim_s: float,
               results_dir: str, stim_names=None):
    """Heatmap + bar-chart for N stimuli.

    Parameters
    ----------
    resp_n       list of N arrays (K_resp, T) — sorted responders
    nums         N=2: [stim1only, both, stim2only]
                 N=1: [n_resp]
                 N≥3: [n_resp_per_stim, …]
    group_sizes  list of int — rows per sidebar colour band
    stim_names   optional list of N axis labels
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    N  = len(resp_n)
    n  = resp_n[0].shape[0]
    T  = resp_n[0].shape[1] if n > 0 else (stim_onset_idx + round(stim_s / fp))
    names = _stim_names(N, stim_names)
    gcols = _group_colors(N, group_sizes)

    stim_dur_f = T - stim_onset_idx
    analyzed_s = round(stim_dur_f * fp)
    tick_frames = [0, stim_onset_idx, T]
    tick_labels = [f"-{round(stim_onset_idx * fp)}", "0", f"+{analyzed_s}"]
    stim_note = (f"  ⚠ recording shorter than requested {round(stim_s)} s"
                 if analyzed_s < round(stim_s) else "")

    # ── responder heatmap ─────────────────────────────────────────────────────
    fig = plt.figure(constrained_layout=True, figsize=(2 + 10 * N, 5))
    gs  = gridspec.GridSpec(1, 1 + N, figure=fig, width_ratios=[1] + [10] * N)
    ax_side = fig.add_subplot(gs[0])
    ax_heats = [fig.add_subplot(gs[1 + j]) for j in range(N)]

    ax_side.set_xlim(0, 1); ax_side.set_ylim(0, max(n, 1)); ax_side.invert_yaxis()
    ax_side.xaxis.set_visible(False)
    cumsum = 0
    for gsize, gcol in zip(group_sizes, gcols):
        ax_side.axhspan(cumsum, cumsum + gsize, facecolor=gcol)
        cumsum += gsize
    ax_side.set_yticks([0, max(n - 1, 0)], [1, max(n, 1)])
    ax_side.set_ylabel("Neuron #")

    im = None
    for j, ax in enumerate(ax_heats):
        if n > 0:
            im = ax.imshow(resp_n[j], aspect="auto", vmin=0, vmax=8)
        ax.axvline(stim_onset_idx, color="w", lw=0.8, ls="--")
        ax.set_title(names[j])
        ax.yaxis.set_visible(False)
        ax.set_xlabel("Time (s, stim onset = 0)")
        ax.set_xticks(tick_frames, tick_labels)
    if im is not None:
        fig.colorbar(im, ax=ax_heats, shrink=0.5, label="z-score")

    fig.suptitle(f"Responder heatmap  ({n} neurons,  stim = {analyzed_s} s){stim_note}")
    fig.savefig(Path(results_dir) / "heatmap.png", dpi=150, bbox_inches="tight")

    # ── bar chart ─────────────────────────────────────────────────────────────
    if N == 2:
        bar_labels = [f"{names[0]}\nonly", "Both", f"{names[1]}\nonly"]
        bar_vals   = nums
        bar_colors = gcols
        n_denom    = max(sum(nums), 1)
        ylabel     = "% responsive neurons"
        pct = [v / n_denom * 100 for v in bar_vals]
    elif N == 1:
        bar_labels = [names[0]]
        bar_vals   = nums
        bar_colors = gcols[:1]
        pct = [nums[0] / max(n, 1) * 100]
        ylabel = "% responsive neurons"
    else:
        bar_labels = names
        bar_vals   = nums
        bar_colors = gcols[:N]
        pct = [v / max(n, 1) * 100 for v in nums]
        ylabel = "% neurons responding to each stim"

    fig2, ax2 = plt.subplots(figsize=(max(4, 2 * len(bar_labels)), 4))
    ax2.bar(bar_labels, pct, color=bar_colors)
    ax2.set_ylabel(ylabel)
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(Path(results_dir) / "breakdown.png", dpi=150, bbox_inches="tight")

    plt.show()


# ── show_region_plots ─────────────────────────────────────────────────────────

def show_region_plots(region_results: dict,
                      stim_onset_idx: int, ses_f: int,
                      fp: float, pre_s: float, stim_s: float,
                      results_dir: str, stim_names=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    regions   = list(region_results.keys())
    n_regions = len(regions)
    reg_colors = ["#e6c84a", "#4ac0e6"]

    # Infer N from the first region's data
    first = next(iter(region_results.values()))
    N = len(first['resp_n'])
    names = _stim_names(N, stim_names)

    if N == 2:
        bar_labels = [f"{names[0]}\nonly", "Both", f"{names[1]}\nonly"]
        bar_colors = _COLORS_2STIM
    else:
        bar_labels = names
        bar_colors = [_TAB10[j % len(_TAB10)] for j in range(N)]

    tick_frames = [0, stim_onset_idx]
    tick_labels = [f"-{round(stim_onset_idx * fp)}", "0"]

    # ── bar chart per region ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_regions, figsize=(4 * n_regions, 4),
                             squeeze=False)
    for col, (reg_name, rr) in enumerate(region_results.items()):
        ax = axes[0, col]
        nums_r   = rr['nums']
        n_total  = rr['n_total']

        if N == 2:
            n_resp1 = nums_r[0] + nums_r[1]
            n_resp2 = nums_r[1] + nums_r[2]
            info = (f"Total: {n_total}   |   "
                    f"Stim1 resp: {n_resp1}   Stim2 resp: {n_resp2}")
        else:
            parts = "   ".join(f"{names[j]} resp: {nums_r[j]}" for j in range(N))
            info  = f"Total: {n_total}   |   {parts}"

        ax.bar(bar_labels, nums_r, color=bar_colors)
        ax.set_title(reg_name, color=reg_colors[col % 2],
                     fontweight="bold", fontsize=12)
        ax.set_ylabel("Neuron count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, 1.12, info, ha="center", transform=ax.transAxes,
                fontsize=8, color="gray")

    fig.suptitle("Sub-region neuron breakdown", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(results_dir) / "region_breakdown.png", dpi=150, bbox_inches="tight")

    # ── mean ± std trace per region × stimulus ────────────────────────────────
    fig2 = plt.figure(figsize=(7 * n_regions, 4 * N), constrained_layout=True)
    outer = gridspec.GridSpec(1, n_regions, figure=fig2)

    for col, (reg_name, rr) in enumerate(region_results.items()):
        inner = gridspec.GridSpecFromSubplotSpec(N, 1, subplot_spec=outer[col],
                                                 hspace=0.4)
        for j in range(N):
            resp = rr['resp_n'][j]
            ax = fig2.add_subplot(inner[j])
            if resp.shape[0] > 0:
                mean = resp.mean(axis=0)
                std  = resp.std(axis=0)
                x    = np.arange(resp.shape[1])
                ax.plot(x, mean, color=reg_colors[col % 2], lw=1.5)
                ax.fill_between(x, mean - std, mean + std,
                                color=reg_colors[col % 2], alpha=0.25)
            ax.axvline(stim_onset_idx, color="gray", lw=0.8, ls="--")
            ax.axhline(0, color="gray", lw=0.5, ls=":")
            ax.set_title(f"{reg_name} — {names[j]}", fontsize=9)
            ax.set_ylabel("z-score")
            ax.set_xlabel("Time (s, stim onset = 0)")
            ax.set_xticks(tick_frames, tick_labels)
            ax.spines[["top", "right"]].set_visible(False)

    fig2.suptitle("Sub-region mean z-score responses", fontweight="bold")
    fig2.savefig(Path(results_dir) / "region_traces.png", dpi=150, bbox_inches="tight")

    plt.show()


# ── show_spatial_response_map ─────────────────────────────────────────────────

def show_spatial_response_map(label: str, spatial_data: list,
                               threshold: float, results_dir: str):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from skimage.measure import find_contours

    reg_colors = ["#e6c84a", "#4ac0e6"]

    for d in spatial_data:
        z         = d['z']
        anatomy   = d['anatomy']
        centers   = d['centers']
        stim1_mdn = d['stim1_mdn']
        stim2_mdn = d['stim2_mdn']
        sreg      = d.get('subregion_masks')

        stim_idx = stim2_mdn - stim1_mdn
        stim_idx[np.maximum(stim1_mdn, stim2_mdn) < threshold] = np.nan

        p_lo = np.percentile(anatomy, 0.5)
        p_hi = np.percentile(anatomy, 99.5)
        anat_disp = np.clip((anatomy - p_lo) / max(p_hi - p_lo, 1e-9), 0, 1)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(anat_disp, cmap='gray', origin='upper')
        scat = ax.scatter(centers[:, 1], centers[:, 0],
                          c=stim_idx, cmap='bwr', vmin=-10, vmax=10,
                          s=120, edgecolors='none', alpha=0.9)

        if sreg is not None:
            sigma = max(8, min(anatomy.shape) // 30)
            for reg_i, color in enumerate(reg_colors):
                if reg_i != 0:
                    continue
                if reg_i >= sreg.shape[0]:
                    break
                ctrs = find_contours(sreg[reg_i].astype(float), 0.5)
                if not ctrs:
                    continue
                c = max(ctrs, key=len)
                sr = gaussian_filter1d(c[:, 0], sigma=sigma, mode='wrap')
                sc = gaussian_filter1d(c[:, 1], sigma=sigma, mode='wrap')
                ax.plot(np.append(sc, sc[0]), np.append(sr, sr[0]),
                        color=color, lw=2.5, ls='--', alpha=0.9)

        fig.colorbar(scat, ax=ax, shrink=0.7,
                     label='Stim 2 − Stim 1 (median z-score)')
        ax.set_title(f'{label}  —  {z}', fontsize=12, fontweight='bold')
        ax.axis('off')
        fig.tight_layout()
        fname = Path(results_dir) / f'spatial_response_map_{label}_{z}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()
