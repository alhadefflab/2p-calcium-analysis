from pathlib import Path
import numpy as np


def show_plots(resp1, resp2, nums,
               stim_onset_idx: int, ses_f: int,
               fp: float, pre_s: float, stim_s: float, results_dir: str):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 10, 10])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    n = resp1.shape[0]
    ax0.set_xlim(0, 1); ax0.set_ylim(0, n); ax0.invert_yaxis()
    ax0.xaxis.set_visible(False)
    ax0.axhspan(0,               nums[0],             facecolor="#4fa1ca")
    ax0.axhspan(nums[0],         nums[0] + nums[1],   facecolor="#bb70b6")
    ax0.axhspan(nums[0]+nums[1], n,                   facecolor="#110979")
    ax0.set_yticks([0, max(n - 1, 0)], [1, n])
    ax0.set_ylabel("Neuron #")

    im = ax1.imshow(resp1, aspect="auto", vmin=0, vmax=8)
    ax2.imshow(resp2, aspect="auto", vmin=0, vmax=8)

    stim_dur_f  = resp1.shape[1] - stim_onset_idx
    analyzed_s  = round(stim_dur_f * fp)
    tick_frames = [0, stim_onset_idx, resp1.shape[1]]
    tick_labels = [
        f"-{round(stim_onset_idx * fp)}",
        "0",
        f"+{analyzed_s}",
    ]
    for ax, title in ((ax1, "Stimulus 1"), (ax2, "Stimulus 2")):
        ax.axvline(stim_onset_idx, color="w", lw=0.8, ls="--")
        ax.set_title(title)
        ax.yaxis.set_visible(False)
        ax.set_xlabel("Time (s, stim onset = 0)")
        ax.set_xticks(tick_frames, tick_labels)
    fig.colorbar(im, ax=[ax1, ax2], shrink=0.5, label="z-score")
    stim_note = (f"  ⚠ recording shorter than requested {round(stim_s)} s"
                 if analyzed_s < round(stim_s) else "")
    fig.suptitle(f"Responder heatmap  (stim = {analyzed_s} s){stim_note}")
    fig.savefig(Path(results_dir) / "heatmap.png", dpi=150, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(4, 4))
    pct = [v / n * 100 for v in nums] if n else [0, 0, 0]
    ax.bar(["Stim 1\nonly", "Both", "Stim 2\nonly"], pct,
           color=["#4fa1ca", "#bb70b6", "#110979"])
    ax.set_ylabel("% responsive neurons")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(Path(results_dir) / "breakdown.png", dpi=150, bbox_inches="tight")

    plt.show()


def show_region_plots(region_results: dict,
                      stim_onset_idx: int, ses_f: int,
                      fp: float, pre_s: float, stim_s: float,
                      results_dir: str):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    regions    = list(region_results.keys())
    n_regions  = len(regions)
    bar_colors = ["#4fa1ca", "#bb70b6", "#110979"]
    reg_colors = ["#e6c84a", "#4ac0e6"]

    fig, axes = plt.subplots(1, n_regions, figsize=(4 * n_regions, 4),
                             sharey=False, squeeze=False)
    for col, (reg_name, rr) in enumerate(region_results.items()):
        ax = axes[0, col]
        nums_r   = rr['nums']
        n_total  = rr['n_total']
        n_resp1  = nums_r[0] + nums_r[1]
        n_resp2  = nums_r[1] + nums_r[2]

        labels = ["Stim 1\nonly", "Both", "Stim 2\nonly"]
        ax.bar(labels, nums_r, color=bar_colors)
        ax.set_title(reg_name, color=reg_colors[col % 2],
                     fontweight="bold", fontsize=12)
        ax.set_ylabel("Neuron count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, 1.12,
                f"Total: {n_total}   |   Stim1 resp: {n_resp1}   Stim2 resp: {n_resp2}",
                ha="center", transform=ax.transAxes,
                fontsize=8, color="gray")

    fig.suptitle("Sub-region neuron breakdown", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(results_dir) / "region_breakdown.png", dpi=150, bbox_inches="tight")

    tick_frames = [0, stim_onset_idx]
    tick_labels = [f"-{round(stim_onset_idx * fp)}", "0"]

    fig2 = plt.figure(figsize=(7 * n_regions, 7), constrained_layout=True)
    outer = gridspec.GridSpec(1, n_regions, figure=fig2, hspace=0.05)

    for col, (reg_name, rr) in enumerate(region_results.items()):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[col],
                                                 hspace=0.4)
        for row, (stim_label, resp) in enumerate([
                ("Stimulus 1", rr['resp1']), ("Stimulus 2", rr['resp2'])]):
            ax = fig2.add_subplot(inner[row])
            if resp.shape[0] > 0:
                mean = resp.mean(axis=0)
                std  = resp.std(axis=0)
                x    = np.arange(resp.shape[1])
                ax.plot(x, mean, color=reg_colors[col % 2], lw=1.5)
                ax.fill_between(x, mean - std, mean + std,
                                color=reg_colors[col % 2], alpha=0.25)
            ax.axvline(stim_onset_idx, color="gray", lw=0.8, ls="--")
            ax.axhline(0, color="gray", lw=0.5, ls=":")
            ax.set_title(f"{reg_name} — {stim_label}", fontsize=9)
            ax.set_ylabel("z-score")
            ax.set_xlabel("Time (s, stim onset = 0)")
            ax.set_xticks(tick_frames, tick_labels)
            ax.spines[["top", "right"]].set_visible(False)

    fig2.suptitle("Sub-region mean z-score responses", fontweight="bold")
    fig2.savefig(Path(results_dir) / "region_traces.png", dpi=150, bbox_inches="tight")

    plt.show()


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
