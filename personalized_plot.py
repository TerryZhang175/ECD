from __future__ import annotations

import matplotlib.pyplot as plt

import personalized_config as cfg

try:
    from unidec.IsoDec.plots import cplot, on_scroll
except Exception:
    cplot = None
    on_scroll = None


def plot_overlay(
    experimental,
    overlays,
    mz_min=None,
    mz_max=None,
    noise_cutoff=None,
) -> None:
    """
    overlays: list of (isodist, color, label_text)
    """
    xmin = float(mz_min) if mz_min is not None else float(min(experimental[:, 0]))
    xmax = float(mz_max) if mz_max is not None else float(max(experimental[:, 0]))

    if noise_cutoff is not None:
        exp = experimental[experimental[:, 1] >= float(noise_cutoff)]
    else:
        exp = experimental

    if cplot is not None:
        cplot(exp, color="k", factor=1)
        for dist, color, _ in overlays:
            cplot(dist, color=color, factor=-1)
        plt.hlines(0, xmin, xmax, color="k", linewidth=0.8)
        if cfg.LABEL_MATCHES:
            for dist, color, label in overlays:
                peak_idx = int(dist[:, 1].argmax())
                plt.text(float(dist[peak_idx, 0]), float(dist[peak_idx, 1]), label, fontsize=8, color=color)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        max_pos = float(max(exp[:, 1])) if len(exp) else 0.0
        max_neg = float(max([max(d[:, 1]) for d, _, _ in overlays])) if overlays else 0.0
        max_y = max(max_pos, max_neg)
        plt.ylim(-1.1 * max_y, 1.1 * max_y)
        if on_scroll is not None:
            plt.connect("scroll_event", on_scroll)
    else:
        plt.plot(exp[:, 0], exp[:, 1], color="k", linewidth=0.8, label="Experimental")
        for dist, color, label in overlays:
            plt.vlines(dist[:, 0], 0, dist[:, 1], color=color, linewidth=1.0, label=label)

    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()
