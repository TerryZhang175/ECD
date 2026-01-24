from __future__ import annotations

import numpy as np

import personalized_config as cfg

_zoom_levels = []

def _plt():
    import matplotlib.pyplot as plt
    return plt


def on_scroll(event):
    plt = _plt()
    if event.inaxes is None:
        return
    ax = event.inaxes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    mouse_x = event.xdata
    mouse_y = event.ydata
    zoom_factor = 0.7

    if event.button == "up":
        _zoom_levels.append((xlim, ylim))
        new_x_range = (xlim[1] - xlim[0]) * (1 - zoom_factor)
        new_y_range = (ylim[1] - ylim[0]) * (1 - zoom_factor)
        ax.set_xlim([mouse_x - new_x_range / 2, mouse_x + new_x_range / 2])
        ax.set_ylim([mouse_y - new_y_range / 2, mouse_y + new_y_range / 2])
    elif event.button == "down" and _zoom_levels:
        last_xlim, last_ylim = _zoom_levels.pop()
        ax.set_xlim(last_xlim)
        ax.set_ylim(last_ylim)
    plt.draw()


def _fast_vlines(centroids, color, base, factor):
    plt = _plt()
    xpairs = np.transpose([centroids[:, 0], centroids[:, 0]])
    ypairs = np.transpose([base + np.zeros(len(centroids)), base + factor * centroids[:, 1]])
    xlist = []
    ylist = []
    for xends, yends in zip(xpairs, ypairs):
        xlist.extend(xends)
        xlist.append(None)
        ylist.extend(yends)
        ylist.append(None)
    plt.plot(xlist, ylist, color=color, linewidth=0.8)


def cplot(centroids, color="r", factor=1, base=0, **_):
    if centroids is None or len(centroids) == 0:
        return
    _fast_vlines(centroids, color=color, base=base, factor=factor)


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
        plt = _plt()
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
        plt = _plt()
        plt.plot(exp[:, 0], exp[:, 1], color="k", linewidth=0.8, label="Experimental")
        for dist, color, label in overlays:
            plt.vlines(dist[:, 0], 0, dist[:, 1], color=color, linewidth=1.0, label=label)

    plt = _plt()
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()
