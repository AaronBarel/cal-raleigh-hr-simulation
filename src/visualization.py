# src/visualization.py
"""Plotting helpers for the Cal Raleigh HR simulation.

All of the simulation plots share the same conventions:

* Distributions of simulated season totals are drawn as *probability* mass
  functions (one bar per integer HR total, bar height = P(HR == k)) rather
  than raw sample counts, so the y-axis is readable independently of how many
  posterior draws happened to be taken.
* Cal Raleigh's realised 2025 total is drawn as a labelled reference line in
  ink, not as a series colour, because it is an observed outcome and not
  another model.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns

# Categorical slots 1-3 of the chart palette, on a light surface.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
INK = '#0b0b0b'
INK_MUTED = '#52514e'


def plot_hr_pmf(series, title, actual=None, xlabel='Total Home Runs',
                ax=None, figsize=(12, 7), kde=True):
    """Plots simulated HR totals as probability mass functions.

    Args:
        series (dict): Maps a label to a 1-D array of simulated HR totals.
        title (str): Plot title.
        actual (int, optional): Cal Raleigh's realised total, drawn as a
            labelled vertical reference line.
        xlabel (str): Label for the x-axis.
        ax (matplotlib.axes.Axes, optional): Axes to draw on.
        figsize (tuple): Figure size used when ``ax`` is not supplied.
        kde (bool): Overlay a smoothed density on each histogram.

    Returns:
        matplotlib.axes.Axes: The axes that were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Solid bars read best for one distribution; two overlapping fills turn to
    # mud, so multiple series are drawn as outlined steps with a light fill.
    overlaid = len(series) > 1
    style = (dict(element='step', fill=True, alpha=0.30, linewidth=2)
             if overlaid else
             dict(alpha=0.75, edgecolor='white', linewidth=1))

    for (label, samples), color in zip(series.items(), SERIES_COLORS):
        sns.histplot(samples, discrete=True, stat='probability', kde=kde,
                     color=color, label=label, ax=ax, **style)

    if actual is not None:
        _actual_line(ax, actual)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability')
    # A lone distribution is named by the title; anything more needs a legend.
    if overlaid or actual is not None:
        ax.legend()
    return ax


def plot_actual_vs_predictions(predictions, actual, title=None,
                               ax=None, figsize=None):
    """Compares each model's projection against the realised HR total.

    Draws one row per model showing its 95% credible interval and median
    projection, with the actual total as a vertical reference line and a
    right-hand column of direct labels.

    Args:
        predictions (dict): Maps a model label to a 1-D array of simulated
            season HR totals.
        actual (int): The realised season HR total.
        title (str, optional): Plot title.
        ax (matplotlib.axes.Axes, optional): Axes to draw on.
        figsize (tuple, optional): Figure size; defaults to a height that
            scales with the number of models.

    Returns:
        matplotlib.axes.Axes: The axes that were drawn on.
    """
    labels = list(predictions)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (12, 1.0 * len(labels) + 1.5))
        # Reserve the right-hand strip for the direct-label column.
        fig.subplots_adjust(left=0.26, right=0.68, top=0.86, bottom=0.14)

    # Top-to-bottom reading order.
    positions = np.arange(len(labels))[::-1]

    summaries = {}
    for label, y in zip(labels, positions):
        samples = np.asarray(predictions[label])
        low, median, high = np.percentile(samples, [2.5, 50, 97.5])
        summaries[label] = (low, median, high)

        ax.hlines(y, low, high, color=SERIES_COLORS[0], linewidth=7,
                  alpha=0.35, zorder=2)
        ax.plot([median], [y], marker='o', markersize=10,
                color=SERIES_COLORS[0], markeredgecolor='white',
                markeredgewidth=2, zorder=3)

    _actual_line(ax, actual, direct_label=True)

    # Keep the x-scale tight to the data; the label column lives outside the
    # axes on the right so that gridlines never run through the text.
    data_left = min(min(s[0] for s in summaries.values()), actual)
    data_right = max(max(s[2] for s in summaries.values()), actual)
    pad = 0.08 * (data_right - data_left)
    ax.set_xlim(data_left - pad, data_right + pad)

    # x in axes fraction, y in data coordinates.
    label_transform = mtransforms.blended_transform_factory(
        ax.transAxes, ax.transData)
    for label, y in zip(labels, positions):
        low, median, high = summaries[label]
        covers = low <= actual <= high
        verdict = 'covers actual' if covers else f'off by {abs(actual - median):.0f}'
        ax.text(1.03, y,
                f'median {median:.0f}   95% CI {low:.0f}-{high:.0f}   {verdict}',
                transform=label_transform, va='center', fontsize=9,
                color=INK_MUTED, clip_on=False)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Season Home Run Total')
    ax.set_ylabel('')
    ax.set_ylim(-0.55, len(labels) - 0.45)
    if title:
        ax.set_title(title, pad=24)
    return ax


def _actual_line(ax, actual, direct_label=False):
    """Draws the realised total as a labelled ink reference line.

    The label rides on the line itself when ``direct_label`` is set, and is
    carried by the legend otherwise.
    """
    text = f'Actual 2025 total: {actual} HR'
    ax.axvline(actual, color=INK, linestyle='--', linewidth=2.5, zorder=4,
               label=None if direct_label else text)
    if direct_label:
        ax.annotate(text, xy=(actual, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color=INK)
