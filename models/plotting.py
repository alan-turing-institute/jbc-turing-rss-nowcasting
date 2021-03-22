import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

import numpy as np
import pandas as pd


def plot_posterior(
    ax,
    posterior_means,
    obs=None,
    posterior_confidence_intervals=None,
    title=None,
    legend=False,
    posterior_color="xkcd:orange",
    posterior_label="Posterior mean",
    confidence_interval_color="xkcd:sky blue",
    confidence_interval_label=None,
    obs_label=None,
    tex_plot=True,
    start_idx=0,
    alpha=0.2
):
    """
    Plotting function for pretty plotting of posterior distributions. Arguments are self-explanatory if not listed.

    Args:
        - ax: (matplotlib axis object) axis on which to plot
        - obs: (np.array) sequence of best-available count values [shape (n_time_steps,)]
        - posterior_means: (np.array) posterior means [shape (n_time_steps,)]
        - posterior_confidence_intervals: (np.array) posterior confidence intervals. column 1/2 lower/upper. [shape (n_time_steps, 2)]
    """

    if tex_plot:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

    support = np.arange(len(posterior_means)) + start_idx
    ax.plot(support, posterior_means, "r", label=posterior_label, color=posterior_color)

    if not obs is None:
        ax.plot(support[:-3], obs[:-3], label=obs_label, color="xkcd:denim blue")
        ax.plot(support[-4:], obs[-4:], "--", color="xkcd:denim blue")

    if posterior_confidence_intervals is not None:
        upper, lower = (
            posterior_confidence_intervals[:, 0],
            posterior_confidence_intervals[:, 1],
        )
        ax.fill_between(
            support,
            posterior_means,
            upper,
            alpha=alpha,
            color=confidence_interval_color,
            label=confidence_interval_label,
        )
        ax.fill_between(
            support, posterior_means, lower, alpha=alpha, color=confidence_interval_color
        )

    if title is not None:
        ax.set_title(title, size=16)

    if legend:
        ax.legend(loc=0, fontsize=16)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
