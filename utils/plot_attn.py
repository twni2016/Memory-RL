import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import logger


def plot_attnmaps(attns: np.ndarray, rew: np.ndarray, idx: str):
    layers, heads, seq_len, attn_span = attns.shape
    for layer in range(layers):
        for head in range(heads):
            plot_attnmap(attns[layer, head], rew, idx + f"_l{layer+1}h{head+1}")


def plot_attnmap(attn: np.ndarray, rew: np.ndarray, idx: str):
    """
    attn: (T, inference_attn_span)
    rew: (T)
    """

    def construct_attnmap(attn):
        seq_len, attn_span = attn.shape
        attnmap = np.zeros((seq_len, seq_len))
        mask = np.ones((seq_len, seq_len), dtype=np.bool)  # by default, not show

        # part1: first attn_span * attn_span elements
        len1 = min(attn_span, seq_len)
        attnmap[:len1, :len1] = attn[:len1, :len1]

        mask[np.tril_indices_from(np.eye(len1))] = False
        if attn_span >= seq_len:
            return attnmap, mask

        # part2: rest (T-attn_span) * attn_span elements
        xs = np.arange(attn_span, seq_len)
        xs = np.repeat(xs[:, None], axis=-1, repeats=attn_span).flatten()

        ys = np.arange(attn_span)
        ys = np.repeat(ys[None, :], axis=0, repeats=seq_len - attn_span)
        ys += np.arange(1, seq_len - attn_span + 1)[:, None]
        ys = ys.flatten()

        attnmap[xs, ys] = attn[attn_span:].flatten()
        mask[xs, ys] = False

        return attnmap, mask

    attnmap, mask = construct_attnmap(attn)

    seq_length = rew.shape[-1]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw=dict(height_ratios=[seq_length, 1], hspace=0.0),
    )
    # set the height ratios, and leave no space between subfigures

    axes = axes.flatten()

    sns.heatmap(
        rew[None, :],
        ax=axes[1],
        cmap="Reds",
        cbar=True,
        yticklabels=False,
    )  # a col vec
    axes[1].collections[0].colorbar.set_ticks([])

    # map 2d matrix to df: the columns are minus 1 to align with previous hidden states
    # then transpose
    attndf = pd.DataFrame(
        attnmap, columns=[str(t - 1) for t in range(0, seq_length)]
    ).transpose()
    sns.heatmap(
        attndf,
        ax=axes[0],
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        xticklabels=False,
        mask=mask.transpose(),
    )

    axes[1].set_xlabel("Time Step")
    axes[0].set_ylabel("Attended Time Step")
    axes[0].invert_yaxis()  # make 0 starts from bottom left in y-axis
    axes[0].set_title("Self-Attention Heatmap")

    plt.savefig(
        os.path.join(logger.get_dir(), "plt", f"{idx}.png"),
        dpi=max(200, seq_length),  # 200
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()
