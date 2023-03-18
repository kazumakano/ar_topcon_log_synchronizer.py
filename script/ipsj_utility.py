import math
import os.path as path
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from . import utility as util


def _scs2gcs(acc: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return Rotation(quat).apply(acc)

def create_acc_distrib_figure(inertial_val_dict: dict[str, np.ndarray], pos_dict: dict[str, np.ndarray], result_file_name: Optional[str] = None) -> None:
    plt.rcParams["axes.edgecolor"] = "gray"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(ncols=3, sharex=True, figsize=(12, 4), dpi=1200)
    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, wspace=0.1)

    major_ticks = [round(v, 1) for v in np.arange(-0.4, 0.41, 0.2)]
    minor_ticks = [round(v, 1) for v in np.arange(-0.4, 0.41, 0.1)]
    for i, k in enumerate(inertial_val_dict.keys()):
        axes[i].axis("equal")
        axes[i].set_xlim(left=-0.5, right=0.5)
        axes[i].set_ylim(bottom=-0.5, top=0.5)

        direct = math.degrees(math.atan(np.polyfit(pos_dict[k][:, 0], pos_dict[k][:, 1], 1)[0]))
        if pos_dict[k][-1, 0] - pos_dict[k][0, 0] < 0:
            direct += 180
        hori_acc = util.rot(45 - direct, _scs2gcs(inertial_val_dict[k][:, :3], inertial_val_dict[k][:, 12:])[:, :2])

        axes[i].grid(color="gray", linewidth=0.2)
        axes[i].scatter(hori_acc[:, 0], hori_acc[:, 1], s=2, marker=".", )
        axes[i].arrow(0, 0, 0.25, 0.25, head_width=0.015, color="tab:orange")
        axes[i].set_title(f"({('a', 'b', 'c')[i]}) {k.capitalize()}", y=-0.25)
        axes[i].set_xlabel("X component of acceleration [G]")
        axes[i].set_xticks(ticks=major_ticks, labels=major_ticks, fontsize=8)
        axes[i].set_xticks(ticks=minor_ticks)
        if i == 0:
            axes[i].set_ylabel("Y component of acceleration [G]")
            axes[0].set_yticks(ticks=major_ticks, labels=major_ticks, fontsize=8)
            axes[i].set_yticks(ticks=minor_ticks)
        else:
            axes[i].set_yticks(ticks=minor_ticks, labels=())
        axes[i].tick_params(color="gray", length=2, width=0.8)

    if result_file_name is not None:
        fig.savefig(path.join(path.dirname(__file__), "../result/", result_file_name + ".eps"))
        fig.savefig(path.join(path.dirname(__file__), "../result/", result_file_name + ".png"))
    fig.show()
