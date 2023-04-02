import math
import os.path as path
from typing import Iterable, Optional
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from . import utility as util


def _ipsj_rcparams(func):
    def decorated_func(*args, **kwargs):
        plt.rcParams["axes.edgecolor"] = "gray"
        plt.rcParams["axes.linewidth"] = 0.8

        return func(*args, **kwargs)

    return decorated_func

def _scs2gcs(acc: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return Rotation(quat).apply(acc)

@_ipsj_rcparams
def create_acc_distrib_figure(inertial_val_dict: dict[str, np.ndarray], pos_dict: dict[str, np.ndarray], enable_grid: bool = False, result_file_name: Optional[str] = None) -> None:
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

        if enable_grid:
            axes[i].grid(color="gray", linewidth=0.1)
        axes[i].scatter(hori_acc[:, 0], hori_acc[:, 1], s=2, marker=".")
        axes[i].arrow(0, 0, 0.25, 0.25, head_width=0.015, color="tab:orange")
        axes[i].set_title(f"({('a', 'b', 'c')[i]}) {k.capitalize()}", y=-0.3)
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

def _interp_ticks(major_ticks: Iterable) -> np.ndarray:
    minor_ticks = np.empty(2 * len(major_ticks) - 1, dtype=np.float32)

    minor_ticks[::2] = major_ticks
    for i in range(1, len(minor_ticks), 2):
        minor_ticks[i] = (minor_ticks[i - 1] + minor_ticks[i + 1]) / 2

    return minor_ticks

@_ipsj_rcparams
def create_course_figure(pos_dict: dict[str, np.ndarray], quat_dict: dict[str, np.ndarray], ticks_dict: Optional[dict[str, tuple[Iterable, Iterable]]] = None, result_file_name: Optional[str] = None) -> None:
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4), dpi=1200)
    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, wspace=0.15)

    for i, k in enumerate(pos_dict.keys()):
        direct = np.deg2rad(util._quat2direct(quat_dict[k][::400]))
        axes[i].axis("equal")
        axes[i].scatter(pos_dict[k][:, 0], pos_dict[k][:, 1], s=1, c=np.arange(len(pos_dict[k])), marker=".")
        axes[i].quiver(pos_dict[k][::400, 0], pos_dict[k][::400, 1], np.cos(direct), np.sin(direct), np.arange(len(pos_dict[k]), step=400), scale=32, width=0.004)
        axes[i].set_title(f"({('a', 'b', 'c')[i]}) {k.capitalize()}", y=-0.3)
        axes[i].set_xlabel("Position [m]")
        if ticks_dict is not None:
            axes[i].set_xticks(ticks=ticks_dict[k][0], labels=ticks_dict[k][0], fontsize=8)
            axes[i].set_xticks(ticks=_interp_ticks(ticks_dict[k][0]))
            axes[i].set_yticks(ticks=ticks_dict[k][1], labels=ticks_dict[k][1], fontsize=8)
            axes[i].set_yticks(ticks=_interp_ticks(ticks_dict[k][1]))
        axes[i].tick_params(color="gray", length=2, width=0.8)
    axes[0].set_ylabel("Position [m]")

    if result_file_name is not None:
        fig.savefig(path.join(path.dirname(__file__), "../result/", result_file_name + ".pdf"))
        fig.savefig(path.join(path.dirname(__file__), "../result/", result_file_name + ".png"))
    fig.show()

@_ipsj_rcparams
def create_sync_figure(acc: np.ndarray, height: np.ndarray, inertial_jump_idxes: np.ndarray, topcon_jump_idxes: np.ndarray, acc_ticks: Optional[tuple[Iterable, Iterable]] = None, height_ticks: Optional[tuple[Iterable, Iterable]] = None, result_file_name: Optional[str] = None) -> None:
    acc_norm = np.linalg.norm(acc, axis=1)

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    fig, axes = plt.subplots(nrows=2, figsize=(6, 4), dpi=1200)
    fig.align_ylabels()

    acc_margin = 0.15 * (inertial_jump_idxes[1] - inertial_jump_idxes[0])
    acc_lim = (inertial_jump_idxes[0] - acc_margin, inertial_jump_idxes[1] + acc_margin)
    axes[0].set_xlim(acc_lim)
    height_margin = 0.15 * (topcon_jump_idxes[1] - topcon_jump_idxes[0])
    height_lim = (topcon_jump_idxes[0] - height_margin, topcon_jump_idxes[1] + height_margin)
    axes[1].set_xlim(height_lim)

    acc_lim = (max(math.floor(acc_lim[0]), 0), min(math.ceil(acc_lim[1]), len(acc_norm)))
    axes[0].plot(range(*acc_lim), acc_norm[acc_lim[0]:acc_lim[1]], linewidth=0.8)
    axes[0].scatter(inertial_jump_idxes, acc_norm[inertial_jump_idxes], s=16, c="tab:orange")
    axes[0].set_ylabel("Acceleration norm [G]")
    if acc_ticks is not None:
        axes[0].set_xticks(ticks=acc_ticks[0], labels=acc_ticks[0], fontsize=8)
        axes[0].set_xticks(ticks=_interp_ticks(acc_ticks[0]))
        axes[0].set_yticks(ticks=acc_ticks[1], labels=acc_ticks[1], fontsize=8)
        # axes[0].set_yticks(ticks=_interp_ticks(acc_ticks[1]))
    axes[0].tick_params(color="gray", length=2, width=0.8)
    height_lim = (max(math.floor(height_lim[0]), 0), min(math.ceil(height_lim[1]), len(height)))
    axes[1].plot(range(*height_lim), height[height_lim[0]:height_lim[1]], linewidth=0.8)
    axes[1].scatter(topcon_jump_idxes, height[topcon_jump_idxes], s=16, c="tab:orange")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Height [m]")
    if height_ticks is not None:
        axes[1].set_xticks(ticks=height_ticks[0], labels=height_ticks[0], fontsize=8)
        # axes[1].set_xticks(ticks=_interp_ticks(height_ticks[0]))
        axes[1].set_yticks(ticks=height_ticks[1], labels=height_ticks[1], fontsize=8)
        # axes[1].set_yticks(ticks=_interp_ticks(height_ticks[1]))
    axes[1].tick_params(color="gray", length=2, width=0.8)

    if result_file_name is not None:
        fig.savefig(path.join(path.dirname(__file__), "../result/", result_file_name + ".eps"))
        fig.savefig(path.join(path.dirname(__file__), "../result/", result_file_name + ".png"))
    fig.show()
