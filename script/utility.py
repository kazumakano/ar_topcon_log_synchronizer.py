import csv
import math
import os.path as path
import pickle
from datetime import datetime, timedelta
from typing import Literal, Optional
import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def _datetime2unix(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(object)

    t: datetime
    for i, t in enumerate(ts):
        ts[i] = t.timestamp()

    return ts.astype(np.float64)

def adjust_freq(inertial_ts: np.ndarray, topcon_ts: np.ndarray, topcon_pos: np.ndarray, topcon_height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if inertial_ts[0] < topcon_ts[0] or inertial_ts[-1] > topcon_ts[-1]:
        raise Exception(f"range of topcon log must cover range of inertial sensor log but ({topcon_ts[0]}, {topcon_ts[-1]}) and ({inertial_ts[0]}, {inertial_ts[-1]}) were given")

    inertial_ts, topcon_ts = _datetime2unix(inertial_ts), _datetime2unix(topcon_ts)

    resampled_pos_x = interp1d(topcon_ts, topcon_pos[:, 0])(inertial_ts)
    resampled_pos_y = interp1d(topcon_ts, topcon_pos[:, 1])(inertial_ts)
    resampled_height = interp1d(topcon_ts, topcon_height)(inertial_ts)

    return np.hstack((resampled_pos_x[:, np.newaxis], resampled_pos_y[:, np.newaxis])), resampled_height

def adjust_ts_offset(inertial_jump_idxes: np.ndarray, inertial_ts: np.ndarray, topcon_jump_idxes: np.ndarray, topcon_ts: np.ndarray, use_jump_idxes: Literal["both", "former", "latter"] = "both") -> np.ndarray:
    match use_jump_idxes:
        case "both":
            return topcon_ts + (inertial_ts[inertial_jump_idxes] - topcon_ts[topcon_jump_idxes]).mean()
        case "former":
            return topcon_ts + (inertial_ts[inertial_jump_idxes[0]] - topcon_ts[topcon_jump_idxes[0]])
        case "latter":
            return topcon_ts + (inertial_ts[inertial_jump_idxes[1]] - topcon_ts[topcon_jump_idxes[1]])

def cut_with_padding(ar: np.ndarray, cut_idxes: np.ndarray, padding: int) -> np.ndarray:
    return ar[cut_idxes[0] + padding:cut_idxes[1] - padding + 1]

def export_log(ts: np.ndarray, inertial_val: np.ndarray, pos: np.ndarray, height: np.ndarray, file_name: str) -> None:
    with open(path.join(path.dirname(__file__), "../synced/", file_name + ".csv"), mode="w", newline="") as f:
        writer = csv.writer(f)
        t: datetime
        for i, t in enumerate(ts):
            writer.writerow((t.strftime("%Y-%m-%d %H:%M:%S.%f"), *inertial_val[i], *pos[i], height[i]))
    print(f"written to {file_name}.csv")

    with open(path.join(path.dirname(__file__), "../synced/", file_name + ".pkl"), mode="wb") as f:
        pickle.dump((ts, inertial_val, pos, height), f)
    print(f"written to {file_name}.pkl")

def _find_separated_max_2_idxes(ar: np.ndarray, min_interval: int) -> np.ndarray:
    max_idxes = np.empty(2, dtype=int)
    for i, j in enumerate(reversed(np.argsort(ar))):
        if i == 0:
            max_idxes[0] = j
        elif abs(j - max_idxes[0]) > min_interval:
            max_idxes[1] = j
            break

    max_idxes.sort()

    return max_idxes

def find_jump_in_inertial(ts: np.ndarray, acc: np.ndarray, min_interval: int) -> np.ndarray:
    acc_norm = np.linalg.norm(acc, axis=1)
    jump_idxes = _find_separated_max_2_idxes(acc_norm, min_interval)

    plt.figure(figsize=(16, 4))
    plt.plot(ts, acc_norm)
    plt.scatter(ts[jump_idxes], acc_norm[jump_idxes], c="tab:orange")
    plt.xlabel("time")
    plt.ylabel("acceleration norm")
    plt.show()

    return jump_idxes

def find_jump_in_topcon(ts: np.ndarray, pos: np.ndarray, height: np.ndarray, min_interval: int, begin: Optional[datetime] = None, end: Optional[datetime] = None) -> np.ndarray:
    if begin is None:
        begin = ts[0]
        begin_idx = 0
    else:
        for i, t in enumerate(ts):
            if t > begin:
                begin_idx = i
                break
    if end is None:
        end = ts[-1]
        end_idx = len(ts)
    else:
        for i, t in enumerate(ts[begin_idx + 1:]):
            if t > end:
                end_idx = begin_idx + 1 + i
                break
    jump_idxes = begin_idx + _find_separated_max_2_idxes(height[begin_idx:end_idx], min_interval)

    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(16, 12))
    for i in range(3):
        axes[i].set_xlim(left=begin, right=end)
    axes[0].plot(ts, pos[:, 0])
    axes[0].vlines(ts[jump_idxes], pos[:, 0].min(), pos[:, 0].max(), colors="tab:orange")
    axes[0].set_ylabel("position x")
    axes[1].plot(ts, pos[:, 1])
    axes[1].vlines(ts[jump_idxes], pos[:, 1].min(), pos[:, 1].max(), colors="tab:orange")
    axes[1].set_ylabel("position y")
    axes[2].plot(ts, height)
    axes[2].scatter(ts[jump_idxes], height[jump_idxes], c="tab:orange")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("height")
    fig.show()

    return jump_idxes

def load_inertial_log(file: str) -> tuple[np.ndarray, np.ndarray]:
    with open(file, mode="rb") as f:
        ts, val = pickle.load(f)

    print(f"{path.basename(file)} has been loaded")

    return ts, val

def load_topcon_log(file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, pos, height = [], [], []

    with open(file) as f:
        for row in csv.reader(f):
            ts.append(datetime.strptime(f"{row[4]} {row[5]}", "%Y-%m-%d %H:%M:%S"))
            pos.append([float(row[1]), float(row[2])])
            height.append(float(row[3]))

    print(f"{path.basename(file)} has been loaded")

    return np.array(ts, dtype=datetime), np.array(pos, dtype=np.float32), np.array(height, dtype=np.float32)

def make_ts_unique(ts: np.ndarray, pos: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_ts, unique_pos, unique_height = [], [], []

    last_ts = None
    for i, t in enumerate(ts):
        if last_ts is None:
            last_idx = i
            last_ts = t

        elif t != last_ts:
            for j in range(i - last_idx):
                unique_ts.append(last_ts + timedelta(seconds=(j + 1) / (i - last_idx + 1)))
                unique_pos.append(pos[last_idx + j])
                unique_height.append(height[last_idx + j])

            last_idx = i
            last_ts = t

    for j in range(i + 1 - last_idx):
        unique_ts.append(last_ts + timedelta(seconds=(j + 1) / (i + 1 - last_idx + 1)))
        unique_pos.append(pos[last_idx + j])
        unique_height.append(height[last_idx + j])

    return np.array(unique_ts, dtype=datetime), np.array(unique_pos, dtype=np.float32), np.array(unique_height, dtype=np.float32)

def plot(ts: np.ndarray, acc: np.ndarray, pos: np.ndarray) -> None:
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(16, 12))
    axes[0].plot(ts, np.linalg.norm(acc, axis=1))
    axes[0].set_ylabel("acceleration norm")
    axes[1].plot(ts, pos[:, 0])
    axes[1].set_ylabel("position x")
    axes[2].plot(ts, pos[:, 1])
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("position y")
    fig.show()

def rot(angle: float, pos: np.ndarray) -> np.ndarray:
    angle = math.radians(angle)
    return np.dot(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))), pos.T).T

def vis_traj(pos: np.ndarray, ref_direct_tan: Optional[float] = None) -> None:
    plt.axis("equal")
    if ref_direct_tan is not None:
        ref_direct_y_range = np.array((pos[:, 1].min() - 1, pos[:, 1].max() + 1), dtype=np.float32)
        plt.plot(ref_direct_y_range / ref_direct_tan, ref_direct_y_range, c="tab:orange")
    plt.scatter(pos[1:, 0], pos[1:, 1], s=1, marker=".")
    plt.scatter(pos[0, 0], pos[0, 1])
    plt.xlabel("position x")
    plt.ylabel("position y")
    plt.show()

def write_conf(file_name: str, padding: int, rot_angle: int) -> None:
    with open(path.join(path.dirname(__file__), "../synced/", file_name + ".yaml"), mode="w") as f:
        yaml.safe_dump({
            "padding": padding,
            "rot_angle": rot_angle
        }, f)
    
    print(f"written to {file_name}.yaml")
