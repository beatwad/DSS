import math
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil


@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)


def pad_if_needed(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)


def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int, 
                      max_step: int, duration: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = list(set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist()))
    # negative_positions = np.zeros(num_steps)
    
    # # negative position must not be too close to positive position
    # for p in positive_positions:
    #     negative_positions[max(p - duration // 2, 0) : min(p + duration // 2, num_steps)] = 1

    # negative_positions = list(np.where(negative_positions == 0)[0])

    # # if can't find far enough negative samples - just select any
    # if len(negative_positions) == 0:
    negative_positions = np.zeros(num_steps)
    negative_positions[positive_positions] = 1
    negative_positions = list(np.where(negative_positions == 0)[0])
    
    # do not consider negative samples from the end of the strange series
    # negative_positions = [n for n in negative_positions if n < max_step]

    pos = random.sample(negative_positions, 1)[0]

    return pos


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label

def add_gaussian_sleep(label, offset=20):
    idxs = np.where(np.diff(label) == 1)
    
    for idx in idxs:
        if len(idx) > 0:
            idx = idx[0]

            z = gaussian_kernel(offset*2, sigma=offset//2)[:offset*2]
            
            if idx-offset < 0:
                z = z[offset-idx:]
            elif idx+offset > label.shape[0]:
                z = z[:label.shape[0]-idx-offset]

            label[max(idx-offset, 0):min(idx+offset, label.shape[0])] = z

    idxs = np.where(np.diff(label) == -1)
    
    for idx in idxs:
        if len(idx) > 0:
            idx = idx[0]

            z = gaussian_kernel(offset*2, sigma=offset//2)[offset*2+1:]
            
            if idx-offset < 0:
                z = z[offset-idx:]
            elif idx+offset > label.shape[0]:
                z = z[:label.shape[0]-idx-offset]

            label[max(idx-offset, 0):min(idx+offset, label.shape[0])] = z

    return label
