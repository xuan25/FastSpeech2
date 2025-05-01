import csv
import itertools
import os
from typing import Iterator, List, Tuple
import librosa
import numpy as np
from scipy import spatial
import tqdm
# from .evaluate_mcd import calculate

import pysptk
import soundfile as sf
from fastdtw import fastdtw
import pyworld as pw

import multiprocessing as mp

def world_extract(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> np.ndarray:
    """Extract World-based acoustic features.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Minimum f0 value (default=40).
        f0 (int): Maximum f0 value (default=800).
        n_shift (int): Shift length in point (default=256).
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).

    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0

def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")

def compute_log_f0(
    tgt_path: str,
    ref_path: str,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 1024,
    n_shift: int = 256,
    mcep_dim: int = None,
    mcep_alpha: float = None,
) -> float:
    """Calculate MCD."""

    # load wav file as float64
    tgt_x, tgt_r = sf.read(tgt_path, dtype="float64")
    ref_x, ref_r = sf.read(ref_path, dtype="float64")

    samplingRate = tgt_r
    if tgt_r != ref_r:
        ref_x = librosa.resample(ref_x.astype(np.float64), orig_sr=ref_r, target_sr=tgt_r)

    tgt_mcep, tgt_f0 = world_extract(
        x=tgt_x,
        fs=samplingRate,
        f0min=f0min,
        f0max=f0max,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )
    ref_mcep, ref_f0 = world_extract(
        x=ref_x,
        fs=samplingRate,
        f0min=f0min,
        f0max=f0max,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )

    # DTW
    _, path = fastdtw(tgt_mcep, ref_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    tgt_f0_dtw = tgt_f0[twf[0]]
    ref_f0_dtw = ref_f0[twf[1]]

    # Get voiced part
    nonzero_idxs = np.where((tgt_f0_dtw != 0) & (ref_f0_dtw != 0))[0]
    tgt_f0_dtw_voiced = np.log(tgt_f0_dtw[nonzero_idxs])
    ref_f0_dtw_voiced = np.log(ref_f0_dtw[nonzero_idxs])

    # log F0 RMSE
    log_f0_rmse = np.sqrt(np.mean((tgt_f0_dtw_voiced - ref_f0_dtw_voiced) ** 2))

    return log_f0_rmse

def _thread_wrapper(obj):
    return obj[0](*obj[1:])

class LogF0BatchProcesser(Iterator):

    def __init__(
            self,
            tgt_paths: str,
            ref_paths: str,
            f0min: int = 40,
            f0max: int= 800,
            n_fft: int = 1024,
            n_shift: int = 256,
            mcep_dim: int = None,
            mcep_alpha: float = None,
            num_threads: int = 32,
        ):
        self.tgt_paths = tgt_paths
        self.ref_paths = ref_paths
        self.f0min = f0min
        self.f0max = f0max
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha
        self.num_threads = num_threads
  
    def __iter__(self):
        p_pool = mp.Pool(self.num_threads)
        self.it = p_pool.imap(_thread_wrapper, zip(
            itertools.repeat(compute_log_f0), 
            self.tgt_paths, 
            self.ref_paths,
            itertools.repeat(self.f0min),
            itertools.repeat(self.f0max),
            itertools.repeat(self.n_fft),
            itertools.repeat(self.n_shift),
            itertools.repeat(self.mcep_dim),
            itertools.repeat(self.mcep_alpha)
        ))
        return self

    def __next__(self) -> float:
        return next(self.it)

def main():

    tgt_paths = [
        "data/hyp1/LJ022-0023.wav",
        "data/hyp2/LJ022-0023.wav",
    ]

    ref_paths = [
        "data/ref/LJ022-0023.wav",
        "data/ref/LJ022-0023.wav",
    ]
    
    # compute single MCD

    mcd = compute_log_f0(tgt_paths[0], ref_paths[0],
            n_fft=1024,
            n_shift=256,
            mcep_dim=None,
            mcep_alpha=None,
        )
    print(f"Log-F0 1: {mcd:.3f}")

    mcd = compute_log_f0(tgt_paths[1], ref_paths[1],
            n_fft=1024,
            n_shift=256,
            mcep_dim=None,
            mcep_alpha=None,
        )
    print(f"Log-F0 2: {mcd:.3f}")

    # batch computing MCD (default hyperparameters)

    processer = MCDBatchProcesser(ref_paths, tgt_paths, num_threads=2)

    for tgt_path, mcd in tqdm.tqdm(
        zip(
            tgt_paths,
            processer
        ),
        desc="[computing]", dynamic_ncols=True, total=len(tgt_paths)):

        tqdm.tqdm.write(f"{tgt_path} Log-F0: {mcd:.3f}")

if __name__ == "__main__":
    main()
