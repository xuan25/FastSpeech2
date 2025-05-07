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

import multiprocessing as mp

def sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Sampling rate
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
        is_padding (bool): Whether to pad the end of signal (default=False).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).

    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")

    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1

    # get window function
    win = pysptk.sptk.hamming(n_fft)

    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calculate spectrogram
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)

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

def compute_mcd(
    tgt_path: str,
    ref_path: str,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> float:
    """Calculate MCD."""

    # load wav file as float64
    tgt_x, tgt_r = sf.read(tgt_path, dtype="float64")
    ref_x, ref_r = sf.read(ref_path, dtype="float64")

    samplingRate = tgt_r
    if tgt_r != ref_r:
        ref_x = librosa.resample(ref_x.astype(np.float64), orig_sr=ref_r, target_sr=tgt_r)

    # extract ground truth and converted features
    tgt_mcep = sptk_extract(
        x=tgt_x,
        fs=samplingRate,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )
    ref_mcep = sptk_extract(
        x=ref_x,
        fs=samplingRate,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )

    # DTW
    _, path = fastdtw(tgt_mcep, ref_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    tgt_mcep_dtw = tgt_mcep[twf[0]]
    ref_mcep_dtw = ref_mcep[twf[1]]

    # MCD
    diff2sum = np.sum((tgt_mcep_dtw - ref_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    return mcd

def _thread_wrapper(obj):
    return obj[0](*obj[1:])

class MCDBatchProcesser(Iterator):

    def __init__(
            self,
            tgt_paths: list[str],
            ref_paths: list[str],
            n_fft: int = 512,
            n_shift: int = 256,
            mcep_dim: int = 25,
            mcep_alpha: float = 0.41,
            num_threads: int = 32,
        ):
        self.tgt_paths = tgt_paths
        self.ref_paths = ref_paths
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha
        self.num_threads = num_threads
  
    def __iter__(self):
        p_pool = mp.Pool(self.num_threads)
        self.it = p_pool.imap(_thread_wrapper, zip(
            itertools.repeat(compute_mcd), 
            self.tgt_paths, 
            self.ref_paths,
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

    mcd = compute_mcd(tgt_paths[0], ref_paths[0],
            n_fft=1024,
            n_shift=256,
        )
    print(f"MCD 1: {mcd:.3f}")

    mcd = compute_mcd(tgt_paths[1], ref_paths[1],
            n_fft=1024,
            n_shift=256,
        )
    print(f"MCD 2: {mcd:.3f}")

    # batch computing MCD (default hyperparameters)

    processer = MCDBatchProcesser(ref_paths, tgt_paths, num_threads=2)

    for tgt_path, mcd in tqdm.tqdm(
        zip(
            tgt_paths,
            processer
        ),
        desc="[computing]", dynamic_ncols=True, total=len(tgt_paths)):

        tqdm.tqdm.write(f"{tgt_path} MCD: {mcd:.3f}")

if __name__ == "__main__":
    main()
