import numpy as np
import yaml
from typing import Union, Dict
from scipy.signal import firwin2, filtfilt, lfilter
import sounddevice as sd


def _to_cfg_dict(cfg_or_path: Union[Dict, str]) -> Dict:
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    with open(cfg_or_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def play_and_record(signal: np.ndarray, fs: int, channels: int = 1) -> np.ndarray:
    rec = sd.playrec(signal, samplerate=fs, channels=channels)
    sd.wait()
    return rec

def custom_band_weighting_mag(
    freqs_hz: np.ndarray,
    low_boost_db: float = 6.0,
    mid_cut_db: float = -6.0,
    high_boost_db: float = 6.0,
    f_low1: float = 20.0,
    f_high1: float = 600.0,
    f_low2: float = 600.0,
    f_high2: float = 1000.0,
    f_low3: float = 1000.0,
    f_high3: float = 10000.0,
) -> np.ndarray:
    f = np.asarray(freqs_hz, dtype=float)
    mag_db = np.zeros_like(f, dtype=float)
    m1 = (f >= f_low1) & (f < f_high1)
    m2 = (f >= f_low2) & (f < f_high2)
    m3 = (f >= f_low3) & (f <= f_high3)
    mag_db[m1] += low_boost_db
    mag_db[m2] += mid_cut_db
    mag_db[m3] += high_boost_db
    mag = 10.0 ** (mag_db / 20.0)
    return np.maximum(mag, 1e-12)


def design_custom_weighting_fir(
    fs: float,
    low_boost_db: float = 6.0,
    mid_cut_db: float = -6.0,
    high_boost_db: float = 6.0,
    f_low1: float = 20.0,
    f_high1: float = 600.0,
    f_low2: float = 600.0,
    f_high2: float = 1000.0,
    f_low3: float = 1000.0,
    f_high3: float = 10000.0,
    numtaps: int = 2049,
) -> np.ndarray:
    f = np.linspace(0.0, fs / 2.0, numtaps)
    f_low1 = max(0.0, f_low1)
    f_high1 = min(f_high1, fs / 2.0)
    f_low2 = max(0.0, f_low2)
    f_high2 = min(f_high2, fs / 2.0)
    f_low3 = max(0.0, f_low3)
    f_high3 = min(f_high3, fs / 2.0)
    mag = custom_band_weighting_mag(
        f,
        low_boost_db=low_boost_db,
        mid_cut_db=mid_cut_db,
        high_boost_db=high_boost_db,
        f_low1=f_low1,
        f_high1=f_high1,
        f_low2=f_low2,
        f_high2=f_high2,
        f_low3=f_low3,
        f_high3=f_high3,
    )
    b = firwin2(numtaps, f / (fs / 2.0), mag, window="hamming")
    return b


def apply_custom_weighting(
    signal: np.ndarray,
    fs: float,
    low_boost_db: float = 6.0,
    mid_cut_db: float = -6.0,
    high_boost_db: float = 6.0,
    f_low1: float = 20.0,
    f_high1: float = 600.0,
    f_low2: float = 600.0,
    f_high2: float = 1000.0,
    f_low3: float = 1000.0,
    f_high3: float = 10000.0,
    zero_phase: bool = True,
) -> np.ndarray:
    b = design_custom_weighting_fir(
        fs,
        low_boost_db=low_boost_db,
        mid_cut_db=mid_cut_db,
        high_boost_db=high_boost_db,
        f_low1=f_low1,
        f_high1=f_high1,
        f_low2=f_low2,
        f_high2=f_high2,
        f_low3=f_low3,
        f_high3=f_high3,
    )
    a = np.array([1.0])
    return filtfilt(b, a, signal) if zero_phase else lfilter(b, a, signal)


def generate_log_chirp(f0: float, f1: float, duration: float, fs: float, amplitude: float = 0.7):
    N = int(fs * duration)
    t = np.linspace(0.0, duration, N)
    L = duration
    R = np.log(f1 / f0)
    sweep = amplitude * np.sin(2 * np.pi * f0 * L / R * (np.exp(t * R / L) - 1.0))
    return sweep.astype(float), t

########################################################################################################################
#main1: 根据配置文件过滤信号
def filter_signal_with_cfg(signal: np.ndarray, fs: float, cfg: Union[Dict, str]) -> np.ndarray:
    c = _to_cfg_dict(cfg)
    return apply_custom_weighting(
        signal,
        fs,
        low_boost_db=c.get("low_boost_db", 0.0),
        mid_cut_db=c.get("mid_cut_db", 0.0),
        high_boost_db=c.get("high_boost_db", 0.0),
        f_low1=c.get("f_low1", 20.0),
        f_high1=c.get("f_high1", 600.0),
        f_low2=c.get("f_low2", 600.0),
        f_high2=c.get("f_high2", 1000.0),
        f_low3=c.get("f_low3", 1000.0),
        f_high3=c.get("f_high3", 10000.0),
        zero_phase=True,
    )
########################################################################################################################



