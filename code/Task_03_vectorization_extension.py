"""Task 3 extension - audio signal processing: apply a gain and clip a noisy sine wave.

New case: 5 seconds of a 440 Hz tone sampled at 44.1 kHz (220,500 samples) with noise.
We amplify it by 3x, hard-clip at +/-1 (like an overloaded speaker) and compute RMS,
all with vectorized expressions.
"""
import os
import numpy as np
import timeit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

fs = 44_100
t = np.arange(0, 5, 1 / fs)
rng = np.random.default_rng(1)
signal = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.02 * rng.standard_normal(t.size)

gain = 3.0
amplified = gain * signal
clipped = np.clip(amplified, -1.0, 1.0)

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def clip_loop(x):
    out = []
    for v in x:
        v = gain * v
        out.append(max(-1.0, min(1.0, v)))
    return out

print(f"Number of samples: {t.size}")
print(f"RMS original: {rms(signal):.4f}")
print(f"RMS after gain: {rms(amplified):.4f}")
print(f"RMS after clipping: {rms(clipped):.4f}")
print(f"Share of samples clipped: {np.mean(np.abs(amplified) > 1):.2%}")

lt = timeit.timeit(lambda: clip_loop(signal), number=1)
vt = timeit.timeit(lambda: np.clip(gain * signal, -1, 1), number=20) / 20
print(f"Loop: {lt:.4f}s, vectorized: {vt:.6f}s, speed-up {lt/vt:.0f}x")

n = 441  # first 10 ms
plt.plot(t[:n] * 1000, signal[:n], label="Original")
plt.plot(t[:n] * 1000, clipped[:n], label="Gain x3 + clipping")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("Task 3 extension: gain and clipping of a 440 Hz tone")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(FIG, "task03_ext_audio_clipping.png"), dpi=150, bbox_inches="tight")
