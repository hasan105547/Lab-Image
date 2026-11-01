# lab 13
# Write a MATLAB/Python program and plot multilevel speech resolution. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read speech audio
fs, audio = wavfile.read("voice.wav")

# Convert stereo to mono if needed
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Normalize audio
audio = audio.astype(float)
audio = audio / np.max(np.abs(audio))

# Quantization levels
levels = [2, 4, 8, 16]

plt.figure(figsize=(12, 8))

# Original speech signal
plt.subplot(5, 1, 1)
plt.plot(audio)
plt.title("Original Speech Signal")
plt.ylabel("Amplitude")

# Multilevel quantization
for i, L in enumerate(levels):
    quantized = np.round((audio + 1) * (L - 1) / 2)
    quantized = quantized * 2 / (L - 1) - 1

    plt.subplot(5, 1, i + 2)
    plt.plot(quantized)
    plt.title(f"Speech Signal with {L} Quantization Levels")
    plt.ylabel("Amplitude")

plt.xlabel("Sample Number")
plt.tight_layout()
plt.show()