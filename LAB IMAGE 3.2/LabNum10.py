import librosa
import numpy as np
import matplotlib.pyplot as plt
y, sr = librosa.load("voice.wav", sr=None)

frame_length = 1024
hop_length = 256

rms = librosa.feature.rms(y=y, frame_length=frame_length,
                          hop_length=hop_length)[0]
zcr = librosa.feature.zero_crossing_rate(
    y, frame_length=frame_length, hop_length=hop_length)[0]

times = librosa.frames_to_time(
    np.arange(len(rms)), sr=sr, hop_length=hop_length)

regions = []
for energy, crossing in zip(rms, zcr):
    if energy < 0.02:
        regions.append(0)       # Silence
    elif crossing < 0.15:
        regions.append(1)       # Voiced
    else:
        regions.append(2)       # Unvoiced

plt.figure(figsize=(12, 5))
plt.plot(times, rms, label="Speech Energy")

colors = {0: "gray", 1: "green", 2: "red"}
labels = {0: "Silence", 1: "Voiced", 2: "Unvoiced"}

for i, region in enumerate(regions):
    plt.axvspan(times[i], times[i] + hop_length / sr,
                color=colors[region], alpha=0.25)

plt.xlabel("Time (seconds)")
plt.ylabel("Energy")
plt.title("Voiced, Unvoiced and Silence Region Separation")
plt.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.5,
                  label=labels[k]) for k in colors
])
plt.show()