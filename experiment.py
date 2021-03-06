from scipy.io import wavfile
import librosa
import numpy as np
from scipy import signal
from plotly.offline import plot
import plotly.graph_objs as go
from plotly import tools
from sklearn import preprocessing


nfft = 64
window_length = 0.1

rate, frames = wavfile.read("dataset/dry3/audio_mono.wav")
rate2, frames2 = wavfile.read("dataset/wet3/audio_mono.wav")
rate3, frames3 = wavfile.read("dataset/dry/chevy_dry.wav")
rate4, frames4 = wavfile.read("dataset/wet/chevy_wet.wav")

window = round(window_length * rate)

i = 12965400

f, t, S = signal.spectrogram(frames[i:i + window - 1],
                             rate,
                             nperseg=nfft,
                             noverlap=round(nfft/2),
                             scaling='spectrum')

S = preprocessing.scale(S)
S = S[:12]
f = f[:12]

# plot([dict(x=t, y=f, z=S, type='heatmap')], filename='heatmap.html')

f2, t2, S2 = signal.spectrogram(frames2[i:i + window - 1],
                             rate2,
                             nperseg=nfft,
                             noverlap=round(nfft/2),
                             scaling='spectrum')

S2 = preprocessing.scale(S2)
S2 = S2[:12]
f2 = f2[:12]

# plot([dict(x=t2, y=f2, z=S2, type='heatmap')], filename='heatmap2.html')


mfccs = librosa.feature.mfcc(frames[i:i + window - 1],
                               sr=rate,
                               n_fft=nfft,
                               hop_length=round(nfft / 2),
                               fmax=8000)
mfccs = preprocessing.scale(mfccs)


mfccs2 = librosa.feature.mfcc(frames2[i:i + window - 1],
                               sr=rate2,
                               n_fft=nfft,
                               hop_length=round(nfft / 2),
                               fmax=8000)
mfccs2 = preprocessing.scale(mfccs2)

mfccs3 = librosa.feature.mfcc(frames3[i:i + window - 1],
                               sr=rate3,
                               n_fft=nfft,
                               hop_length=round(nfft / 2),
                               fmax=8000)
mfccs3 = preprocessing.scale(mfccs3)

mfccs4 = librosa.feature.mfcc(frames4[i:i + window - 1],
                               sr=rate4,
                               n_fft=nfft,
                               hop_length=round(nfft / 2),
                               fmax=8000)
mfccs4 = preprocessing.scale(mfccs4)

fig = tools.make_subplots(rows=2, cols=2)
fig.append_trace(go.Heatmap(z=mfccs, zauto=False, zmin=-5, zmax=4), 1, 1)
fig.append_trace(go.Heatmap(z=mfccs2, zauto=False, zmin=-5, zmax=4), 1, 2)
fig.append_trace(go.Heatmap(z=mfccs3, zauto=False, zmin=-5, zmax=4), 2, 1)
fig.append_trace(go.Heatmap(z=mfccs4, zauto=False, zmin=-5, zmax=4), 2, 2)
plot(fig, filename="mfcc4")

# plot([go.Heatmap(z=mfccs, zauto=False, zmin=-5, zmax=3)], filename='mfcc')
# plot([go.Heatmap(z=mfccs2, zauto=False, zmin=-5, zmax=3)], filename='mfcc2')
# plot([go.Heatmap(z=mfccs3, zauto=False, zmin=-5, zmax=3)], filename='mfcc3')
# plot([go.Heatmap(z=mfccs4, zauto=False, zmin=-5, zmax=3)], filename='mfcc4')

'''
print("Shape of the MFCCs:", mfccs.shape)
print("Shape of the spectrogram:", S.shape)
print("Number of frequencies:", len(f))
print("Number of time bins:", len(t))
print("Frequencies:", f)
print("Time bins:", t)


plt.pcolormesh(t, f, S)
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")

plt.show()
'''