from scipy.io import wavfile
import numpy as np
import librosa
from scipy import signal
import plotly
import plotly.graph_objs as go
from sklearn import preprocessing

nfft = 64
window_length = 0.03

rate, frames = wavfile.read("dataset/wet/train_wet.wav")

window = round(window_length * rate)

i = 5000
f, t, S = signal.spectrogram(frames[i:i + window - 1],
                             rate,
                             nperseg=nfft,
                             noverlap=round(nfft/2),
                             scaling='spectrum')

S = preprocessing.scale(S)

plotly.offline.plot([dict(x=t, y=f, z=S, type='surface')],
    filename='surface.html')

mfccs = librosa.feature.mfcc(frames[i:i + window - 1], sr=rate, n_fft=nfft, hop_length=round(nfft/2))
mfccs = preprocessing.scale(mfccs)
print(mfccs.shape)

plotly.offline.plot([dict(z=mfccs, type='surface')],
    filename='mfcc.html')



data = [
    go.Heatmap(
        z=S,
        x=t,
        y=f,
        colorscale='Viridis',
    )
]

layout = go.Layout(
    title='Heatmap'
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='heatmap.html')

print("Number of frequencies:", len(f))
print("Number of time bins:", len(t))
'''
plt.pcolormesh(t, f, S)
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")

plt.show()
'''