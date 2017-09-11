from scipy.io import wavfile
import librosa
from scipy import signal
import plotly
from sklearn import preprocessing

nfft = 64
window_length = 0.1

rate, frames = wavfile.read("dataset/wet/train_wet.wav")

window = round(window_length * rate)

i = 5000
f, t, S = signal.spectrogram(frames[i:i + window - 1],
                             rate,
                             nperseg=nfft,
                             noverlap=round(nfft/2),
                             scaling='spectrum')

S = preprocessing.scale(S)
S = S[:12]

plotly.offline.plot([dict(x=t, y=f, z=S, type='surface')], filename='surface.html')
plotly.offline.plot([dict(x=t, y=f, z=S, type='heatmap')], filename='heatmap.html')


mfccs = librosa.feature.mfcc(frames[i:i + window - 1], sr=rate, n_fft=nfft, hop_length=round(nfft/2), n_mfcc=12)
mfccs = preprocessing.scale(mfccs)

plotly.offline.plot([dict(z=mfccs, type='heatmap')], filename='mfcc.html')

print("Shape of the MFCCs:", mfccs.shape)
print("Shape of the spectrogram:", S.shape)
print("Number of frequencies:", len(f))
print("Number of time bins:", len(t))
print("Frequencies:", f)
print("Time bins:", t)

'''
plt.pcolormesh(t, f, S)
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")

plt.show()
'''