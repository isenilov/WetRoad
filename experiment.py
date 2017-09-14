from scipy.io import wavfile
import librosa
from scipy import signal
import plotly
from sklearn import preprocessing

nfft = 64
window_length = 0.3

rate, frames = wavfile.read("dataset/wet/train_wet.wav")
rate2, frames2 = wavfile.read("dataset/wet/chevy_wet.wav")

window = round(window_length * rate)

i = 1000000
f, t, S = signal.spectrogram(frames[i:i + window - 1],
                             rate,
                             nperseg=nfft,
                             noverlap=round(nfft/2),
                             scaling='spectrum')

S = preprocessing.scale(S)
S = S[:12]
f = f[:12]

# plotly.offline.plot([dict(x=t, y=f, z=S, type='surface')], filename='surface.html')
plotly.offline.plot([dict(x=t, y=f, z=S, type='heatmap')], filename='heatmap.html')

f2, t2, S2 = signal.spectrogram(frames2[i:i + window - 1],
                             rate2,
                             nperseg=nfft,
                             noverlap=round(nfft/2),
                             scaling='spectrum')

S2 = preprocessing.scale(S2)
S2 = S2[:12]
f2 = f2[:12]

plotly.offline.plot([dict(x=t2, y=f2, z=S2, type='heatmap')], filename='heatmap2.html')


mfccs = librosa.feature.mfcc(frames[i:i + window - 1], sr=rate, n_fft=nfft, hop_length=round(nfft/2), n_mfcc=26)
mfccs = preprocessing.scale(mfccs)

plotly.offline.plot([dict(z=mfccs, type='heatmap')], filename='mfcc.html')

mfccs2 = librosa.feature.mfcc(frames2[i:i + window - 1], sr=rate2, n_fft=nfft, hop_length=round(nfft/2), n_mfcc=26)
mfccs2 = preprocessing.scale(mfccs2)

plotly.offline.plot([dict(z=mfccs2, type='heatmap')], filename='mfcc2.html')

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