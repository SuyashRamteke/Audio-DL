import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

FIG_SIZE = (15,10)

file = "blues.00000.wav"

# Load audio file
signal, sample_rate = librosa.load(file, sr = 22050)

plt.plot(figsize = FIG_SIZE)
librosa.display.waveplot(signal, sample_rate, alpha = 0.4)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform")

# FFT Power Spectrum
fft = np.fft.fft(signal)

spectrum = np.abs(fft)

f = np.linspace(0, sample_rate, len(spectrum))

# take half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

# plot spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")

# Perform STFT
hop_length = 512
n_fft = 2048
stft = librosa.stft(signal, n_fft = n_fft, hop_length = hop_length)
spectrogram = np.abs(stft)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")

# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")

# Compute MFCC
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

# show plots
plt.show()