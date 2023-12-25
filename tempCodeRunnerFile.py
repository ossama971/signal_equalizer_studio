from scipy.io import wavfile
import numpy as np

def get_frequency_range(wav_file):
    sample_rate, data = wavfile.read(wav_file)
    
    if len(data.shape) > 1:  # If the WAV file has multiple channels, take the first channel
        data = data[:, 0]

    duration = len(data) / sample_rate
    frequencies = np.fft.fft(data)
    magnitudes = np.abs(frequencies)
    freqs = np.fft.fftfreq(len(frequencies))

    # Find the index corresponding to the maximum magnitude
    max_magnitude_index = np.argmax(magnitudes)
    max_freq = freqs[max_magnitude_index]
    max_freq_hz = abs(max_freq * sample_rate)

    # Find the minimum and maximum frequencies
    min_freq_hz = abs(freqs[1] * sample_rate)  # Excluding 0 Hz
    max_freq_hz = max_freq_hz if max_freq_hz != 0 else sample_rate / 2  # Nyquist frequency

    return min_freq_hz, max_freq_hz, duration

# Replace 'your_file.wav' with the path to your WAV file
file_path = 'filtered_3_PICCOLO.wav'
min_freq, max_freq, duration = get_frequency_range(file_path)

print(f"Frequency Range in {file_path}:")
print(f"Minimum Frequency: {min_freq:.2f} Hz")
print(f"Maximum Frequency: {max_freq:.2f} Hz")
print(f"Duration: {duration:.2f} seconds")
