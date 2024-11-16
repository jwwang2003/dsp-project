#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import modulation as mod

x = np.linspace(0, 10, 10)
mods = mod.GetModulation("64QAM", 10, False)
plt.plot(x, mods[0][0])
plt.plot(x, mods[0][1])
plt.show()

# %%
from scipy.signal import firwin, lfilter

def generate_64qam_cap(num_symbols, symbol_rate, sampling_rate):
    # 64QAM constellation points
    levels = [-7, -5, -3, -1, 1, 3, 5, 7]
    rng = np.random.default_rng()
    symbols_I = rng.choice(levels, num_symbols)
    symbols_Q = rng.choice(levels, num_symbols)

    # Generate 64QAM baseband signal
    qam_symbols = symbols_I + 1j * symbols_Q

    # CAP modulation pulse shaping (orthogonal cosine and sine filters)
    roll_off = 0.2
    num_taps = 101  # Filter length
    ts = 1 / symbol_rate  # Symbol duration
    t = np.arange(-num_taps//2, num_taps//2) / sampling_rate

    # Cosine and sine shaping filters
    cos_filter = firwin(num_taps, roll_off / ts, window='hamming', fs=sampling_rate)
    sin_filter = firwin(num_taps, roll_off / ts, window='hamming', fs=sampling_rate)

    # Interpolate symbols to match sampling rate
    samples_per_symbol = int(sampling_rate / symbol_rate)
    upsampled_I = np.zeros(num_symbols * samples_per_symbol)
    upsampled_Q = np.zeros(num_symbols * samples_per_symbol)
    upsampled_I[::samples_per_symbol] = symbols_I
    upsampled_Q[::samples_per_symbol] = symbols_Q

    # Apply pulse shaping
    shaped_I = lfilter(cos_filter, 1.0, upsampled_I)
    shaped_Q = lfilter(sin_filter, 1.0, upsampled_Q
                       )
    # Combine I and Q components
    cap_signal = shaped_I - shaped_Q  # CAP combines cosine and sine shaping

    return cap_signal, qam_symbols, (shaped_I, shaped_Q)

num_symbols = 1000  # Number of 64QAM symbols
symbol_rate = 1000  # Symbol rate in symbols per second
sampling_rate = 10000  # Sampling rate in Hz

cap_signal, qam_symbols, (shaped_I, shaped_Q) = generate_64qam_cap(num_symbols, symbol_rate, sampling_rate)

# Plot the CAP signal
import matplotlib.pyplot as plt

time = np.arange(len(cap_signal)) / sampling_rate
plt.figure()
plt.plot(time, cap_signal)
plt.title("64QAM-CAP Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# %%
import numpy as np
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt

def generate_64qam_symbols(num_symbols):
    """Generate 64QAM symbols."""
    levels = [-7, -5, -3, -1, 1, 3, 5, 7]
    rng = np.random.default_rng()
    symbols_I = rng.choice(levels, num_symbols)
    symbols_Q = rng.choice(levels, num_symbols)
    return symbols_I, symbols_Q

def cap_modulation(symbols_I, symbols_Q, symbol_rate, sampling_rate, num_taps=101):
    """Apply CAP modulation to 64QAM symbols."""
    samples_per_symbol = int(sampling_rate / symbol_rate)

    # Define CAP pulse shaping filters (cosine and sine filters)
    cos_filter = firwin(num_taps, 0.25, window='hamming', fs=sampling_rate)
    sin_filter = firwin(num_taps, 0.25, window='hamming', fs=sampling_rate)

    # Upsample symbols
    upsampled_I = np.zeros(len(symbols_I) * samples_per_symbol)
    upsampled_Q = np.zeros(len(symbols_Q) * samples_per_symbol)
    upsampled_I[::samples_per_symbol] = symbols_I
    upsampled_Q[::samples_per_symbol] = symbols_Q

    # Apply pulse shaping
    shaped_I = lfilter(cos_filter, 1.0, upsampled_I)
    shaped_Q = lfilter(sin_filter, 1.0, upsampled_Q)

    # Combine I and Q to form the CAP-modulated signal
    cap_signal = shaped_I - shaped_Q
    return cap_signal, shaped_I, shaped_Q, samples_per_symbol

def cap_demodulation(cap_signal, samples_per_symbol, sampling_rate, num_taps=101):
    """Demodulate CAP signal to recover 64QAM symbols."""
    # Define the matched filters (same as the modulation filters)
    cos_filter = firwin(num_taps, 0.25, window='hamming', fs=sampling_rate)
    sin_filter = firwin(num_taps, 0.25, window='hamming', fs=sampling_rate)

    # Apply matched filtering
    demod_I = lfilter(cos_filter, 1.0, cap_signal)
    demod_Q = -lfilter(sin_filter, 1.0, cap_signal)

    # Downsample to recover the original symbols (ensure alignment)
    start_index = num_taps // 2  # Center the sampling
    recovered_I = demod_I[start_index::samples_per_symbol]
    recovered_Q = demod_Q[start_index::samples_per_symbol]

    # Normalize recovered symbols to match original levels
    # recovered_I = np.round(recovered_I).astype(int)
    # recovered_Q = np.round(recovered_Q).astype(int)

    return recovered_I, recovered_Q

# Parameters
num_symbols = 100
symbol_rate = 1000  # Symbols per second
sampling_rate = 10000  # Samples per second

# Step 1: Generate 64QAM symbols
symbols_I, symbols_Q = generate_64qam_symbols(num_symbols)

# Step 2: Apply CAP modulation
cap_signal, shaped_I, shaped_Q, samples_per_symbol = cap_modulation(
    symbols_I, symbols_Q, symbol_rate, sampling_rate
)

# Step 3: Demodulate the CAP signal
recovered_I, recovered_Q = cap_demodulation(cap_signal, samples_per_symbol, sampling_rate)

# print(len(symbols_I), len(recovered_I))
# # Validate Results
# assert len(recovered_I) == len(symbols_I), "Mismatch in recovered I length!"
# assert len(recovered_Q) == len(symbols_Q), "Mismatch in recovered Q length!"
# assert np.allclose(symbols_I, recovered_I), "Mismatch in I symbols!"
# assert np.allclose(symbols_Q, recovered_Q), "Mismatch in Q symbols!"

# Plotting
time = np.arange(len(cap_signal)) / sampling_rate

# Original 64QAM Symbols
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.stem(np.arange(len(symbols_I[:20])) / symbol_rate, symbols_I[:20])
plt.title("Original I Component (64QAM)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.stem(np.arange(len(symbols_Q[:20])) / symbol_rate, symbols_Q[:20])
plt.title("Original Q Component (64QAM)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# CAP-modulated Signal
plt.figure(figsize=(8, 4))
plt.plot(time[:1000], cap_signal[:1000])
plt.title("CAP-modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Recovered 64QAM Symbols
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.stem(np.arange(len(recovered_I[:20])) / symbol_rate, recovered_I[:20])
plt.title("Recovered I Component")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.stem(np.arange(len(recovered_Q[:20])) / symbol_rate, recovered_Q[:20])
plt.title("Recovered Q Component")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Compare Symbols
mismatches_I = np.sum(symbols_I != recovered_I)
mismatches_Q = np.sum(symbols_Q != recovered_Q)
total_mismatches = mismatches_I + mismatches_Q
error_rate = total_mismatches / (2 * num_symbols)  # 2 components (I and Q)

# Print Comparison Results
print("Comparison of Original and Recovered Symbols:")
print(f"Total Symbols: {2 * num_symbols}")
print(f"Mismatches in I: {mismatches_I}")
print(f"Mismatches in Q: {mismatches_Q}")
print(f"Total Mismatches: {total_mismatches}")
print(f"Error Rate: {error_rate:.2%}")

# %%
