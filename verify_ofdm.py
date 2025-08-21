import numpy as np

# INput vector
X = np.array([1-1j, 3+1j, 3-1j, -1+1j])

# perform normalized IFFT
X_ifft = np.fft.ifft(X, norm='ortho')
print("Input Vector:", X)
print("Normalized IFFT Result:", X_ifft)

# Add cyclic prefix
cyclic_prefix_length = 2
cyclic_prefix = X_ifft[-cyclic_prefix_length:]
X_with_cyclic_prefix = np.concatenate((cyclic_prefix, X_ifft))

# channel is [1,2]
hn = np.array([1, 2])

# Convolve the signal with the channel
X_with_cyclic_prefix = np.convolve(X_with_cyclic_prefix, hn, mode='full')

# Perform FFT on the channel impulse response, using length 4
hn = np.pad(hn, (0, 2), 'constant')  # Pad to length 4
Hn = 2*np.fft.fft(hn, norm='ortho')

print("Channel Impulse Response (Hn):", Hn)

print("Output of the channel:", X_with_cyclic_prefix)
# Perform FFT to recover the original signal

# First, remove cyclic prefix
X_without_cyclic_prefix = X_with_cyclic_prefix[cyclic_prefix_length:-1]
print("Signal without Cyclic Prefix:", X_without_cyclic_prefix) 
Rn = np.fft.fft(X_without_cyclic_prefix, norm='ortho')
print("Recovered Signal (FFT):", Rn)

# reconstruct the original signal: Xr = Rn/ Hn
Xr = Rn / Hn
print("Reconstructed Signal:", Xr)