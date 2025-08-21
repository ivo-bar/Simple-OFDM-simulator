# OFDM Simulator

This project is a C++ simulator for an Orthogonal Frequency-Division Multiplexing (OFDM) communication system. It models the entire pipeline from data generation and modulation to transmission over a simulated channel with noise, and finally demodulation and bit error rate calculation.

## Features

- Supports multiple modulation schemes: QPSK, 16QAM, 64QAM, and 256QAM.
- Configurable parameters such as FFT size, cyclic prefix length, and Signal-to-Noise Ratio (SNR).
- Simulates a multi-path fading channel.
- Generates constellation plots for both transmitted and received signals using Gnuplot.
- Calculates and reports the Bit Error Rate (BER) to evaluate performance.

## Prerequisites

- A C++ compiler that supports C++17 (like `g++`).
- `make`
- `gnuplot` (for visualizing the constellation diagrams).

## Building

To build the simulator, run the `make` command from the project's root directory.

```sh
make
```

This will compile [`ofdm_sym.cpp`](ofdm_sym.cpp) and create an executable file named `ofdm_sym`.

## Running the Simulator

You can run the executable with various command-line options to control the simulation parameters.

```sh
./ofdm_sym [options]
```

### Command-Line Options

- `-m <modulation>`: Set the modulation scheme (QPSK, 16QAM, 64QAM, 256QAM). Default is `16QAM`.
- `-l <length>`: Set the number of bits to transmit. Default is `2000`.
- `-n <points>`: Set the FFT size (must be a power of 2). Default is `256`.
- `-u <length>`: Set the cyclic prefix length. Default is `8`.
- `-snr <value>`: Set the Signal-to-Noise Ratio in dB. Default is `30`.
- `-h` or `--help`: Display the help message.

### Example

```sh
./ofdm_sym -m 64QAM -l 10000 -n 512 -u 16 -snr 20
```

## Output Files

The simulation generates the following files:

- `input_constellation.dat`: The constellation points of the transmitted signal.
- `output_constellation.dat`: The constellation points of the received signal after equalization.
- `plot_constellation.plt`: A Gnuplot script to generate the plots.
- `constellation_comparison.pdf`: A PDF file containing the side-by-side constellation plots.