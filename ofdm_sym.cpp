#include <iostream>
#include <string>
#include <unordered_map>
#include <iomanip>
#include <locale>
#include <cmath>
#include <vector>
#include <sstream>
#include <complex>
#include <algorithm>
#include <random>
#include <ctime>
#include <limits>
#include <fstream>
#include <cstring>

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generate QPSK mapping table
const std::vector<std::complex<double>> QPSK_MAPPING = {
    {1.0f, 1.0f},   // 00
    {1.0f, -1.0f},  // 01
    {-1.0f, 1.0f},  // 10
    {-1.0f, -1.0f}  // 11
};

// Generate 16QAM mapping table
const std::vector<std::complex<double>> QAM16_MAPPING = {
    { 1.0,  1.0},  // 0000
    { 1.0,  3.0},  // 0001
    { 1.0, -1.0},  // 0010
    { 1.0, -3.0},  // 0011
    { 3.0,  1.0},  // 0100
    { 3.0,  3.0},  // 0101
    { 3.0, -1.0},  // 0110
    { 3.0, -3.0},  // 0111
    {-1.0,  1.0},  // 1000
    {-1.0,  3.0},  // 1001
    {-1.0, -1.0},  // 1010
    {-1.0, -3.0},  // 1011
    {-3.0,  1.0},  // 1100
    {-3.0,  3.0},  // 1101
    {-3.0, -1.0},  // 1110
    {-3.0, -3.0}   // 1111
};

// Generate 64QAM mapping table
const std::vector<std::complex<double>> QAM64_MAPPING = {
    {-7.0, -7.0}, {-7.0, -5.0}, {-7.0, -3.0}, {-7.0, -1.0}, {-7.0,  1.0}, {-7.0,  3.0}, {-7.0,  5.0}, {-7.0,  7.0},  // 000xxx
    {-5.0, -7.0}, {-5.0, -5.0}, {-5.0, -3.0}, {-5.0, -1.0}, {-5.0,  1.0}, {-5.0,  3.0}, {-5.0,  5.0}, {-5.0,  7.0},  // 001xxx
    {-3.0, -7.0}, {-3.0, -5.0}, {-3.0, -3.0}, {-3.0, -1.0}, {-3.0,  1.0}, {-3.0,  3.0}, {-3.0,  5.0}, {-3.0,  7.0},  // 010xxx
    {-1.0, -7.0}, {-1.0, -5.0}, {-1.0, -3.0}, {-1.0, -1.0}, {-1.0,  1.0}, {-1.0,  3.0}, {-1.0,  5.0}, {-1.0,  7.0},  // 011xxx
    { 1.0, -7.0}, { 1.0, -5.0}, { 1.0, -3.0}, { 1.0, -1.0}, { 1.0,  1.0}, { 1.0,  3.0}, { 1.0,  5.0}, { 1.0,  7.0},  // 100xxx
    { 3.0, -7.0}, { 3.0, -5.0}, { 3.0, -3.0}, { 3.0, -1.0}, { 3.0,  1.0}, { 3.0,  3.0}, { 3.0,  5.0}, { 3.0,  7.0},  // 101xxx
    { 5.0, -7.0}, { 5.0, -5.0}, { 5.0, -3.0}, { 5.0, -1.0}, { 5.0,  1.0}, { 5.0,  3.0}, { 5.0,  5.0}, { 5.0,  7.0},  // 110xxx
    { 7.0, -7.0}, { 7.0, -5.0}, { 7.0, -3.0}, { 7.0, -1.0}, { 7.0,  1.0}, { 7.0,  3.0}, { 7.0,  5.0}, { 7.0,  7.0}   // 111xxx
};

// Generate 256QAM Gray-encoded mapping table (16x16 grid)
const std::vector<std::complex<double>> QAM256_MAPPING = {
    {-15.0, -15.0}, {-15.0, -13.0}, {-15.0, -11.0}, {-15.0,  -9.0}, {-15.0,  -7.0}, {-15.0,  -5.0}, {-15.0,  -3.0}, {-15.0,  -1.0},
    {-15.0,   1.0}, {-15.0,   3.0}, {-15.0,   5.0}, {-15.0,   7.0}, {-15.0,   9.0}, {-15.0,  11.0}, {-15.0,  13.0}, {-15.0,  15.0},
    {-13.0, -15.0}, {-13.0, -13.0}, {-13.0, -11.0}, {-13.0,  -9.0}, {-13.0,  -7.0}, {-13.0,  -5.0}, {-13.0,  -3.0}, {-13.0,  -1.0},
    {-13.0,   1.0}, {-13.0,   3.0}, {-13.0,   5.0}, {-13.0,   7.0}, {-13.0,   9.0}, {-13.0,  11.0}, {-13.0,  13.0}, {-13.0,  15.0},
    {-11.0, -15.0}, {-11.0, -13.0}, {-11.0, -11.0}, {-11.0,  -9.0}, {-11.0,  -7.0}, {-11.0,  -5.0}, {-11.0,  -3.0}, {-11.0,  -1.0},
    {-11.0,   1.0}, {-11.0,   3.0}, {-11.0,   5.0}, {-11.0,   7.0}, {-11.0,   9.0}, {-11.0,  11.0}, {-11.0,  13.0}, {-11.0,  15.0},
    { -9.0, -15.0}, { -9.0, -13.0}, { -9.0, -11.0}, { -9.0,  -9.0}, { -9.0,  -7.0}, { -9.0,  -5.0}, { -9.0,  -3.0}, { -9.0,  -1.0},
    { -9.0,   1.0}, { -9.0,   3.0}, { -9.0,   5.0}, { -9.0,   7.0}, { -9.0,   9.0}, { -9.0,  11.0}, { -9.0,  13.0}, { -9.0,  15.0},
    { -7.0, -15.0}, { -7.0, -13.0}, { -7.0, -11.0}, { -7.0,  -9.0}, { -7.0,  -7.0}, { -7.0,  -5.0}, { -7.0,  -3.0}, { -7.0,  -1.0},
    { -7.0,   1.0}, { -7.0,   3.0}, { -7.0,   5.0}, { -7.0,   7.0}, { -7.0,   9.0}, { -7.0,  11.0}, { -7.0,  13.0}, { -7.0,  15.0},
    { -5.0, -15.0}, { -5.0, -13.0}, { -5.0, -11.0}, { -5.0,  -9.0}, { -5.0,  -7.0}, { -5.0,  -5.0}, { -5.0,  -3.0}, { -5.0,  -1.0},
    { -5.0,   1.0}, { -5.0,   3.0}, { -5.0,   5.0}, { -5.0,   7.0}, { -5.0,   9.0}, { -5.0,  11.0}, { -5.0,  13.0}, { -5.0,  15.0},
    { -3.0, -15.0}, { -3.0, -13.0}, { -3.0, -11.0}, { -3.0,  -9.0}, { -3.0,  -7.0}, { -3.0,  -5.0}, { -3.0,  -3.0}, { -3.0,  -1.0},
    { -3.0,   1.0}, { -3.0,   3.0}, { -3.0,   5.0}, { -3.0,   7.0}, { -3.0,   9.0}, { -3.0,  11.0}, { -3.0,  13.0}, { -3.0,  15.0},
    { -1.0, -15.0}, { -1.0, -13.0}, { -1.0, -11.0}, { -1.0,  -9.0}, { -1.0,  -7.0}, { -1.0,  -5.0}, { -1.0,  -3.0}, { -1.0,  -1.0},
    { -1.0,   1.0}, { -1.0,   3.0}, { -1.0,   5.0}, { -1.0,   7.0}, { -1.0,   9.0}, { -1.0,  11.0}, { -1.0,  13.0}, { -1.0,  15.0},
    {  1.0, -15.0}, {  1.0, -13.0}, {  1.0, -11.0}, {  1.0,  -9.0}, {  1.0,  -7.0}, {  1.0,  -5.0}, {  1.0,  -3.0}, {  1.0,  -1.0},
    {  1.0,   1.0}, {  1.0,   3.0}, {  1.0,   5.0}, {  1.0,   7.0}, {  1.0,   9.0}, {  1.0,  11.0}, {  1.0,  13.0}, {  1.0,  15.0},
    {  3.0, -15.0}, {  3.0, -13.0}, {  3.0, -11.0}, {  3.0,  -9.0}, {  3.0,  -7.0}, {  3.0,  -5.0}, {  3.0,  -3.0}, {  3.0,  -1.0},
    {  3.0,   1.0}, {  3.0,   3.0}, {  3.0,   5.0}, {  3.0,   7.0}, {  3.0,   9.0}, {  3.0,  11.0}, {  3.0,  13.0}, {  3.0,  15.0},
    {  5.0, -15.0}, {  5.0, -13.0}, {  5.0, -11.0}, {  5.0,  -9.0}, {  5.0,  -7.0}, {  5.0,  -5.0}, {  5.0,  -3.0}, {  5.0,  -1.0},
    {  5.0,   1.0}, {  5.0,   3.0}, {  5.0,   5.0}, {  5.0,   7.0}, {  5.0,   9.0}, {  5.0,  11.0}, {  5.0,  13.0}, {  5.0,  15.0},
    {  7.0, -15.0}, {  7.0, -13.0}, {  7.0, -11.0}, {  7.0,  -9.0}, {  7.0,  -7.0}, {  7.0,  -5.0}, {  7.0,  -3.0}, {  7.0,  -1.0},
    {  7.0,   1.0}, {  7.0,   3.0}, {  7.0,   5.0}, {  7.0,   7.0}, {  7.0,   9.0}, {  7.0,  11.0}, {  7.0,  13.0}, {  7.0,  15.0},
    {  9.0, -15.0}, {  9.0, -13.0}, {  9.0, -11.0}, {  9.0,  -9.0}, {  9.0,  -7.0}, {  9.0,  -5.0}, {  9.0,  -3.0}, {  9.0,  -1.0},
    {  9.0,   1.0}, {  9.0,   3.0}, {  9.0,   5.0}, {  9.0,   7.0}, {  9.0,   9.0}, {  9.0,  11.0}, {  9.0,  13.0}, {  9.0,  15.0},
    { 11.0, -15.0}, { 11.0, -13.0}, { 11.0, -11.0}, { 11.0,  -9.0}, { 11.0,  -7.0}, { 11.0,  -5.0}, { 11.0,  -3.0}, { 11.0,  -1.0},
    { 11.0,   1.0}, { 11.0,   3.0}, { 11.0,   5.0}, { 11.0,   7.0}, { 11.0,   9.0}, { 11.0,  11.0}, { 11.0,  13.0}, { 11.0,  15.0},
    { 13.0, -15.0}, { 13.0, -13.0}, { 13.0, -11.0}, { 13.0,  -9.0}, { 13.0,  -7.0}, { 13.0,  -5.0}, { 13.0,  -3.0}, { 13.0,  -1.0},
    { 13.0,   1.0}, { 13.0,   3.0}, { 13.0,   5.0}, { 13.0,   7.0}, { 13.0,   9.0}, { 13.0,  11.0}, { 13.0,  13.0}, { 13.0,  15.0},
    { 15.0, -15.0}, { 15.0, -13.0}, { 15.0, -11.0}, { 15.0,  -9.0}, { 15.0,  -7.0}, { 15.0,  -5.0}, { 15.0,  -3.0}, { 15.0,  -1.0},
    { 15.0,   1.0}, { 15.0,   3.0}, { 15.0,   5.0}, { 15.0,   7.0}, { 15.0,   9.0}, { 15.0,  11.0}, { 15.0,  13.0}, { 15.0,  15.0}
};

const std::unordered_map<std::string, int> MODULATION_TABLE = {
    { "QPSK", 4},
    {"16QAM", 16},
    {"64QAM", 64},
    {"256QAM", 256},
};
/* Helper Functions 

These functions are used to format complex numbers and print vectors of complex numbers.
They are not directly related to the OFDM simulator but are useful for debugging and displaying results.
*/

std::string formatComplex(const std::complex<double>& z) {
    std::ostringstream oss;
    oss << z.real();
    if (z.imag() >= 0)
        oss << " + " << z.imag() << "i";
    else
        oss << " - " << -z.imag() << "i";
    return oss.str();
}

void printComplexVector(const std::vector<std::complex<double>>& vec, const std::string& label = "") {
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << formatComplex(vec[i]);
        if (i != vec.size() - 1)
            std::cout << ",  ";
    }
    std::cout << std::endl;
}

void printComplexVectorLimited(const std::vector<std::complex<double>>& vec, const std::string& label = "", size_t maxElements = 10) {
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    size_t limit = std::min(maxElements, vec.size());
    for (size_t i = 0; i < limit; ++i) {
        std::cout << formatComplex(vec[i]);
        if (i != limit - 1)
            std::cout << ",  ";
    }
    if (vec.size() > maxElements) {
        std::cout << " ... (" << vec.size() << " total elements)";
    }
    std::cout << std::endl;
}

void printIntVector(const std::vector<int>& vec, const std::string& label = "") {
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1)
            std::cout << ",  ";
    }
    std::cout << std::endl;
}

// Signal Processing Utility Class
// Contains static methods for common DSP operations like FFT, IFFT, and convolution
class SignalProcessing {
public:
    // Function to perform FFT or IFFT
    // data: The input/output vector of complex numbers
    // N: The size of the data vector (must be a power of 2)
    // inverse: true for IFFT, false for FFT
    static void fft(std::vector<std::complex<double>>& data, int N, bool inverse) {
        if (N <= 1) {
            return;
        }

        // Split into even and odd parts
        std::vector<std::complex<double>> even(N / 2);
        std::vector<std::complex<double>> odd(N / 2);
        for (int i = 0; i < N / 2; ++i) {
            even[i] = data[2 * i];
            odd[i] = data[2 * i + 1];
        }

        // Recursive calls
        fft(even, N / 2, inverse);
        fft(odd, N / 2, inverse);

        // Combine results
        double angle_step = 2.0 * M_PI / N;
        // CRITICAL FIX: The sign of the angle_step is flipped for the forward FFT
        if (!inverse) {
            angle_step = -angle_step;
        }

        for (int k = 0; k < N / 2; ++k) {
            std::complex<double> twiddle_factor = std::exp(std::complex<double>(0.0, k * angle_step));
            data[k] = even[k] + twiddle_factor * odd[k];
            data[k + N / 2] = even[k] - twiddle_factor * odd[k];
        }
    }

    // Wrapper function to handle normalization
    // data: The input/output vector of complex numbers
    // N: The size of the data vector (must be a power of 2)
    // inverse: true for IFFT, false for FFT
    static void fft_normalized(std::vector<std::complex<double>>& data, int N, bool inverse) {
        fft(data, N, inverse);

        // Normalize the result by 1/sqrt(N)
        double normalization_factor = 1.0 / std::sqrt(N);
        for (int i = 0; i < N; ++i) {
            data[i] *= normalization_factor;
        }
    }

    // Static function to calculate convolution between two vectors
    static std::vector<std::complex<double>> calculateConvolution(const std::vector<std::complex<double>>& a, 
                                                                    const std::vector<std::complex<double>>& b, 
                                                                    const std::string& mode = "full") 
    {
        size_t n = a.size() + b.size() - 1; // Length of the result vector

        std::vector<std::complex<double>> result(n, {0.0, 0.0});

        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                result[i + j] += a[i] * b[j];
            }
        }

        // Mode can be "full" or "same". The length for "full" is a + b - 1, for "same" is the size of a
        // start the vector from (b.size()-1)/2 and have the same size as a
        // if (mode == "same") {
        //     // Compute starting index
        //     size_t start = (b.size() - 1) / 2;

        //     // Create shortened vector of length a
        //     result = std::vector<std::complex<double>> (result.begin() + start, result.begin() + start + a.size());

        if (mode == "same") {
            // drop anything beyond the size of a
            if (a.size() < b.size()) {
                throw std::invalid_argument("Size of first vector must be greater than or equal to second vector for 'same' mode.");
            }
            n = a.size();
            result.resize(n); // Resize to match the size of a
        }

        return result;
    }

    // Static function to calculate power of the singal
    static double calculatePower(const std::vector<std::complex<double>>& signal) {
        double power = 0.0;
        for (const auto& sample : signal) {
            power += std::norm(sample); // Use std::norm to get the squared magnitude
        }
        return power / signal.size(); // Return average power
    }

    // Static function to calculate Bit Error Rate (BER)
    // originalBits: The original transmitted bits
    // recoveredBits: The recovered bits after demodulation
    // Returns: BER value and prints detailed statistics
    static double calculateBER(const std::vector<int>& originalBits, const std::vector<int>& recoveredBits) {
        int errors = 0;
        size_t minSize = std::min(originalBits.size(), recoveredBits.size());
        
        for (size_t i = 0; i < minSize; ++i) {
            if (recoveredBits[i] == -1) {
                // Count -1 (demodulation failures) as errors
                errors++;
            } else if (originalBits[i] != recoveredBits[i]) {
                // Count incorrect demodulation as errors
                errors++;
            }
        }
        
        // BER = total errors / total transmitted bits
        double ber = static_cast<double>(errors) / minSize;
        
        return ber;
    }

    static double calculateNoisePowerFromSNR(double signalPower, double SNR_dB) {
         // Convert SNR from dB to linear scale
        double SNR_linear = std::pow(10.0, SNR_dB / 10.0);
        double noisePower = signalPower / SNR_linear;
        return noisePower;
    }

    // Static function to perform single-tap equalization
    // signal: The received signal in frequency domain
    // channelResponse: The channel frequency response (Hn)
    // Returns: Equalized signal
    static std::vector<std::complex<double>> singleTapEqualizer(const std::vector<std::complex<double>>& signal, 
                                                                const std::vector<std::complex<double>>& channelResponse) {
        std::vector<std::complex<double>> equalizedSignal = signal;
        
        for (size_t i = 0; i < equalizedSignal.size(); ++i) {
            if (i < channelResponse.size() && channelResponse[i] != std::complex<double>(0.0, 0.0)) {
                // Avoid division by zero
                equalizedSignal[i] /= channelResponse[i];
            } else {
                // Set to zero if channel response is zero or unavailable
                equalizedSignal[i] = {0.0, 0.0};
            }
        }
        
        return equalizedSignal;
    }

    // Static function to normalize signal power to match a target power
    // signal: The signal to normalize (modified in place)
    // targetPower: The target power level
    static void normalizePower(std::vector<std::complex<double>>& signal, double targetPower) {
        double currentPower = calculatePower(signal);
        if (currentPower > 0) {
            double normalizationFactor = std::sqrt(targetPower / currentPower);
            for (auto& sample : signal) {
                sample *= normalizationFactor;
            }
        }
    }

    // Static function to generate complex Gaussian white noise with specified power
    // length: Number of noise samples to generate
    // noisePower: Target noise power level
    // Returns: Vector of complex noise samples
    static std::vector<std::complex<double>> generateComplexGaussianNoise(size_t length, double noisePower) {
        std::vector<std::complex<double>> noise(length);
        
        // Create random number generators
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Standard deviation for each component (real and imaginary)
        // For complex Gaussian noise with total power P, each component has variance P/2
        double stddev = std::sqrt(noisePower / 2.0);
        std::normal_distribution<double> dist(0.0, stddev);
        
        // Generate complex noise samples
        for (size_t i = 0; i < length; ++i) {
            double real_part = dist(gen);
            double imag_part = dist(gen);
            noise[i] = std::complex<double>(real_part, imag_part);
        }
        
        return noise;
    }

    // Static function to add noise to a signal
    // signal: The signal to add noise to (modified in place)
    // noisePower: The power level of the noise to add
    static void addNoise(std::vector<std::complex<double>>& signal, double noisePower) {
        auto noise = generateComplexGaussianNoise(signal.size(), noisePower);
        for (size_t i = 0; i < signal.size(); ++i) {
            signal[i] += noise[i];
        }

        // print the information about hte maximal noise value
        // double maxNoiseValue = 0.0; 
        // for (const auto& n : noise) {
        //     double noiseMagnitude = std::abs(n);
        //     if (noiseMagnitude > maxNoiseValue) {
        //         maxNoiseValue = noiseMagnitude;
        //     }   
        // }
        // std::cout << "Max noise value added: " << maxNoiseValue << std::endl;
    }

    // Static function to export constellation points to a file for gnuplot
    // signal: The complex signal to export
    // filename: The output filename
    // title: Optional title for the data (used as comment in file)
    static void exportConstellationToFile(const std::vector<std::complex<double>>& signal, 
                                          const std::string& filename, 
                                          const std::string& title = "") {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        
        // Write header comment
        if (!title.empty()) {
            file << "# " << title << std::endl;
        }
        file << "# Real Imaginary" << std::endl;
        
        // Write constellation points
        for (const auto& point : signal) {
            file << point.real() << " " << point.imag() << std::endl;
        }
        
        file.close();
        std::cout << "Constellation data exported to: " << filename << std::endl;
    }

    // Static function to create a gnuplot script for constellation plotting
    // inputFile: Filename of input constellation data
    // outputFile: Filename of output constellation data  
    // scriptName: Name of the gnuplot script to create
    // modulationScheme: The modulation scheme to determine appropriate axis range
    static void createConstellationPlotScript(const std::string& inputFile,
                                              const std::string& outputFile,
                                              const std::string& scriptName,
                                              const std::string& modulationScheme = "16QAM") {
        std::ofstream script(scriptName);
        if (!script.is_open()) {
            std::cerr << "Error: Could not create gnuplot script " << scriptName << std::endl;
            return;
        }
        
        // Determine axis range based on modulation scheme
        int axisRange = 5; // Default for 16QAM
        if (modulationScheme == "QPSK") {
            axisRange = 2;  // QPSK: ±1
        } else if (modulationScheme == "16QAM") {
            axisRange = 5;  // 16QAM: ±3, plus margin
        } else if (modulationScheme == "64QAM") {
            axisRange = 9; // 64QAM: ±7, plus margin
        } else if (modulationScheme == "256QAM") {
            axisRange = 19;  // 256QAM: ±15, plus margin
        }
        
        script << "# Gnuplot script for constellation plotting\n";
        script << "set terminal pdf enhanced color size 12,6\n";
        script << "set output 'constellation_comparison.pdf'\n";
        script << "set multiplot layout 1,2\n\n";
        
        script << "# Input constellation\n";
        script << "set title 'Input Constellation (Transmitted)'\n";
        script << "set xlabel 'Real Part'\n";
        script << "set ylabel 'Imaginary Part'\n";
        script << "set grid\n";
        script << "set size square\n";
        script << "set xrange [-" << axisRange << ":" << axisRange << "]\n";
        script << "set yrange [-" << axisRange << ":" << axisRange << "]\n";
        script << "plot '" << inputFile << "' using 1:2 with points pointtype 7 pointsize 1.2 linecolor rgb 'dark-blue' title 'Input Symbols'\n\n";
        
        script << "# Output constellation\n";
        script << "set title 'Output Constellation (Received & Equalized)'\n";
        script << "set xlabel 'Real Part'\n";
        script << "set ylabel 'Imaginary Part'\n";
        script << "set grid\n";
        script << "set size square\n";
        script << "set xrange [-" << axisRange << ":" << axisRange << "]\n";
        script << "set yrange [-" << axisRange << ":" << axisRange << "]\n";
        script << "plot '" << outputFile << "' using 1:2 with points pointtype 7 pointsize 1.2 linecolor rgb 'dark-blue' title 'Received Symbols'\n\n";
        
        script << "unset multiplot\n";
        script.close();
        
        std::cout << "Gnuplot script created: " << scriptName << std::endl;
        std::cout << "To generate the plot, run: gnuplot " << scriptName << std::endl;
    }
};

// Class that calculates FFT and IFFT for OFDM symbols
class TxRxOFDMS {

private:
    size_t NFFTSize; // Size of the FFT
    size_t cyclicPrefixLength; // Length of the cyclic prefix

public:
    // Constructor to initialize FFT size and cyclic prefix length
    TxRxOFDMS(size_t fftSize, size_t cpLength) : NFFTSize(fftSize), cyclicPrefixLength(cpLength) {}

    // Accessor functions
    size_t getNFFTSize() const { return NFFTSize; }
    size_t getCyclicPrefixLength() const { return cyclicPrefixLength; }

    // Function to set the cyclic prefix length
    void setCyclicPrefixLength(size_t cpLength) {
        cyclicPrefixLength = cpLength;
    }

    // Function to get the cyclic prefix length
    size_t getCyclicPrefix() const {
        return cyclicPrefixLength;
    }

public:
    // Function to generate an OFDM symbol from a vector of complex numbers
    // This function performs the IFFT on the input vector and inserts a cyclic prefix
    // It returns the OFDM symbol as a vector of complex numbers
    std::vector<std::complex<double>> generateOfdmSymbol(const std::vector<std::complex<double>>& input) {
        // Placeholder for FFT implementation

        // If input size < NFFTSize, pad with zeros
        std::vector<std::complex<double>> paddedInput = input;
        if (paddedInput.size() < NFFTSize) {
            paddedInput.resize(NFFTSize, {0.0, 0.0}); // Pad with zeros to 1024 size
        }

        // Implement Cooly-Tukey IFFT algorithm 
        // For now, just return the padded input
        std::vector<std::complex<double>> output = paddedInput; // For now, just return the input
        
        // Now, perform the IFFT
        SignalProcessing::fft_normalized(output, static_cast<int>(NFFTSize), true); // Perform IFFT

        std::vector<std::complex<double>> ofdmSymbol = insertCyclicPrefix(output, cyclicPrefixLength); // Insert cyclic prefix
        return ofdmSymbol;
    }

    // Function that generates an OFDM signal from a longer vector of complex numbers by breaking into NFFTSize chunks 
    // that are then processed by generateOfdmSymbol
    std::vector<std::complex<double>> generateOfdmSignal(const std::vector<std::complex<double>>& input) {
        std::vector<std::complex<double>> ofdmSignal;
        size_t numSymbols = (input.size() + NFFTSize - 1) / NFFTSize; // Calculate number of symbols needed

        for (size_t i = 0; i < numSymbols; ++i) {
            size_t start = i * NFFTSize;
            size_t end = std::min(start + NFFTSize, input.size());
            std::vector<std::complex<double>> symbol(input.begin() + start, input.begin() + end);
            auto ofdmSymbol = generateOfdmSymbol(symbol);
            ofdmSignal.insert(ofdmSignal.end(), ofdmSymbol.begin(), ofdmSymbol.end());
        }
        return ofdmSignal;
    }

    // Function to perform FFT on a vector of complex numbers to recover the original signal from OFDM symbol
    std::vector<std::complex<double>> invertOfdmSymbol(const std::vector<std::complex<double>>& input) {
        // Placeholder for IFFT implementation
        std::vector<std::complex<double>> output = input;
        

        // First, remove cyclic prefix
        output = removeCyclicPrefix(output, cyclicPrefixLength); // Remove cyclic prefix

        // Now, perform the FFT
        SignalProcessing::fft_normalized(output, static_cast<int>(NFFTSize), false); // Perform FFT
        return output;
    }

    // Function to invert the OFDM signal by performing FFT on chunks of NFFTSize+cyclicPrefixLength
    // and applying single-tap equalization
    std::vector<std::complex<double>> invertOfdm(const std::vector<std::complex<double>>& input, 
                                                  const std::vector<std::complex<double>>& channelResponse,
                                                  size_t originalLength = 0) {
        std::vector<std::complex<double>> output;

        size_t chunkSize = NFFTSize + cyclicPrefixLength;
        size_t numChunks = input.size() / chunkSize;

        for (size_t i = 0; i < numChunks; ++i) {
            std::vector<std::complex<double>> chunk(input.begin() + i * chunkSize, input.begin() + (i + 1) * chunkSize);
            
            // Remove cyclic prefix and perform FFT
            auto fftSymbol = invertOfdmSymbol(chunk);
            
            // Apply single-tap equalization immediately after FFT
            auto equalizedSymbol = SignalProcessing::singleTapEqualizer(fftSymbol, channelResponse);
            
            output.insert(output.end(), equalizedSymbol.begin(), equalizedSymbol.end());
        }

        // Truncate to original signal length if specified
        if (originalLength > 0 && output.size() > originalLength) {
            output.resize(originalLength);
        }

        return output;
    }

    // Function to generate inserted cyclic prefix
    static std::vector<std::complex<double>> insertCyclicPrefix(const std::vector<std::complex<double>>& symbol, size_t cpLength) {
        std::vector<std::complex<double>> output;
        // Copy the last cpLength samples to the front
        output.insert(output.end(), symbol.end() - cpLength, symbol.end());
        // Append the original symbol
        output.insert(output.end(), symbol.begin(), symbol.end());
        return output;
    }

    // Function to remove cyclic prefix
    static std::vector<std::complex<double>> removeCyclicPrefix(const std::vector<std::complex<double>>& signal, size_t cpLength) {
        return std::vector<std::complex<double>>(signal.begin() + cpLength, signal.end());
    }

};

// Create a Modulator class to handle modulation schemes
class Modulator {

private:
    // Modulation scheme and its corresponding maximum constellation points
    std::string modulationScheme;
    int Nmax;
    std::vector<std::complex<double>> mappingTable;
public:
    // Modulator constructor initialized by the modulation scheme string
    Modulator(const std::string& modulationScheme) : modulationScheme(modulationScheme), Nmax(0) {
        auto it = MODULATION_TABLE.find(modulationScheme);
        if (it != MODULATION_TABLE.end()) {
            Nmax = it->second; // Set the number of constellation points based on the modulation scheme

            // Also, select the appropriate mapping table based on the modulation order
            if (modulationScheme == "QPSK") {
               
                // Use QPSK_MAPPING
                mappingTable = QPSK_MAPPING;
            } else if (modulationScheme == "16QAM") {
               
                // Use QAM16_MAPPING
                mappingTable = QAM16_MAPPING;
            } else if (modulationScheme == "64QAM") {
             
                // Use QAM64_MAPPING
                mappingTable = QAM64_MAPPING;
            } else if (modulationScheme == "256QAM") {
               
                // Use 256QAM_MAPPING
                mappingTable = QAM256_MAPPING;
            }
            std::cout << "Modulation set with: " << Nmax << " constellation points" << std::endl;
        }   
        else {
            std::cerr << "Unsupported modulation scheme: " << modulationScheme << ", using default value (QPSK)" << std::endl;
            Nmax = MODULATION_TABLE.at("QPSK"); // Default to QPSK if not found
        }
    }

    // Function to get the modulation order
    int getModulationOrder() const {
        return Nmax;
    }

    std::string getModulationScheme() const {
        return modulationScheme;
    }

    // Function to modulate a vector of bits into complex symbols
    std::vector<std::complex<double>> modulateSignal(const std::vector<int>& bits) {
        std::vector<std::complex<double>> modulatedSignal;
        modulatedSignal.reserve(bits.size()); // Reserve space
        // Map bits to QPSK symbols 
        for (size_t i = 0; i < bits.size(); ++i) {
            modulatedSignal.push_back(mappingTable[bits[i]]);
        }
        return modulatedSignal;
    }

    // Function to demodulate complex symbols back to bits
    std::vector<int> demodulateSignal(const std::vector<std::complex<double>>& symbols, double tolerance = 0.1) {
        std::vector<int> demodulatedBits;
        demodulatedBits.reserve(symbols.size());
        
        for (const auto& symbol : symbols) {
            int closestIndex = -1;
            double minDistance = std::numeric_limits<double>::max();
            
            // Find the closest constellation point
            for (size_t i = 0; i < mappingTable.size(); ++i) {
                double distance = std::abs(symbol - mappingTable[i]);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestIndex = static_cast<int>(i);
                }
            }
            
            // Only add if within tolerance, otherwise add -1 to indicate error
            if (minDistance <= tolerance && closestIndex != -1) {
                demodulatedBits.push_back(closestIndex);
            } else {
                demodulatedBits.push_back(-1); // Error indicator
            }
        }
        return demodulatedBits;
    }
};

// Generate a signal class to handle signal generation
class Signal {
private:
    std::vector<int> intSignalData;
    int NFFTSize; // Size of the FFT
    std::vector<std::complex<double>> cmplxSignalData; // Store complex signal data
    std::vector<std::complex<double>> channelImpulseResponse; // Channel impulse response
    std::vector<std::complex<double>> channelFrequencyResponse; // Channel frequency response (Hn)

public:
    // Constructor to initialize the signal with length, max value for random generation and fft size for 
    // frequency domain of the channel impulse response
    // Default modulation order is 4 (QPSK)
    // Default FFT size is 256
    // Default channel impulse response is initialized with 3 paths
    Signal(size_t length, int maxValue = 4, int nfftSize = 256) : NFFTSize(nfftSize) {

        // generate a vecotr with 8 integeers randomly generated in the range from 0 to 3
        // make the random numbers variable with every run
        // Create a random number generator
        std::random_device rd;                         // Seed source
        std::mt19937 gen(rd());                        // Mersenne Twister engine
        std::uniform_int_distribution<> dist(0, maxValue - 1);  // Inclusive range
        std::srand(static_cast<unsigned int>(std::time(nullptr))); // Seed the random number generator
        
        intSignalData.resize(length);

        // Temporary use predetermined values for testing
        // intSignalData = {4, 9, 15, 9, 11, 5, 1, 10, 4, 14, 13, 11, 10, 6, 1, 11}; // Example data for testing
       
        
        // Fill the vector with random values
        std::generate(intSignalData.begin(), intSignalData.end(), [&]() {
            return dist(gen);  // Generate a number between 0 and modulationOrder - 1
        });

        // Initialize default channel impulse response
        initializeChannel();
        
    }
    // Signal(const std::vector<std::complex<double>>& data) : signalData(data) {}

    // Function to initialize the channel impulse response with default values
    void initializeChannel() {
        // Default channel impulse response
        channelImpulseResponse = { // 1., 0.81873075, 0.67032005, 0.54881164, 0.44932896, 0.36787944
            {1.0, 0.0},   // Direct path
            {0.81873075, 0.0},   // First reflection
            {0.67032005, 0.0},   // Second reflection
            {0.54881164, 0.0},   // Third reflection
            {0.44932896, 0.0},   // Fourth reflection
            {0.36787944, 0.0}    // Fifth reflection
        };
        
        // Calculate the frequency domain representation
        calculateChannelFrequencyResponse();
    }

    // Function to set custom channel impulse response
    void setChannelImpulseResponse(const std::vector<std::complex<double>>& channelResponse) {
        channelImpulseResponse = channelResponse;
        calculateChannelFrequencyResponse();
    }

    // Function to calculate the frequency domain representation of the channel
    void calculateChannelFrequencyResponse() {
        // Copy channel impulse response and pad with zeros if necessary
        channelFrequencyResponse = channelImpulseResponse;
        if (channelFrequencyResponse.size() < static_cast<size_t>(NFFTSize)) {
            channelFrequencyResponse.resize(NFFTSize, {0.0, 0.0});
        }
        
        // Perform FFT to get frequency domain representation
        SignalProcessing::fft_normalized(channelFrequencyResponse, static_cast<int>(NFFTSize), false);
    }

    // Function to get the signal data
    const std::vector<int>& getIntSignal() const {
        return intSignalData;
    }

    // Function to get the channel impulse response
    const std::vector<std::complex<double>>& getChannelImpulseResponse() const {
        return channelImpulseResponse;
    }

    // Function to get the channel frequency response
    const std::vector<std::complex<double>>& getChannelFrequencyResponse() const {
        return channelFrequencyResponse;
    }

    // Function to print the signal data
    void printSignal() const {
        printIntVector(intSignalData, "Int Signal Data");
    }

    // Function to print channel information
    void printChannelInfo() const {
        printComplexVector(channelImpulseResponse, "Channel Impulse Response");
        printComplexVector(channelFrequencyResponse, "Channel Frequency Response (Hn)");
    }
};

// Function to parse command line arguments
struct ProgramOptions {
    std::string modulation = "16QAM";
    int length = 2000;
    int nfft = 256;
    int cyclicPrefix = 8;
    double snr = 30.0; // Default SNR in dB
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -m <modulation>    Modulation scheme (QPSK, 16QAM, 64QAM, 256QAM) [default: 16QAM]\n";
    std::cout << "  -l <length>        Number of bits to generate [default: 2000]\n";
    std::cout << "  -n|--nfft <points>  FFT size (must be power of 2) [default: 256]\n";
    std::cout << "  -u <length>        Cyclic prefix length in samples [default: 8]\n";
    std::cout << "  -h|--help          Show this help message\n";
    std::cout << "  -snr <value>       Signal to Noise Ratio in dB [default: 30 dB]\n";
    std::cout << "\nExample: " << programName << " -m 64QAM -l 10000 -n 512 -u 16\n";
}

bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

ProgramOptions parseArguments(int argc, char* argv[]) {
    ProgramOptions options;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                options.modulation = argv[++i];
                // Validate modulation scheme
                if (MODULATION_TABLE.find(options.modulation) == MODULATION_TABLE.end()) {
                    std::cerr << "Error: Invalid modulation scheme '" << options.modulation << "'\n";
                    std::cerr << "Valid options: QPSK, 16QAM, 64QAM, 256QAM\n";
                    exit(1);
                }
            } else {
                std::cerr << "Error: -m requires a modulation scheme\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-l") == 0) {
            if (i + 1 < argc) {
                options.length = std::atoi(argv[++i]);
                if (options.length <= 0) {
                    std::cerr << "Error: Length must be a positive integer\n";
                    exit(1);
                }
            } else {
                std::cerr << "Error: -l requires a length value\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-snr") == 0) {
            if (i + 1 < argc) {
                options.snr = std::atof(argv[++i]);
                
            } else {
                std::cerr << "Error: -snr requires an SNR value\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-nfft") == 0) {
            if (i + 1 < argc) {
                options.nfft = std::atoi(argv[++i]);
                if (!isPowerOfTwo(options.nfft) || options.nfft < 4) {
                    std::cerr << "Error: FFT size must be a power of 2 and >= 4\n";
                    exit(1);
                }
            } else {
                std::cerr << "Error: -n/-nfft requires an FFT size value\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-u") == 0) {
            if (i + 1 < argc) {
                options.cyclicPrefix = std::atoi(argv[++i]);
                if (options.cyclicPrefix < 0) {
                    std::cerr << "Error: Cyclic prefix length must be non-negative\n";
                    exit(1);
                }
            } else {
                std::cerr << "Error: -u requires a cyclic prefix length value\n";
                exit(1);
            }
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        }
        else {
            std::cerr << "Error: Unknown option '" << argv[i] << "'\n";
            printUsage(argv[0]);
            exit(1);
        }
    }
    
    return options;
}

// start the main function

int main(int argc, char* argv[]) {
    // Parse command line arguments
    ProgramOptions options = parseArguments(argc, argv);
    
    std::cout << "OFDM Simulator Configuration:\n";
    std::cout << "  Modulation: " << options.modulation << "\n";
    std::cout << "  Bits: " << options.length << "\n";
    std::cout << "  FFT Size: " << options.nfft << "\n";
    std::cout << "  Cyclic Prefix: " << options.cyclicPrefix << " samples\n";
    std::cout << "  SNR (dB) value: " << options.snr << "\n\n";
  

    // Create an instance of the Modulator class with specified modulation
    Modulator modulator(options.modulation);

    // Get the modulation order
    int modulationOrder = modulator.getModulationOrder();
    std::cout << " Number of constellation points: " << modulationOrder << " for " << modulator.getModulationScheme() << std::endl;

    // Create an instance of the Signal class with specified number of bits
    Signal signal(options.length, modulationOrder, options.nfft);
    // signal.printSignal(); // Print the generated random bits
    // signal.printChannelInfo(); // Print channel information

    // Get the integer signal data
    std::vector<int> randomNumbers = signal.getIntSignal();

    // generate an output complex vector by reserving space for 8 elements
    std::vector<std::complex<double>> outputComplex = modulator.modulateSignal(randomNumbers);

    // Calculate the power of the input signal
    double inputSignalPower = SignalProcessing::calculatePower(outputComplex);
    std::cout << "Input Signal Power: " << inputSignalPower << std::endl;


    // Print the mapped complex numbers
    // printComplexVector(outputComplex, "Mapped Complex Numbers");

    // Create an OFDMSymbol instance with specified FFT size and cyclic prefix length
    // Create an instance of TxRxOFDMS with the specified parameters
    TxRxOFDMS ofdmSignal(options.nfft, options.cyclicPrefix);

    // Perform FFT on the output complex vector
    std::vector<std::complex<double>> fftOutput = ofdmSignal.generateOfdmSignal(outputComplex);
    // printComplexVector(fftOutput, "FFT Output with Cyclic Prefix");

    // Get channel impulse response and frequency response from signal
    std::vector<std::complex<double>> channelImpulseResponse = signal.getChannelImpulseResponse();
    std::vector<std::complex<double>> channelFrequencyResponse = signal.getChannelFrequencyResponse();

    // Perform convolution with the channel impulse response
    std::vector<std::complex<double>> convolvedOutput = SignalProcessing::calculateConvolution(fftOutput, channelImpulseResponse, "same");
    // printComplexVector(convolvedOutput, "Convolved Output");

    // Print the power of the convolved output
    double convolvedOutputPower = SignalProcessing::calculatePower(convolvedOutput);
    std::cout << "Convolved Output Power: " << convolvedOutputPower << std::endl;
    
    // Print the size of the convolved output
    std::cout << "Size of Convolved Output: " << convolvedOutput.size() << std::endl;   

    // determine the noise power based on the desired SNR
    double noisePower = SignalProcessing::calculateNoisePowerFromSNR(convolvedOutputPower, options.snr);
    std::cout << "Adding noise with power: " << noisePower << " for SNR: " << options.snr << " dB" << std::endl;

    // Add noise here:
    SignalProcessing::addNoise(convolvedOutput, noisePower); // Adjust noise power as needed

    // Perform IFFT on the convolved output and apply equalization inside invertOfdm
    // Note: invertOfdm removes cyclic prefix, performs FFT, and applies equalization
    // Pass original signal length to truncate padded zeros
    std::vector<std::complex<double>> equalizedSignal = ofdmSignal.invertOfdm(convolvedOutput, channelFrequencyResponse, outputComplex.size());

    // Calculate the power of the recovered signal and normalize the signal so that its power is equal to the input signal power
    double recoveredSignalPower = SignalProcessing::calculatePower(equalizedSignal);
    std::cout << "Recovered Signal Power: " << recoveredSignalPower << std::endl;

    // Normalize the recovered signal to match the input signal power
    SignalProcessing::normalizePower(equalizedSignal, inputSignalPower);

    // Print first 10 outputComplex and equalizedSignal values for debugging
    // printComplexVectorLimited(outputComplex, "Input Complex Signal (Transmitted)", 10);
    // printComplexVectorLimited(equalizedSignal, "Equalized Signal (Received & Equalized)", 10);

    // Export constellation points for gnuplot visualization
    SignalProcessing::exportConstellationToFile(outputComplex, "input_constellation.dat", "Input Constellation (Transmitted)");
    SignalProcessing::exportConstellationToFile(equalizedSignal, "output_constellation.dat", "Output Constellation (Received & Equalized)");
    
    // Create gnuplot script for constellation comparison
    SignalProcessing::createConstellationPlotScript("input_constellation.dat", "output_constellation.dat", "plot_constellation.plt", modulator.getModulationScheme());

    // Demodulate the recovered signal using the modulator's demodulation method
    std::vector<int> recoveredBits = modulator.demodulateSignal(equalizedSignal, 5.0); // Use higher tolerance due to channel effects
    
    // Print the recovered bits
    // printIntVector(recoveredBits, "Recovered Bits");
    
    // Compare with original signal
    std::vector<int> originalBits = signal.getIntSignal();

    
    // printIntVector(originalBits, "Original Bits");
    
    // Calculate bit error rate using SignalProcessing utility
    double ber = SignalProcessing::calculateBER(originalBits, recoveredBits);
    auto errorCount = ber* originalBits.size();
    int errors = static_cast<int>(errorCount);
    int validBits = static_cast<int>(originalBits.size());
    
    // Print the Bit Error Rate (BER)
    std::cout << "Bit Error Rate (BER): " << ber << " (" << errors << " errors out of " << validBits << " transmitted bits)" << std::endl;

    return 0;
}