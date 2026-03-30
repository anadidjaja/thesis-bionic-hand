import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the CSV file
csv_filename = 'readings.csv'  # Update this path to your actual CSV file
try:
    df = pd.read_csv(csv_filename)
    ecg_data = df['voltage'].values # Get the sensor values as a numpy array
    sampling_rate = 1000  # Hz (as specified)

    # --- Perform FFT ---
    # Calculate the FFT of the signal
    fft_result = np.fft.fft(ecg_data)

    # Calculate the corresponding frequencies
    n = len(ecg_data)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)

    # Take the magnitude of the FFT result (we are interested in the amplitude of each frequency)
    # We only need the positive frequencies, which are the first half of the result
    magnitude = np.abs(fft_result)[:n//2]
    freqs_positive = frequencies[:n//2]

    # --- Plot the Frequency Spectrum ---
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_positive, magnitude)
    plt.title('Frequency Spectrum of ECG Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0.5, 500) # Set x-axis limits
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()

    print("Successfully computed and plotted the FFT of the ECG data.")

except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found. Please make sure the data was saved.")
except KeyError as e:
    print(f"Error: Column {e} not found in the CSV file.")
except Exception as e:
    print(f"An unexpected error occurred during FFT analysis: {e}")