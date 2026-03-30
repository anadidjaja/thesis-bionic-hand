import serial
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd # Import pandas

# --- Serial Port Parameters ---
# Make sure to change 'COM9' to the correct port your Arduino is connected to.
# On Linux/macOS, this might look like '/dev/ttyACM0' or '/dev/ttyUSB0'.
serial_port = "/dev/cu.usbserial-0001"
baud_rate = 115200

# --- Data Acquisition Parameters ---
sampling_rate = 1000  # Hz
duration = 15         # seconds
num_samples = int(sampling_rate * duration)

# --- Data Storage ---
ecg_data = []

print(f"Attempting to open serial port {serial_port} at {baud_rate} baud...")

try:
    # --- Open Serial Connection ---
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    time.sleep(2) # Give the Arduino time to reset

    print(f"Serial port opened successfully. Reading {num_samples} samples over {duration} seconds...")

    # --- Real-time Plot Setup ---
    plt.ion()
    rt_fig, rt_ax = plt.subplots(figsize=(12, 6))
    rt_line, = rt_ax.plot([], [])
    rt_time_text = rt_ax.text(0.02, 0.95, "Time: 0.00 s", transform=rt_ax.transAxes, va='top')
    rt_ax.set_title('Real-time ECG Signal')
    rt_ax.set_xlabel('Time (s)')
    rt_ax.set_ylabel('Sensor Value')
    rt_ax.grid(True)

    # --- Read Data ---
    start_time = time.time()
    while len(ecg_data) < num_samples:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line: # Check if the line is not empty
                    value = float(line)
                    ecg_data.append(value)
                    # --- Update Real-time Plot ---
                    if len(ecg_data) % 5 == 0:
                        current_time_axis = np.linspace(0, duration, len(ecg_data))
                        rt_line.set_data(current_time_axis, ecg_data)
                        rt_time_text.set_text(f"Time: {time.time() - start_time:.2f} s")
                        rt_ax.relim()
                        rt_ax.autoscale_view()
                        rt_fig.canvas.draw()
                        rt_fig.canvas.flush_events()
                    # Optional: print progress
                    # print(f"Read sample {len(ecg_data)}/{num_samples}")
            except ValueError:
                # print(f"Could not convert data to int: {line}")
                pass # Ignore lines that aren't integers
            except UnicodeDecodeError:
                # print(f"Could not decode line: {line}")
                pass # Ignore decoding errors

    end_time = time.time()
    print(f"Finished reading data. Total time: {end_time - start_time:.2f} seconds")

    # --- Close Serial Connection ---
    ser.close()
    print("Serial port closed.")

    # --- Process and Plot Data ---
    if ecg_data:
        ecg_data = np.array(ecg_data)
        time_axis = np.linspace(0, duration, len(ecg_data))

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, ecg_data)
        plt.title('ECG Signal from Arduino')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.grid(True)
        plt.show()

        # --- Save Data to CSV ---
        df = pd.DataFrame({'Time (s)': time_axis, 'Sensor Value': ecg_data})
        csv_filename = 'test.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Data saved to {csv_filename}")

    else:
        print("No data was collected.")

except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    print("Please check:")
    print("1. If the Arduino is connected.")
    print("2. If the correct serial port ('COM9' or similar) is selected.")
    print("3. If the Arduino IDE Serial Monitor is closed (only one program can access the port).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
