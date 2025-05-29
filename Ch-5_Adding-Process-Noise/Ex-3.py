"""
Estimating the Temperature of Heating Liquid – Example 8 (High Process Noise)

This script simulates a Kalman Filter with a much higher process noise (q = 0.15),
which makes the filter respond quickly to changes in temperature and reduces lag.

Reference:
"Kalman Filter from The Ground Up" by Alex Becker — Chapter 5, Example 8

Same measurements and true values as Example 7, but:
- q = 0.15 (instead of 0.0001)
- r = 0.01
- Better dynamic response and tracking
"""

import numpy as np
import matplotlib.pyplot as plt

# Reuse data
true_temps = [50.505, 50.994, 51.493, 52.001, 52.506,
              52.998, 53.521, 54.005, 54.5, 54.997]

measurements = [50.486, 50.963, 51.597, 52.001, 52.518,
                53.05, 53.438, 53.858, 54.465, 55.114]

# Initial settings
x_est = 10.0
p_est = 10000.0
q = 0.15
r = 0.01

predictions, estimates, kalman_gains, variances = [], [], [], []
time_steps = list(range(1, len(measurements) + 1))

for z in measurements:
    x_pred = x_est
    p_pred = p_est + q

    K = p_pred / (p_pred + r)
    x_est = x_pred + K * (z - x_pred)
    p_est = (1 - K) * p_pred

    predictions.append(x_pred)
    estimates.append(x_est)
    kalman_gains.append(K)
    variances.append(p_est)

# Print table
print(f"{'Time':<6} {'True':<10} {'Measured':<10} {'Predicted':<12} {'Estimate':<10} {'Gain':<10} {'Variance':<10}")
print("-" * 72)
for i in range(len(time_steps)):
    print(f"{time_steps[i]:<6} {true_temps[i]:<10.3f} {measurements[i]:<10.3f} {predictions[i]:<12.3f} "
          f"{estimates[i]:<10.3f} {kalman_gains[i]:<10.6f} {variances[i]:<10.6f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time_steps, true_temps, 'k--', label='True Temperature')
plt.plot(time_steps, measurements, 'ro-', label='Measured')
plt.plot(time_steps, predictions, 'g^-', label='Predicted')
plt.plot(time_steps, estimates, 'bs-', label='Estimated')
plt.title("Example 8 – KF with Higher Process Noise (q = 0.15)")
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show(block=False)

# Kalman gain plot
plt.figure(figsize=(10, 4))
plt.plot(time_steps, kalman_gains, 'mo-', label='Kalman Gain')
plt.title("Kalman Gain Over Time – Example 8")
plt.xlabel("Time Step")
plt.ylabel("Gain")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

input("\nPress Enter to close plots...")
