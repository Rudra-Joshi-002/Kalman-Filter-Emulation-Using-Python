"""
Estimating the Temperature of Liquid in a Tank using 1D Kalman Filter (with Process Noise)

This script implements a one-dimensional Kalman Filter that includes **process noise variance (q)**.
The example simulates estimating the temperature of a liquid using noisy measurements and assuming a 
slight fluctuation in the true liquid temperature (process noise).

Theoretical background is based on:
"Kalman Filter from The Ground Up" by Alex Becker — Chapter 5, Example 6

-----------------------------------------------------------------------------------------
MODEL AND EQUATIONS USED:
-----------------------------------------------------------------------------------------

We assume a **constant system model** with small fluctuations. So the true value is nearly constant,
but includes some random process noise.

Let:
- x̂ₙ,ₙ   = Estimated state (temperature) at time step n (after measurement)
- x̂ₙ,ₙ₋₁ = Predicted temperature from previous state (before measurement)
- zₙ     = Measured temperature at time step n
- pₙ,ₙ   = Posterior variance (after update)
- pₙ,ₙ₋₁ = Prior variance (after prediction)
- q      = Process noise variance (model uncertainty)
- r      = Measurement noise variance (sensor uncertainty)
- Kₙ     = Kalman Gain at step n

Initialization:
    x̂₀,₀ = 60.0   → initial guess of temperature (far from true value)
    p₀,₀ = 100² = 10000  → high initial uncertainty

Kalman Filter Steps at each time step n:

1. Prediction:
    x̂ₙ,ₙ₋₁ = x̂ₙ₋₁,ₙ₋₁              # For constant system
    pₙ,ₙ₋₁ = pₙ₋₁,ₙ₋₁ + q           # Add process noise variance

2. Update:
    Kₙ = pₙ,ₙ₋₁ / (pₙ,ₙ₋₁ + r)       # Kalman Gain
    x̂ₙ,ₙ = x̂ₙ,ₙ₋₁ + Kₙ * (zₙ - x̂ₙ,ₙ₋₁)
    pₙ,ₙ = (1 - Kₙ) * pₙ,ₙ₋₁

The Kalman Filter balances prediction and measurement using the Kalman Gain (Kₙ). 
As the uncertainty of estimate decreases, Kₙ decreases and the filter trusts the prediction more.

-----------------------------------------------------------------------------------------
Given Data for the Example:
-----------------------------------------------------------------------------------------
- True temperature values: [50.005, 49.994, 49.993, 50.001, 50.006, 49.998, 50.021, 50.005, 50.0, 49.997]
- Measured values: [49.986, 49.963, 50.09, 50.001, 50.018, 50.05, 49.938, 49.858, 49.965, 50.114]
- Initial estimate: 60.0
- Initial variance: 10000.0
- Measurement variance (r): 0.1² = 0.01
- Process noise variance (q): 0.0001
"""

import numpy as np
import matplotlib.pyplot as plt

# ==== Initialization ====

# True values and measurements
true_temperatures = [50.005, 49.994, 49.993, 50.001, 50.006,
                     49.998, 50.021, 50.005, 50.0, 49.997]

measurements = [49.986, 49.963, 50.09, 50.001, 50.018,
                50.05, 49.938, 49.858, 49.965, 50.114]

# Initial estimate (far from true value)
x_est = 60.0

# Variances
p_est = 100**2                 # Initial variance = 10000
measurement_variance = 0.1**2  # r = 0.01
process_variance = 0.0001      # q = 0.0001

# Lists for logging results
estimates = []
predictions = []
variances = []
kalman_gains = []
time_steps = list(range(1, len(measurements) + 1))

# ==== Kalman Filter Iteration ====

for n, z in enumerate(measurements, start=1):

    # Step 1: Prediction
    x_pred = x_est                      # Prediction equals previous estimate
    p_pred = p_est + process_variance  # Add process noise
    predictions.append(x_pred)

    # Step 2: Update
    K = p_pred / (p_pred + measurement_variance)           # Kalman Gain
    x_est = x_pred + K * (z - x_pred)                      # Updated estimate
    p_est = (1 - K) * p_pred                               # Updated variance

    # Store results
    estimates.append(x_est)
    variances.append(p_est)
    kalman_gains.append(K)

# ==== Tabular Output ====
print(f"{'Time':<6} {'True Temp':<10} {'Measured':<10} {'Predicted':<12} {'Estimate':<10} {'Variance':<10} {'Gain':<10}")
print('-' * 72)
for i in range(len(measurements)):
    print(f"{time_steps[i]:<6} {true_temperatures[i]:<10.3f} {measurements[i]:<10.3f} {predictions[i]:<12.3f} {estimates[i]:<10.3f} {variances[i]:<10.6f} {kalman_gains[i]:<10.6f}")

# ==== Plotting Results ====

# Plot true, measured, predicted, estimated temperatures
plt.figure(figsize=(10, 6))
plt.plot(time_steps, true_temperatures, 'k--', label='True Temperature')
plt.plot(time_steps, measurements, 'ro-', label='Measured Temperature')
plt.plot(time_steps, predictions, 'g^-', label='Predicted Temperature')
plt.plot(time_steps, estimates, 'bs-', label='Estimated Temperature')
plt.title('1D Kalman Filter with Process Noise: Temperature Estimation')
plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)

# Plot Kalman Gain over time
plt.figure(figsize=(10, 4))
plt.plot(time_steps, kalman_gains, 'mo-', label='Kalman Gain')
plt.title('Kalman Gain over Time')
plt.xlabel('Time Step')
plt.ylabel('Kalman Gain')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Wait for user to close plots
input("\nPress Enter to close all plots...")

