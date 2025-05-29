"""
Estimating the Temperature of Heating Liquid – Example 7 (Low Process Noise)

This script simulates a 1D Kalman Filter where:
- True system: Heating liquid (temperature increasing linearly).
- Filter assumption: Constant system (temperature does not change).
- Process noise (q) is very low → leads to a lag error.

Reference:
"Kalman Filter from The Ground Up" by Alex Becker — Chapter 5, Example 7

------------------------------------------------------------------------------------------
MODEL AND EQUATIONS USED:
------------------------------------------------------------------------------------------

1. Initialization:
   x̂₀,₀ = 10.0      # Initial estimate (°C)
   p₀,₀ = 100² = 10000   # Initial variance

2. Prediction:
   x̂ₙ,ₙ₋₁ = x̂ₙ₋₁,ₙ₋₁
   pₙ,ₙ₋₁ = pₙ₋₁,ₙ₋₁ + q

3. Update:
   Kₙ = pₙ,ₙ₋₁ / (pₙ,ₙ₋₁ + r)
   x̂ₙ,ₙ = x̂ₙ,ₙ₋₁ + Kₙ * (zₙ - x̂ₙ,ₙ₋₁)
   pₙ,ₙ = (1 - Kₙ) * pₙ,ₙ₋₁

Where:
- zₙ: measurement at step n
- x̂ₙ,ₙ: estimated temperature
- pₙ,ₙ: estimate variance
- Kₙ: Kalman gain
- q = 0.0001, r = 0.01 (variance of process and measurement noise)
"""

import numpy as np
import matplotlib.pyplot as plt

# True values and measurements
true_temps = [50.505, 50.994, 51.493, 52.001, 52.506,
              52.998, 53.521, 54.005, 54.5, 54.997]

measurements = [50.486, 50.963, 51.597, 52.001, 52.518,
                53.05, 53.438, 53.858, 54.465, 55.114]

# Initial values
x_est = 10.0
p_est = 10000.0
q = 0.0001
r = 0.01

# Storage
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
plt.title("Example 7 – KF with Low Process Noise (q = 0.0001)")
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show(block=False)

# Kalman gain plot
plt.figure(figsize=(10, 4))
plt.plot(time_steps, kalman_gains, 'mo-', label='Kalman Gain')
plt.title("Kalman Gain Over Time – Example 7")
plt.xlabel("Time Step")
plt.ylabel("Gain")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

input("\nPress Enter to close plots...")
