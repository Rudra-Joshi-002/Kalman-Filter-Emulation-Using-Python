"""
Estimating the Height of a Building using One-Dimensional Kalman Filter

This script demonstrates the use of a simple 1D Kalman Filter for estimating the height of a building,
based on noisy altimeter measurements. The true building height is constant, and the dynamic model
is static (i.e., it does not change over time).

The theoretical background and full derivation of the equations used in this script are explained in:
"Kalman Filter from The Ground Up" by Alex Becker — Chapter 4, Example 5.

-----------------------------------------------------------------------------------------
Kalman Filter Equations Used (Constant Model - No Process Noise):
-----------------------------------------------------------------------------------------

Initialization (at n = 0):
    x̂₀,₀ = initial state estimate
    p₀,₀ = initial estimate variance (uncertainty)

At each time step n ≥ 1:

1. Prediction Step (since system is static, predicted state remains same):
    x̂ₙ,ₙ₋₁ = x̂ₙ₋₁,ₙ₋₁
    pₙ,ₙ₋₁ = pₙ₋₁,ₙ₋₁

2. Measurement:
    zₙ     = Measured height at time n
    rₙ     = Measurement variance (from sensor specs)

3. Kalman Gain Calculation:
    Kₙ = pₙ,ₙ₋₁ / (pₙ,ₙ₋₁ + rₙ)

4. State Update:
    x̂ₙ,ₙ = x̂ₙ,ₙ₋₁ + Kₙ * (zₙ - x̂ₙ,ₙ₋₁)
    pₙ,ₙ = (1 - Kₙ) * pₙ,ₙ₋₁

Where:
    x̂ₙ,ₙ = Estimated height at time step n
    pₙ,ₙ = Updated uncertainty (variance) at time step n
    Kₙ   = Kalman Gain (weight assigned to measurement vs prediction)
    zₙ   = Noisy measurement
    rₙ   = Measurement uncertainty (variance)

-----------------------------------------------------------------------------------------
Given Data:
-----------------------------------------------------------------------------------------
- True Height = 50 meters
- Initial Estimate (or guess) = 60 meters
- Initial Variance (or Uncertainity due to Measurment Errors) = 225 (std dev = 15 m)
- Measurement Variance = 25 (std dev = 5 m)
- Measurements = [49.03, 48.44, 55.21, 49.98, 50.6, 52.61, 45.87, 42.64, 48.26, 55.84]
"""

import numpy as np
import matplotlib.pyplot as plt

# ==== Initialization ====

true_height = 50.0                     # True value of the building height
initial_estimate = 60.0               # Initial guess of the height
initial_variance = 15 ** 2            # Variance = std_dev^2 = 225
measurement_variance = 5 ** 2         # Measurement variance (r) = 25

measurements = [49.03, 48.44, 55.21, 49.98, 50.6,
                52.61, 45.87, 42.64, 48.26, 55.84]

# Initialize lists to store results
time_steps = list(range(1, len(measurements) + 1))
estimates = []
variances = []
kalman_gains = []
predictions = []

# Initialize the first estimate and variance
x_est = initial_estimate
p_est = initial_variance

# ==== Kalman Filter Iteration ====

for n, z in enumerate(measurements, start=1):

    # Step 1: Prediction
    x_pred = x_est           # Since system is static
    p_pred = p_est

    # Store predicted value for plotting
    predictions.append(x_pred)

    # Step 2: Measurement update
    K = p_pred / (p_pred + measurement_variance)           # Kalman Gain
    x_est = x_pred + K * (z - x_pred)                      # Update estimate
    p_est = (1 - K) * p_pred                               # Update variance

    # Store results
    kalman_gains.append(K)
    estimates.append(x_est)
    variances.append(p_est)

# ==== Tabular Output ====
print(f"{'Time Step':<10} {'Measurement':<12} {'Prediction':<12} {'Estimate':<12} {'Variance':<10} {'Kalman Gain':<12}")
print('-' * 70)
for i in range(len(measurements)):
    print(f"{time_steps[i]:<10} {measurements[i]:<12.2f} {predictions[i]:<12.2f} {estimates[i]:<12.2f} {variances[i]:<10.2f} {kalman_gains[i]:<12.2f}")

# ==== Plotting ====

# Plot of true height, measurements, estimates
plt.figure(figsize=(10, 6))
plt.plot(time_steps, [true_height] * len(time_steps), 'k--', label='True Height')
plt.plot(time_steps, measurements, 'ro-', label='Measured Height')
plt.plot(time_steps, estimates, 'bs-', label='Estimated Height')
plt.plot(time_steps, predictions, 'g^-', label='Predicted Height')
plt.title('1D Kalman Filter: Building Height Estimation')
plt.xlabel('Measurement Number')
plt.ylabel('Height (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)

# Plot Kalman Gain vs time
plt.figure(figsize=(10, 4))
plt.plot(time_steps, kalman_gains, 'mo-', label='Kalman Gain')
plt.title('Kalman Gain Over Time')
plt.xlabel('Measurement Number')
plt.ylabel('Kalman Gain')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

input("\nPress Enter to close all plots...")
