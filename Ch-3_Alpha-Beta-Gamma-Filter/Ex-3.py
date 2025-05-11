"""
Tracking the Accelerating Aircraft

The following is a example of applying an Introductory Kalman Filter which
showcases the use of state update equations via python programming for a Dynamic System
i.e. the one in which the true value of system parameter changes with time like position, altitude,
and velocity, etc.

This code implements α−β filter to track an accelerating aircraft. The Theory and background
related to the example can be found in the book "Kalman Filter from The Ground Up" By Alex Becker
in Topic 3.3

Scenario:
- The aircraft moves at constant velocity for 20 seconds.
- After 20 seconds, it accelerates at 8 m/s² for 35 seconds.
- Radar measurements are noisy.
- The α–β filter attempts to track the aircraft range and velocity.

State Extrapolation Equations:
x'(n,n-1)   = x'(n-1,n-1) + Δt * vx'(n-1,n-1)
vx'(n,n-1)  = vx'(n-1,n-1)

State Update Equations:
x'(n,n)     = x'(n,n-1) + α * ( z(n) - x'(n,n-1) )
vx'(n,n)    = vx'(n,n-1) + β * [ ( z(n) - x'(n,n-1) ) / Δt ]

where,

x'(n,n)     = Estimated Range of Aircraft at time 'n'
x'(n,n-1)   = Predicted Range of Aircraft at time 'n' from time 'n-1'
z(n)        = Measured Value of Range at time 'n' from Radar
vx'(n,n)    = Estimated Velocity at time 'n'
vx'(n,n-1)  = Predicted Velocity at time 'n' from time 'n-1'
α, β        = Filter constants (gain), values depend on Radar precision
"""

import numpy as np
import matplotlib.pyplot as plt

# Filter constants for imprecise radar
alpha = 0.2
beta = 0.1

# Initial conditions
x_est = 30000     # Initial estimated range (m)
vx_est = 50       # Initial estimated velocity (m/s)
delta_t = 5       # Track-to-track interval in seconds

# Measured radar range values (m)
measurements = [30221, 30453, 30906, 30999, 31368, 31978, 32526, 33379, 34698, 36275]

# Lists to store results
x_estimates = []
vx_estimates = []
x_predictions = []
vx_predictions = []
true_ranges = []
true_velocities = []
time_steps = list(range(1, len(measurements) + 1))

# True motion model parameters
x0_true = 30000  # Initial true position (m)
v0_true = 50     # Initial velocity (m/s)
a_true = 8       # Acceleration (m/s²)

"""
enumerate(measurements, start=1):
- enumerate() provides both index and value.
- n: index starting from 1 (1st measurement).
- z: measured radar value at time step n.
"""
for n, z in enumerate(measurements, start=1):

    # ===== True Value Calculations =====
    total_time = n * delta_t
    if total_time <= 20:
        # Constant velocity phase
        true_range = x0_true + v0_true * total_time
        true_velocity = v0_true
    else:
        # Accelerated phase
        t_accel = total_time - 20
        true_range = x0_true + v0_true * 20 + v0_true * t_accel + 0.5 * a_true * t_accel ** 2
        true_velocity = v0_true + a_true * t_accel

    true_ranges.append(true_range)
    true_velocities.append(true_velocity)

    # ===== Prediction Step =====
    x_pred = x_est + delta_t * vx_est     # Predicted range
    vx_pred = vx_est                      # Predicted velocity remains unchanged
    x_predictions.append(x_pred)
    vx_predictions.append(vx_pred)

    # ===== Update Step =====
    residual = z - x_pred                 # Innovation (difference between measurement and prediction)
    x_est = x_pred + alpha * residual     # Update range
    vx_est = vx_pred + (beta * residual / delta_t)  # Update velocity
    x_estimates.append(x_est)
    vx_estimates.append(vx_est)

# ===== Print Tabular Output =====
print(f"{'Time Step':<10} {'Measured':<10} {'Predicted':<12} {'Estimated':<12} {'True':<10} {'Est. Velocity':<15} {'True Velocity':<15}")
print('-' * 90)
for i in range(len(measurements)):
    print(f"{time_steps[i]:<10} {measurements[i]:<10} {x_predictions[i]:<12.2f} {x_estimates[i]:<12.2f} "
          f"{true_ranges[i]:<10.2f} {vx_estimates[i]:<15.2f} {true_velocities[i]:<15.2f}")

# ===== Plotting =====
plt.figure(figsize=(10, 6))

# Plot true range
plt.plot(time_steps, true_ranges, 'k--', label='True Range')

# Plot measured values
plt.plot(time_steps, measurements, 'ro-', label='Measured Range')

# Plot predicted range
plt.plot(time_steps, x_predictions, 'g^-', label='Predicted Range')

# Plot estimated range
plt.plot(time_steps, x_estimates, 'bs-', label='Estimated Range')

plt.title('α–β Filter Tracking of Accelerating Aircraft')
plt.xlabel('Time Step (n)')
plt.ylabel('Range (meters)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Optional: Velocity Plot =====
plt.figure(figsize=(10, 6))

# Plot true velocity
plt.plot(time_steps, true_velocities, 'k--', label='True Velocity')

# Plot predicted velocity
plt.plot(time_steps, vx_predictions, 'g^-', label='Predicted Velocity')

# Plot estimated velocity
plt.plot(time_steps, vx_estimates, 'bs-', label='Estimated Velocity')

plt.title('α–β Filter Velocity Tracking of Accelerating Aircraft')
plt.xlabel('Time Step (n)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
