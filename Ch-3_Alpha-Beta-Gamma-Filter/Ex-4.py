"""
Tracking the Constant Acceleration Aircraft using Alpha-Beta-Gamma Filter

The following is an example of applying an Introductory Kalman Filter 
(α–β–γ filter) which showcases the use of state update and extrapolation 
equations via Python programming for a dynamic system — in this case,
an aircraft moving with constant velocity followed by constant acceleration.

The theoretical background and step-by-step explanation of this example
can be found in the book "Kalman Filter from The Ground Up" by Alex Becker
in Topic 3.4 (Example 4).

The following assumptions are made while implementing the example:

1. Aircraft moves radially in one dimension.
2. For the first 20 seconds, the velocity is constant (50 m/s).
3. After 20 seconds, the aircraft accelerates at 8 m/s² for 35 seconds.
4. Radar provides noisy measurements of range every 5 seconds.
5. The initial acceleration is assumed to be 0 m/s².

State Extrapolation Equations (Prediction Step):

x'(n+1,n)   = x'(n,n) + vx'(n,n) * Δt + 0.5 * ax'(n,n) * Δt²
vx'(n+1,n)  = vx'(n,n) + ax'(n,n) * Δt
ax'(n+1,n)  = ax'(n,n)

State Update Equations (Correction Step):

residual    = z(n) - x'(n,n-1)

x'(n,n)     = x'(n,n-1) + α * residual
vx'(n,n)    = vx'(n,n-1) + β * (residual / Δt)
ax'(n,n)    = ax'(n,n-1) + γ * (residual / (0.5 * Δt²))

where,

x'(n,n)     = Estimated Range at time step 'n' after update
vx'(n,n)    = Estimated Velocity at time step 'n' after update
ax'(n,n)    = Estimated Acceleration at time step 'n' after update

x'(n+1,n)   = Predicted Range at next step based on current state
vx'(n+1,n)  = Predicted Velocity at next step
ax'(n+1,n)  = Predicted Acceleration (unchanged)

z(n)        = Measured Range using radar at time 'n'
Δt          = Time interval between two measurements

α, β, γ     = Filter constants:
              α = 0.5 (position gain)
              β = 0.4 (velocity gain)
              γ = 0.1 (acceleration gain)
              Δt = 5 seconds
"""


import numpy as np
import matplotlib.pyplot as plt

# Filter constants
alpha = 0.5
beta = 0.4
gamma = 0.1
delta_t = 5  # Time interval (s)

# Initial estimates
x_est = 30000     # Initial range estimate (m)
vx_est = 50       # Initial velocity estimate (m/s)
ax_est = 0        # Initial acceleration estimate (m/s²)

# Radar measurements (in meters)
measurements = [30221, 30453, 30906, 30999, 31368, 31978, 32526, 33379, 34698, 36275]

# Initialize lists to store filter values
x_estimates = []
vx_estimates = []
ax_estimates = []
x_predictions = []
vx_predictions = []
ax_predictions = []
true_ranges = []
true_velocities = []
true_accelerations = []
time_steps = list(range(1, len(measurements) + 1))

# True motion model parameters
x0_true = 30000
v0_true = 50
a_true = 8

"""
Loop over each measurement:
- Generate true position, velocity, acceleration
- Predict next state using extrapolation equations
- Update state using measurement with correction equations
"""
for n, z in enumerate(measurements, start=1):

    # ====== TRUE STATE CALCULATION ======
    total_time = n * delta_t
    if total_time <= 20:
        true_range = x0_true + v0_true * total_time
        true_velocity = v0_true
        true_accel = 0
    else:
        t_accel = total_time - 20
        true_range = x0_true + v0_true * 20 + v0_true * t_accel + 0.5 * a_true * t_accel ** 2
        true_velocity = v0_true + a_true * t_accel
        true_accel = a_true

    true_ranges.append(true_range)
    true_velocities.append(true_velocity)
    true_accelerations.append(true_accel)

    # ====== PREDICTION STEP ======
    x_pred = x_est + delta_t * vx_est + 0.5 * ax_est * delta_t ** 2
    vx_pred = vx_est + ax_est * delta_t
    ax_pred = ax_est

    # Store predicted values
    x_predictions.append(x_pred)
    vx_predictions.append(vx_pred)
    ax_predictions.append(ax_pred)

    # ====== UPDATE STEP ======
    residual = z - x_pred
    x_est = x_pred + alpha * residual
    vx_est = vx_pred + beta * (residual / delta_t)
    ax_est = ax_pred + gamma * (residual / (0.5 * delta_t ** 2))

    # Store updated estimates
    x_estimates.append(x_est)
    vx_estimates.append(vx_est)
    ax_estimates.append(ax_est)

# ===== PRINT TABULATED OUTPUT =====
print(f"{'Time Step':<10} {'Measured':<10} {'Predicted':<12} {'Estimated':<12} {'True':<10} {'Est. Vel.':<12} {'True Vel.':<12} {'Est. Acc.':<12} {'True Acc.':<10}")
print('-' * 110)
for i in range(len(measurements)):
    print(f"{time_steps[i]:<10} {measurements[i]:<10} {x_predictions[i]:<12.2f} {x_estimates[i]:<12.2f} "
          f"{true_ranges[i]:<10.2f} {vx_estimates[i]:<12.2f} {true_velocities[i]:<12.2f} {ax_estimates[i]:<12.2f} {true_accelerations[i]:<10.2f}")

# ===== PLOTTING =====

# Plot range
plt.figure(figsize=(10, 6))
plt.plot(time_steps, true_ranges, 'k--', label='True Range')
plt.plot(time_steps, measurements, 'ro-', label='Measured Range')
plt.plot(time_steps, x_predictions, 'g^-', label='Predicted Range')
plt.plot(time_steps, x_estimates, 'bs-', label='Estimated Range')
plt.title('α–β–γ Filter: Range Tracking')
plt.xlabel('Time Step (n)')
plt.ylabel('Range (meters)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot velocity
plt.figure(figsize=(10, 6))
plt.plot(time_steps, true_velocities, 'k--', label='True Velocity')
plt.plot(time_steps, vx_predictions, 'g^-', label='Predicted Velocity')
plt.plot(time_steps, vx_estimates, 'bs-', label='Estimated Velocity')
plt.title('α–β–γ Filter: Velocity Tracking')
plt.xlabel('Time Step (n)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot acceleration
plt.figure(figsize=(10, 6))
plt.plot(time_steps, true_accelerations, 'k--', label='True Acceleration')
plt.plot(time_steps, ax_predictions, 'g^-', label='Predicted Acceleration')
plt.plot(time_steps, ax_estimates, 'bs-', label='Estimated Acceleration')
plt.title('α–β–γ Filter: Acceleration Tracking')
plt.xlabel('Time Step (n)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
