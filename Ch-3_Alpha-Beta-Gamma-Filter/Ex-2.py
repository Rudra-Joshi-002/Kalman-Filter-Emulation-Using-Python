"""
Tracking the Constant Velocity Aircraft

The following is a example of applying an Introductory Kalman Filter which
showcases the use of state update equations via python programming for a Dynamic System
i.e. the one in which the true value of system parameter changes with time like position, altitude,
and velocity, etc.

The following code is the implementation of predicting the range/distance of an Aircraft using 
Kalman Filter Equations. The Theory and Necessary background related to the example can be
found in the book "Kalman Filter from The Ground Up" By Alex Becker in Topic 3.2

Here the following assumptions are made while implementing the example:

1. Aircraft is moving Radially Away or Towards the Radar
2. Angle to Radar is Constant
3. Aircraft Altitude is Constant
4. The 1-D Velocity of Aircraft is Also Constant [***More Imp. Point to Note***]
5. The Radar sends a track beam in the direction of the target at a constant rate. This 
   track-to-track time interval is '(delta) t'.   
6. We have an imprecise Radar and a low-speed UAV Target

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

# Filter constant values for low precision radar
alpha = 0.2
beta = 0.1

# Initial guess of aircraft range and velocity
x_est = 30000      # Initial estimated range in meters
vx_est = 40        # Initial estimated velocity in m/s
delta_t = 5        # Time interval between measurements (seconds)

# Radar measurements (in meters)
measurements = [30171, 30353, 30756, 30799, 31018, 31278, 31276, 31379, 31748, 32175]

# Lists to record estimates and predictions
x_estimates = []
vx_estimates = []
x_predictions = []
vx_predictions = []
time_steps = list(range(1, len(measurements) + 1))  # Time steps for plotting

# Compute true range values based on constant velocity motion model
x0_true = 30000  # Initial true range (m)
v_true = 40      # True velocity (m/s)

# Generate true range at each time step
true_ranges = []


"""
enumerate(measurements, start=1):
- enumerate() provides both the index and value for each radar measurement.
- start=1 aligns with the notation n=1,2,3,... used in the example.
- n: time step
- z: measurement at time n
"""
for n, z in enumerate(measurements, start=1):

    # ==== True Value Calculations ====
    true_range = x0_true + v_true * n * delta_t
    true_ranges.append(true_range)

    # ===== Prediction Step using State Extrapolation Equations =====
    x_pred = x_est + delta_t * vx_est  # Predicted range based on previous state
    vx_pred = vx_est                   # Predicted velocity remains constant

    # Save predicted values
    x_predictions.append(x_pred)
    vx_predictions.append(vx_pred)

    # ===== Update Step using α−β Filter Equations =====
    residual = z - x_pred  # Innovation: difference between measurement and prediction

    # Update estimated range using α gain
    x_est = x_pred + alpha * residual

    # Update estimated velocity using β gain
    vx_est = vx_pred + (beta * residual / delta_t)

    # Save current estimates
    x_estimates.append(x_est)
    vx_estimates.append(vx_est)

# ===== Print All Values in Tabular Format =====
print(f"{'Time Step':<10} {'Measured':<10} {'Predicted':<12} {'Estimated':<12} {'Est. Velocity':<15}")
print('-' * 65)
for i in range(len(measurements)):
    print(f"{time_steps[i]:<10} {measurements[i]:<10} {x_predictions[i]:<12.2f} {x_estimates[i]:<12.2f} {vx_estimates[i]:<15.2f}")

# ===== Plotting the Results =====
plt.figure(figsize=(10, 6))

# Plot true range values based on initial conditions and motion model
plt.plot(time_steps, true_ranges, 'k--', label='True Range')

# Plot measured values (radar readings)
plt.plot(time_steps, measurements, 'ro-', label='Measured Range')

# Plot predicted positions (based on model)
plt.plot(time_steps, x_predictions, 'g^-', label='Predicted Range')

# Plot estimated positions (after Kalman correction)
plt.plot(time_steps, x_estimates, 'bs-', label='Estimated Range')

plt.title('α–β Filter Tracking of Constant Velocity Aircraft')
plt.xlabel('Time Step (n)')
plt.ylabel('Range (meters)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
