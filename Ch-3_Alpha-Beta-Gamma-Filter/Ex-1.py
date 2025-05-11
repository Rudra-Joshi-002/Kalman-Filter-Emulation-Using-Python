"""

The following is a example of applying a Introductory Kalman Filter which
showcases the use of state update equations via python programming for a Static System
i.e. is the one in which the true value of system parameter is fixed like weight, height,
and width, etc.

The following code is the implementation of predicting the weight of Gold Bar using 
Kalman Filter Equation the Theory and Necessary background related to the example can be
found in the book "Kalman Filter from The Ground Up" By Alex Becker in Topic 3.1

State Update Equation Used Here is As Follows:

x'(n,n) = x'(n,n-1) + (1/n)*(z(n)-x'(n,n-1))

where,

x'(n,n) = Estimated Weight of Gold Bar at time 'n'

x'(n,n-1) = Prior Predicted Weight i.e. Prediction/Estimate of weight for time 'n' made at time 'n-1'

z(n) = Measured Value of Weight using a Weighing Scale at time 'n'

x(n) = True Value of Weight Which remains constant throught the Experiment

1/n = alpha (As Per Book) = Kalman Gain

"""

import numpy as np
import matplotlib.pyplot as plt

# Initialize the filter with an initial guess for the gold bar weight (grams)
x_est = 1000.0  # initial guess/estimate (hat{x}_{0,0})

# Known true weight (for comparison plot)
true_weight = 1000.0

# Given sequence of noisy measurements (grams)
measurements = [996, 994, 1021, 1000, 1002, 1010, 983, 971, 993, 1023]

# Lists to record filter estimates and (optional) predicted values
estimates = []
predictions = []

# Loop over each measurement index (1-based) and value

"""""
enumerate(measurements, start=1):
enumerate() gives both the index and the value at each step while looping.
start=1 means indexing starts from 1 instead of 0, matching the mathematical notation 
n=1,2,3,...
n=1,2,3,... used in the PDF.
n is the measurement number (first, second, third...).
z is the actual measured value at that step.
"""
for n, z in enumerate(measurements, start=1):
    
    # Prediction step: for a static system, prediction = previous estimate
    x_pred = x_est
    predictions.append(x_pred)
    
    # Compute Kalman gain (alpha) for this step: alpha = 1/n
    alpha = 1.0 / n
    
    # Update step (state update): combine prediction and measurement
    # x_est = x_pred + alpha * (z - x_pred)
    x_est = x_pred + alpha * (z - x_pred)
    estimates.append(x_est)

# Prepare data for plotting: time indices starting at 1

# Create a list of time steps starting from 1 up to the number of measurements.
# This list will be used as the x-axis values in the plot.
# Example: if there are 10 measurements, time_steps = [1, 2, 3, ..., 10]
time_steps = list(range(1, len(measurements) + 1))

# Print all values neatly in a table format
# Print a header for the table
print(f"{'Time Step':<10} {'Measured Value':<15} {'Predicted Value':<17} {'Estimated Value':<17} {'True Value':<12}")
# Print a separator line
print('-' * 75)
# Loop through and print each row of data
for n in range(len(measurements)):
    print(f"{time_steps[n]:<10} {measurements[n]:<15} {predictions[n]:<17.2f} {estimates[n]:<17.2f} {true_weight:<12}")

# Create the plot
plt.figure(figsize=(8, 5))

# Plot the true weight (constant line)
# Create a list where the true weight value is repeated for each time step.
# This allows plotting a constant reference line showing the true weight across all measurements.
# Example: if true_weight = 1000 and there are 10 steps, 
# the list will be [1000, 1000, ..., 1000] (10 times)
plt.plot(time_steps, [true_weight]*len(time_steps), 'k--', label='True Weight')

# Plot the noisy measurements
plt.plot(time_steps, measurements, 'ro-', label='Measurements')

# Plot the filter estimates
plt.plot(time_steps, estimates, 'bs-', label='Kalman Estimate')
plt.title('Kalman Filter: Gold Bar Weighing')
plt.xlabel('Measurement Index (n)')
plt.ylabel('Weight (grams)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
