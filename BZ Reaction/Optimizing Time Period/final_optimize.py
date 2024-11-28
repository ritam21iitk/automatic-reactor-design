import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt

# Define the differential equations for the Oregonator model
def oregonator_deriv(t, conc, A, B, H_plus):
    k2 = 8E+05 * H_plus
    k3 = 1.28 * H_plus ** 2
    k4 = 2E+03
    k5 = 8 * H_plus
    k0 = 1

    eta = (k0 * B) / (k5 * A)
    delta = (2 * k0 * k4 * B) / (k2 * k5 * A)
    q = (2 * k3 * k4) / (k2 * k5)
    f = 2 / 3  # f generally between 1/2 to 1

    x, y, z = conc
    dxdt = (q * y - x * y + x * (1.0 - x)) / eta
    dydt = (-q * y - x * y + f * z) / delta
    dzdt = x - z

    return [dxdt, dydt, dzdt]

# Function to calculate the period of the X concentration
def calculate_period(time, x_concentration):
    peaks, _ = find_peaks(x_concentration)
    if len(peaks) > 1:
        periods = np.diff(time[peaks])
        period = np.mean(periods)
    else:
        period = np.nan
    return period

# Objective function to minimize the difference between actual and target period
def objective(params, target_period, init_X, init_Y, init_Z):
    A, B, H_plus = params
    y0 = [init_X, init_Y, init_Z]
    k0 = 1  # As per model definition
    T0 = 1 / (k0 * B)  # Scaling factor for time
    tspan = [0, 200 / T0]  # Non-dimensional time span
    t_eval = np.linspace(0, 200 / T0, 1000)  # Non-dimensional time points
    sol = solve_ivp(oregonator_deriv, tspan, y0, args=(A, B, H_plus), t_eval=t_eval)
    
    # Convert back to dimensional time for period calculation
    period = calculate_period(sol.t * T0, sol.y[0])
    return (period - target_period) ** 2 if not np.isnan(period) else np.inf

# Set the target period (in seconds) and fixed initial concentrations
target_period = 100  # Desired period in seconds
init_X, init_Y, init_Z = 0.01, 0.01, 0.01  # Adjusted initial conditions for oscillations

# Initial guesses for A, B, and H+ and their bounds
bounds = [(0.01, 0.1), (0.01, 0.1), (0.1, 1.0)]  # Reasonable ranges for A, B, and H+

# Run the optimization with differential evolution
result = differential_evolution(objective, bounds, args=(target_period, init_X, init_Y, init_Z), strategy='best1bin', maxiter=1000)

if result.success:
    optimal_A, optimal_B, optimal_H_plus = result.x
    print(f"Optimal values for target period = {target_period} seconds:")
    print(f"  A = {optimal_A:.6f} M")
    print(f"  B = {optimal_B:.6f} M")
    print(f"  H+ = {optimal_H_plus:.6f} M")

    # Calculate scaling factors for original units
    k2 = 8E+05 * optimal_H_plus
    k4 = 2E+03
    k5 = 8 * optimal_H_plus
    k0 = 1
    X0 = k5 * optimal_A / (2 * k4)
    Y0 = k5 * optimal_A / k2
    Z0 = (k5 * optimal_A) ** 2 / (k4 * k0 * optimal_B)

    # Diagnostic test for 500 seconds
    target_total_time_test = 500  # Test time duration
    T0 = 1 / (k0 * optimal_B)
    tspan_dimensional_test = [0, target_total_time_test / T0]
    t_eval_dimensional_test = np.linspace(0, target_total_time_test / T0, 1000)
    sol_test = solve_ivp(oregonator_deriv, tspan_dimensional_test, [init_X, init_Y, init_Z], args=(optimal_A, optimal_B, optimal_H_plus), t_eval=t_eval_dimensional_test)

    original_time_test = sol_test.t * T0
    original_x_test = sol_test.y[0] * X0
    original_y_test = sol_test.y[1] * Y0
    original_z_test = sol_test.y[2] * Z0

    plt.figure(figsize=(12, 6))
    plt.plot(original_time_test, original_x_test, label='X (M)', color='b')
    plt.plot(original_time_test, original_y_test, label='Y (M)', color='g')
    plt.plot(original_time_test, original_z_test, label='Z (M)', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (M)')
    plt.title(f'Test Oscillations (500 seconds)')
    plt.legend()
    plt.savefig('test_oscillations_500s.png')
    print("Test oscillation plot saved as 'test_oscillations_500s.png'")
    plt.show()

    # Simulate for 2000 seconds
    target_total_time = 2000  # Total time in dimensional units
    tspan_dimensional = [0, target_total_time / T0]
    t_eval_dimensional = np.linspace(0, target_total_time / T0, 5000)
    sol = solve_ivp(oregonator_deriv, tspan_dimensional, [init_X, init_Y, init_Z], args=(optimal_A, optimal_B, optimal_H_plus), t_eval=t_eval_dimensional)

    original_time = sol.t * T0
    original_x = sol.y[0] * X0
    original_y = sol.y[1] * Y0
    original_z = sol.y[2] * Z0

    # Save data to CSV in original units
    data = {
        'Time (s)': original_time,
        'X (M)': original_x,
        'Y (M)': original_y,
        'Z (M)': original_z
    }
    df = pd.DataFrame(data)
    df.to_csv('optimal_A_B_H_plus_solution_2000s_original_units.csv', index=False)
    print("Data saved to 'optimal_A_B_H_plus_solution_2000s_original_units.csv'")

    # Plot in original units
    plt.figure(figsize=(12, 6))
    plt.plot(original_time, original_x, label='X (M)', color='b')
    plt.plot(original_time, original_y, label='Y (M)', color='g')
    plt.plot(original_time, original_z, label='Z (M)', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (M)')
    plt.title(f'Concentration Profiles with Optimal A, B, and H+ (2000 seconds)')
    plt.legend()
    plt.savefig('optimal_A_B_H_plus_plot_2000s_original_units.png')
    print("Plot saved as 'optimal_A_B_H_plus_plot_2000s_original_units.png'")
    plt.show()
else:
    print("Optimization did not converge.")
