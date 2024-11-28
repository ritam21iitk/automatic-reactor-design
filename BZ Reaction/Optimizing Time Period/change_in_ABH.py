import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import itertools

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

# Function to solve and plot the Oregonator ODEs for different parameters
def oregonator_solve_ivp(A_vals, B_vals, H_vals):
    y0 = [1, 1, 1]  # Initial concentrations (dimensionless)
    t0, tstop = 0.0, 100
    tspan = np.array([t0, tstop])
    t = np.linspace(t0, tstop, 401)

    # Loop over each combination of A, B, and H+
    for A, B, H_plus in itertools.product(A_vals, B_vals, H_vals):
        # Calculate scaling factors based on given A, B, and H+
        k2 = 8E+05 * H_plus
        k4 = 2E+03
        k5 = 8 * H_plus
        k0 = 1
        X0 = k5 * A / (2 * k4)
        Y0 = k5 * A / k2
        Z0 = (k5 * A) ** 2 / (k4 * k0 * B)
        T0 = 1 / (k0 * B)

        sol = solve_ivp(oregonator_deriv, tspan, y0, args=(A, B, H_plus), t_eval=t)

        # Convert dimensionless results to original concentration and time
        original_time = sol.t * T0
        original_x = sol.y[0] * X0
        original_y = sol.y[1] * Y0
        original_z = sol.y[2] * Z0

        # Save data to CSV
        data = {
            't': original_time,
            'x(t)': original_x,
            'y(t)': original_y,
            'z(t)': original_z
        }
        filename_data = f'oregonator_solution_A{A}_B{B}_H{H_plus}_original.csv'
        df = pd.DataFrame(data)
        df.to_csv(filename_data, index=False)
        print(f'Data saved to "{filename_data}"')

        # Plot original x(t), y(t), z(t) over time
        plt.plot(original_time, original_x, label=f'H+={H_plus}, A={A}, B={B}')
        plt.xlabel('Time (s)')
        plt.ylabel('x(t) (M)')
        plt.title(f'Concentration Profile of x(t) for A={A}, B={B}, H+={H_plus}')
        plt.grid(True)
        filename_x = f'oregonator_solve_ivp_x_A{A}_B{B}_H{H_plus}_original.png'
        plt.savefig(filename_x)
        plt.close()
        
        plt.plot(original_time, original_y, label=f'H+={H_plus}, A={A}, B={B}')
        plt.xlabel('Time (s)')
        plt.ylabel('y(t) (M)')
        plt.title(f'Concentration Profile of y(t) for A={A}, B={B}, H+={H_plus}')
        plt.grid(True)
        filename_y = f'oregonator_solve_ivp_y_A{A}_B{B}_H{H_plus}_original.png'
        plt.savefig(filename_y)
        plt.close()
        
        plt.plot(original_time, original_z, label=f'H+={H_plus}, A={A}, B={B}')
        plt.xlabel('Time (s)')
        plt.ylabel('z(t) (M)')
        plt.title(f'Concentration Profile of z(t) for A={A}, B={B}, H+={H_plus}')
        plt.grid(True)
        filename_z = f'oregonator_solve_ivp_z_A{A}_B{B}_H{H_plus}_original.png'
        plt.savefig(filename_z)
        plt.close()

        # 3D Plot of x, y, z in original concentrations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(original_x, original_y, original_z, 'b-', linewidth=2)
        ax.set_xlabel('x(t) (M)')
        ax.set_ylabel('y(t) (M)')
        ax.set_zlabel('z(t) (M)')
        ax.set_title(f'Oregonator 3D Plot for A={A}, B={B}, H+={H_plus}')
        filename_xyz = f'oregonator_solve_ivp_xyz_A{A}_B{B}_H{H_plus}_original.png'
        plt.savefig(filename_xyz)
        plt.close()

        print(f'Plots saved for A={A}, B={B}, H+={H_plus}')

# Define the values of A0, B0, and [H+] to explore
A_vals = [0.06, 0.08, 0.1]      # Different values for A0
B_vals = [0.02, 0.03, 0.04]     # Different values for B0
H_vals = [0.6, 0.8, 1.0]        # Different values for [H+]

# Run the simulation for different values
oregonator_solve_ivp(A_vals, B_vals, H_vals)
