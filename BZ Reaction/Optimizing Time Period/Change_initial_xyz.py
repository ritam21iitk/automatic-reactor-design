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

# Function to solve and plot the Oregonator ODEs for different initial concentrations
def oregonator_solve_ivp(initial_conditions):
    # Set constant parameters
    A = 0.06  # BrO3- initial concentration
    B = 0.02  # Organic species initial concentration
    H_plus = 0.8  # H+ concentration

    # Calculate scaling factors
    k2 = 8E+05 * H_plus
    k4 = 2E+03
    k5 = 8 * H_plus
    k0 = 1
    X0 = k5 * A / (2 * k4)
    Y0 = k5 * A / k2
    Z0 = (k5 * A) ** 2 / (k4 * k0 * B)
    T0 = 1 / (k0 * B)

    t0, tstop = 0.0, 100
    tspan = np.array([t0, tstop])
    t = np.linspace(t0, tstop, 401)

    # Loop over each set of initial conditions
    for init_X, init_Y, init_Z in initial_conditions:
        # Convert initial conditions to dimensionless values
        y0 = [init_X / X0, init_Y / Y0, init_Z / Z0]

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
        filename_data = f'oregonator_solution_X{init_X}_Y{init_Y}_Z{init_Z}_original.csv'
        df = pd.DataFrame(data)
        df.to_csv(filename_data, index=False)
        print(f'Data saved to "{filename_data}"')

        # Plot original x(t), y(t), z(t) over time
        plt.plot(original_time, original_x, label=f'X0={init_X}, Y0={init_Y}, Z0={init_Z}')
        plt.xlabel('Time (s)')
        plt.ylabel('x(t) (M)')
        plt.title(f'Concentration Profile of x(t) for Initial X0={init_X}, Y0={init_Y}, Z0={init_Z}')
        plt.grid(True)
        filename_x = f'oregonator_solve_ivp_x_X{init_X}_Y{init_Y}_Z{init_Z}_original.png'
        plt.savefig(filename_x)
        plt.close()

        plt.plot(original_time, original_y, label=f'X0={init_X}, Y0={init_Y}, Z0={init_Z}')
        plt.xlabel('Time (s)')
        plt.ylabel('y(t) (M)')
        plt.title(f'Concentration Profile of y(t) for Initial X0={init_X}, Y0={init_Y}, Z0={init_Z}')
        plt.grid(True)
        filename_y = f'oregonator_solve_ivp_y_X{init_X}_Y{init_Y}_Z{init_Z}_original.png'
        plt.savefig(filename_y)
        plt.close()

        plt.plot(original_time, original_z, label=f'X0={init_X}, Y0={init_Y}, Z0={init_Z}')
        plt.xlabel('Time (s)')
        plt.ylabel('z(t) (M)')
        plt.title(f'Concentration Profile of z(t) for Initial X0={init_X}, Y0={init_Y}, Z0={init_Z}')
        plt.grid(True)
        filename_z = f'oregonator_solve_ivp_z_X{init_X}_Y{init_Y}_Z{init_Z}_original.png'
        plt.savefig(filename_z)
        plt.close()

        # 3D Plot of x, y, z in original concentrations
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(original_x, original_y, original_z, 'b-', linewidth=2)
        ax.set_xlabel('x(t) (M)')
        ax.set_ylabel('y(t) (M)')
        ax.set_zlabel('z(t) (M)')
        ax.set_title(f'Oregonator 3D Plot for Initial X0={init_X}, Y0={init_Y}, Z0={init_Z}')
        filename_xyz = f'oregonator_solve_ivp_xyz_X{init_X}_Y{init_Y}_Z{init_Z}_original.png'
        plt.savefig(filename_xyz)
        plt.close()

        print(f'Plots saved for Initial X0={init_X}, Y0={init_Y}, Z0={init_Z}')

# Define different initial concentrations for X, Y, Z
initial_conditions = [
    (0.0001, 0.0000005, 0.005),
    (0.00012, 0.0000006, 0.006),
    (0.00015, 0.0000007, 0.007)
]

# Run the simulation for different initial conditions
oregonator_solve_ivp(initial_conditions)

