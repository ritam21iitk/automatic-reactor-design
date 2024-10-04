'''
Concentration:
Malonic acid 0.2 M
Sodium bromate 0.3 M
Sulfuric acid 0.3 M
Ferroin 0.005 M

X = [HBrO2] (hypobromous acid),
Y = [Br−] (bromide)
Z = [Ce(IV)] (cerium-4)
A = [BrO−3 ] (bromate)
B = [Org] (organic species)
P = [HOBr].

k3 = kR3[H+]2AY
k2 = kR2[H+]XY
k5 = kR5[H+]AX
k4 = kR4X2

k2 = 8 × 10^5 M−1 s−1,
k3 = 1.28 M−1 s−1,
k4 = 2 × 10^3 M−1 s−1,
k5 = 8.0 M−1 s−1.

A0 = [BrO−3 ]0 = 0.06 M and B0 = [Org]0 = 0.02 M, k0 =1M−1 s−1

ε = 4 × 10^−2, δ = 4 × 10^−4, q = 8 × 10^−4.
'''

def oregonator_deriv(t, conc):
    import numpy as np
    k2 = 8E+05
    k3 = 1.28
    k4 = 2E+03
    k5 = 8
    k0 = 1
    A = 0.06
    B = 0.02

    eta = (k0*B)/(k5*A)
    delta = (2*k0*k4*B)/(k2*k5*A)
    q = (2*k3*k4)/(k2*k5)
    f = 2/3 #f generally between 1/2 to 1

    x = conc[0]
    y = conc[1]
    z = conc[2]
    
    dxdt = (q * y - x * y + x * (1.0 - x)) / eta
    dydt = (-q * y - x * y + f * z) / delta
    dzdt = x-z
    
    dcdt = np.array([dxdt, dydt, dzdt])
    return dcdt

def oregonator_solve_ivp():
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print('')
    print('oregonator_solve_ivp():')
    print('  Solve oregonator_ode() using solve_ivp().')
    y0 = [1,1,1]
    f = oregonator_deriv
    t0 = 0.0
    tstop = 25
    tspan = np.array([t0, tstop])
    t = np.linspace(t0, tstop, 101)
    sol = solve_ivp(f, tspan, y0, t_eval=t)
    data = {
        't': sol.t,
        'x(t)': sol.y[0],
        'y(t)': sol.y[1],
        'z(t)': sol.y[2]
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('oregonator_solution.csv', index=False)
    print('Data has been saved to "oregonator_solution.csv".')
    plt.plot(t, np.log10(sol.y[0]), 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('<-- Time -->')
    plt.ylabel('<-- log10(x(t)) -->')
    plt.title('oregonator_ode(): solve_ivp, log10(x(t))')
    filename = 'oregonator_solve_ivp_x.png'
    plt.savefig(filename)
    print('  Graphics saved as "%s"' % (filename))
    plt.show(block=False)
    plt.close()
    plt.plot(t, np.log10(sol.y[1]), 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('<-- Time -->')
    plt.ylabel('<-- log10(y(t)) -->')
    plt.title('oregonator_ode(): solve_ivp, log10(y(t))')
    filename = 'oregonator_solve_ivp_y.png'
    plt.savefig(filename)
    print('  Graphics saved as "%s"' % (filename))
    plt.show(block=False)
    plt.close()
    plt.plot(t, np.log10(sol.y[2]), 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('<-- Time -->')
    plt.ylabel('<-- log10(z(t)) -->')
    plt.title('oregonator_ode(): solve_ivp, log10(z(t))')
    filename = 'oregonator_solve_ivp_z.png'
    plt.savefig(filename)
    print('  Graphics saved as "%s"' % (filename))
    plt.show(block=False)
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Corrected line
    ax.plot(np.log10(sol.y[0]), np.log10(sol.y[1]), np.log10(sol.y[2]), 'b-', linewidth=2)
    ax.grid(True)
    ax.set_xlabel('<-- log10(x(t)) -->')
    ax.set_ylabel('<-- log10(y(t)) -->')
    ax.set_zlabel('<-- log10(z(t)) -->')
    ax.set_title('oregonator_ode(): solve_ivp, log10(x,y,z)')
    filename = 'oregonator_solve_ivp_xyz.png'
    plt.savefig(filename)
    print('  Graphics saved as "%s"' % (filename))
    plt.show(block=False)
    plt.close()
def oregonator_ode_test():
    import platform
    print('')
    print('oregonator_ode_test():')
    print('  Python version: %s' % (platform.python_version()))
    oregonator_solve_ivp()

def timestamp():
    import time
    t = time.time()
    print(time.ctime(t))

if (__name__ == '__main__'):
    timestamp()
    oregonator_ode_test()
    timestamp()
