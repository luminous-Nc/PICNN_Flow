import numpy as np
import sympy as sp
import pylbm
import matplotlib.pyplot as plt

def generate_lbm_image(image_number, point, vectorA, vectorB, Tf=100):
    Re = 30.0

    parallelogramPoint = point
    parallelogramVectorA = vectorA
    parallelogramVectorB = vectorB
    X, Y, LA = sp.symbols('X, Y, LA')
    rho, qx, qy = sp.symbols('rho, qx, qy')
    Tf = Tf  # final time of the simulation

    def bc_in(f, m, x, y):
        # Define non-uniform inlet condition
        # Example: Linear profile from v0 at the bottom to 0 at the top
        m[qx] = rhoo * v0 * (1 - 2 * y / (ymax - ymin))
        m[qy] = 0.0

    def vorticity(sol):
        ux = sol.m[qx] / sol.m[rho]
        uy = sol.m[qy] / sol.m[rho]
        V = np.abs(uy[2:, 1:-1] - uy[0:-2, 1:-1] - ux[1:-1, 2:] + ux[1:-1, 0:-2]) / (2 * sol.domain.dx)
        return -V

    def calculatePolygonCoordinates(point, vectorA, vectorB, dx):
        px = point[0]
        py = point[1]
        Ax = vectorA[0]
        Ay = vectorA[1]
        Bx = vectorB[0]
        By = vectorB[1]
        return [[px / dx, py / dx], [(px + Ax) / dx, (py + Ay) / dx], [(px + Ax + Bx) / dx, (py + Ay + By) / dx],
                [(px + Bx) / dx, (py + By) / dx]]

    # parameters
    rayon = 0.05

    dx = 1. / 64  # spatial step
    la = 1.  # velocity of the scheme

    v0 = la / 20  # maximal velocity obtained in the middle of the channel
    rhoo = 1.  # mean value of the density
    mu = 1.e-3  # bulk viscosity
    eta = rhoo * v0 * 2 * rayon / Re  # shear viscosity

    # initialization
    xmin, xmax, ymin, ymax = 0., 4., 0., 2.
    dummy = 3.0 / (la * rhoo * dx)
    s_mu = 1.0 / (0.5 + mu * dummy)
    s_eta = 1.0 / (0.5 + eta * dummy)
    s_q = s_eta
    s_es = s_mu
    s = [0., 0., 0., s_mu, s_es, s_q, s_q, s_eta, s_eta]
    dummy = 1. / (LA ** 2 * rhoo)
    qx2 = dummy * qx ** 2
    qy2 = dummy * qy ** 2
    q2 = qx2 + qy2
    qxy = dummy * qx * qy

    # print("Reynolds number: {:.2f}".format(Re))
    # print("Bulk viscosity : {0:10.3e}".format(mu))
    # print("Shear viscosity: {0:10.3e}".format(eta))
    # print("relaxation parameters: {0}".format(s))

    dico = {
        'box': {'x': [xmin, xmax],
                'y': [ymin, ymax],
                'label': [0, 1, 2, 2]
                },
        'elements': [pylbm.Parallelogram(parallelogramPoint, parallelogramVectorA, parallelogramVectorB, label=3)],
        'space_step': dx,
        'scheme_velocity': la,
        'parameters': {LA: la},
        'schemes': [
            {
                'velocities': list(range(9)),
                'conserved_moments': [rho, qx, qy],
                'polynomials': [
                    1, LA * X, LA * Y,
                       3 * (X ** 2 + Y ** 2) - 4,
                       (9 * (X ** 2 + Y ** 2) ** 2 - 21 * (X ** 2 + Y ** 2) + 8) / 2,
                       3 * X * (X ** 2 + Y ** 2) - 5 * X, 3 * Y * (X ** 2 + Y ** 2) - 5 * Y,
                       X ** 2 - Y ** 2, X * Y
                ],
                'relaxation_parameters': s,
                'equilibrium': [
                    rho, qx, qy,
                    -2 * rho + 3 * q2,
                    rho - 3 * q2,
                    -qx / LA, -qy / LA,
                    qx2 - qy2, qxy
                ],
            },
        ],
        'init': {rho: rhoo,
                 qx: 0.,
                 qy: 0.
                 },
        'boundary_conditions': {
            0: {'method': {0: pylbm.bc.BouzidiBounceBack}, 'value': bc_in},  # Velocity inlet
            1: {'method': {0: pylbm.bc.NeumannX}},  # Constant pressure outlet
            2: {'method': {0: pylbm.bc.BouzidiBounceBack}},  # No-slip top wall
            3: {'method': {0: pylbm.bc.BouzidiBounceBack}},  # Object boundary
        },
        'generator': 'cython',
    }

    sol = pylbm.Simulation(dico)
    while sol.t < Tf:
        sol.one_time_step()

    pas = 8
    y, x = np.meshgrid(sol.domain.y[::pas] * 64, sol.domain.x[::pas] * 64)
    u = sol.m[qx][::pas, ::pas] / sol.m[rho][::pas, ::pas]
    v = sol.m[qy][::pas, ::pas] / sol.m[rho][::pas, ::pas]
    nv = np.sqrt(u ** 2 + v ** 2)
    u = u / (nv + 1e-5)
    v = v / (nv + 1e-5)
    plt.quiver(x, y, u, v, nv, pivot='mid')
    picture_name = f"{image_number}_lbm.png"
    plt.savefig(picture_name)

    u_x = sol.m[qx] / sol.m[rho]

    # 保存图像为PNG格式
    plt.imsave(f'{image_number}_lbmx.png', np.flipud(u_x).transpose(), cmap='coolwarm', vmin=None, vmax=None,
               format='png')

    u_y = sol.m[qy] / sol.m[rho]

    # 保存图像为PNG格式
    plt.imsave(f'{image_number}_lbmy.png', np.flipud(u_y).transpose(), cmap='coolwarm', vmin=None, vmax=None,
               format='png')

    print(f'{image_number} saved.')