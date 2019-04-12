# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:20:40 2016

@author: Julian Vogel (AkaDrBird)

Navier Stokes Fourier solver for side heated and cooled cavity
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from drawnow import drawnow
from pathlib import Path

# Output
saveFiguresToFile = False
outputFilename = 'anim'

# Geometry
xmin = 0.
xmax = 0.046
ymin = 0.
ymax = 0.046

# Spatial discretization
dx = 0.001
dy = 0.001

# Initial conditions
T0 = 7.

# Boundary conditions
# Wall temperature: [W, E, S, N], np.nan = symmetry
Twall = [12, 2, np.nan, np.nan]
# Wall x-velocity: [W, E, S, N], np.nan = symmetry
uWall = [0., 0., 0., 0]
# Wall y-velocity: [W, E, S, N], np.nan = symmetry
vWall = [0., 0., 0., 0.]
# Wall pressure: [W, E, S, N], np.nan = symmetry
pWall = [np.nan, np.nan, np.nan, np.nan]

# Physical constants
g = 9.81

# Material properties
rho = 1.24
c = 1006.1
k = 0.0247
a = k/(rho*c)
mu = 1.76e-5
nu = mu/rho
beta = 3.58e-3
Tref = 7.

# Temporal discretization
tMax = 10.
dtVisc = min(dx**2/(2*nu), dy**2/(2*nu))
dtThrm = min(dx**2/a, dy**2/a)
sigma = 0.25
dt0 = sigma*min(dtVisc, dtThrm)
dt0 = 0.005
nit = 50  # iterations of pressure poisson equation

# Visualization
dtOut = 0.1  # output step length
nOut = int(round(tMax/dtOut))
figureSize = (10, 6.25)
minContour1 = 2
maxContour1 = 12
colormap1 = 'coolwarm'
plotContourLevels1 = np.linspace(minContour1, maxContour1, num=21)
ticks1 = np.linspace(minContour1, maxContour1, num=11)
minContour2 = 0
maxContour2 = 0.04
colormap2 = 'jet'
plotContourLevels2 = np.linspace(minContour2, maxContour2, num=21)
ticks2 = np.linspace(minContour2, maxContour2, num=5)

# Mesh generation
nx = int((xmax-xmin)/dx)+1
ny = int((ymax-ymin)/dy)+1
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Initial values
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
T = T0*np.ones((ny, nx))
U = np.zeros((ny, nx))

# Functions


def animateContoursAndVelocityVectors():

    # plot temperature and pressure
    # Axis
    ax1 = fig.add_subplot(121)
    ax1.set_aspect(1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Temperature contours')
    # Filled contours for temperature
    ctf1 = ax1.contourf(X, Y, T, plotContourLevels1, extend='both',
                        alpha=1, linestyles=None, cmap=colormap1)
    # Colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True, ticks=ticks1)
    cBar1.set_label('T / °C')

    # Plot velocity
    ax2 = fig.add_subplot(122)
    ax2.set_aspect(1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Velocity contours and vectors')
    # Filled contours
    ctf2 = ax2.contourf(X, Y, U, plotContourLevels2, extend='both',
                        alpha=1, linestyles=None, cmap=colormap2)
    # plot velocity vectors
    m = 1
    ax2.quiver(X[::m, ::m], Y[::m, ::m], u[::m, ::m], v[::m, ::m])
    # Colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cBar2 = fig.colorbar(ctf2, cax=cax2, extendrect=True, ticks=ticks2)
    cBar2.set_label('U / m/s')

    plt.tight_layout()

    plt.show()

    if saveFiguresToFile:
        formattedFilename = '{0}_{1:5.3f}.png'.format(outputFilename, t)
        path = Path('out') / formattedFilename
        plt.savefig(path)


def solveMomentumEquation(u, v, un, vn, dt, dx, dy, nu, beta, T, Tref, g):

    # Solve u-velocity
    Qu = -dt/(dx)*((np.maximum(un[1:-1, 1:-1], 0)*un[1:-1, 1:-1] +
                   np.minimum(un[1:-1, 1:-1], 0)*un[1:-1, 2:]) - (
                   np.maximum(un[1:-1, 1:-1], 0)*un[1:-1, 0:-2] +
                   np.minimum(un[1:-1, 1:-1], 0)*un[1:-1, 1:-1])) + \
        -dt/(dy)*((np.maximum(vn[1:-1, 1:-1], 0)*un[1:-1, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*un[2:, 1:-1]) - (
                  np.maximum(vn[1:-1, 1:-1], 0)*un[0:-2, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*un[1:-1, 1:-1])) + \
        nu * (
            dt/dx**2*(un[1:-1, 2:]-2*un[1:-1, 1:-1]+un[1:-1, 0:-2]) +
            dt/dy**2*(un[2:, 1:-1]-2*un[1:-1, 1:-1]+un[0:-2, 1:-1]))
    u[1:-1, 1:-1] = un[1:-1, 1:-1] + Qu

    # Solve v-velocity
    Qv = -dt/(dx)*((np.maximum(un[1:-1, 1:-1], 0)*vn[1:-1, 1:-1] +
                   np.minimum(un[1:-1, 1:-1], 0)*vn[1:-1, 2:]) - (
                   np.maximum(un[1:-1, 1:-1], 0)*vn[1:-1, 0:-2] +
                   np.minimum(un[1:-1, 1:-1], 0)*vn[1:-1, 1:-1])) + \
        -dt/(dy)*((np.maximum(vn[1:-1, 1:-1], 0)*vn[1:-1, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*vn[2:, 1:-1]) - (
                  np.maximum(vn[1:-1, 1:-1], 0)*vn[0:-2, 1:-1] +
                  np.minimum(vn[1:-1, 1:-1], 0)*vn[1:-1, 1:-1])) + \
        nu * (
            dt/dx**2*(vn[1:-1, 2:]-2*vn[1:-1, 1:-1]+vn[1:-1, 0:-2]) +
            dt/dy**2*(vn[2:, 1:-1]-2*vn[1:-1, 1:-1]+vn[0:-2, 1:-1]))
    Sv = dt*beta*g*(T[1:-1, 1:-1]-Tref)
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + Qv + Sv

    return u, v


def setVelocityBoundaries(u, v, uWall, vWall):

    # West
    u[:, 0] = uWall[0]
    if np.isnan(vWall[0]):
        v[:, 0] = v[:, 1]  # symmetry
    else:
        v[:, 0] = vWall[0]
    # East
    u[:, -1] = uWall[1]
    if np.isnan(vWall[1]):
        v[:, -1] = v[:, -2]  # symmetry
    else:
        v[:, -1] = vWall[1]
    # South
    if np.isnan(uWall[2]):
        u[0, :] = u[1, :]  # symmetry
    else:
        u[0, :] = uWall[2]
    v[0, :] = vWall[2]
    # North
    if np.isnan(uWall[3]):
        u[-1, :] = u[-2, :]  # symmetry
    else:
        u[-1, :] = uWall[3]
    v[-1, :] = vWall[3]

    return u, v


def solvePoissonEquation(p, pn, b, rho, dt, dx, dy, u, v, nit, pWall,
                         beta, g, T):

    # Right hand side
    b[1:-1, 1:-1] = rho*(
        1/dt*((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) +
              (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))
        - (((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 +
           2*((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy) *
              (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) +
           ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2)
        + beta*g*(T[2:, 1:-1] - T[0:-2, 1:-1])/(2*dy))

    # Reference pressure in upper left corner
    # p[-1, 1] = 0

    # Solve iteratively for pressure
    for nit in range(50):

        p[1:-1, 1:-1] = (dy**2*(p[1:-1, 2:]+p[1:-1, :-2]) +
                         dx**2*(p[2:, 1:-1]+p[:-2, 1:-1]) -
                         b[1:-1, 1:-1]*dx**2*dy**2)/(2*(dx**2+dy**2))

        # Boundary conditions
        # West
        if np.isnan(pWall[0]):
            p[:, 0] = p[:, 1]  # symmetry
        else:
            p[:, 0] = pWall[0]
        # East
        if np.isnan(pWall[1]):
            p[:, -1] = p[:, -2]  # symmetry
        else:
            p[:, -1] = pWall[1]
        # South
        if np.isnan(pWall[2]):
            p[0, :] = p[1, :]  # symmetry
        else:
            p[0, :] = pWall[2]
        # North
        if np.isnan(pWall[3]):
            p[-1, :] = p[-2, :]  # symmetry
        else:
            p[-1, :] = pWall[3]

    return p


def correctPressure(u, v, p, rho, dt, dx, dy):

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - 1/rho*dt * \
        (p[1:-1, 2:]-p[1:-1, 0:-2])/(2*dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - 1/rho*dt * \
        (p[2:, 1:-1]-p[0:-2, 1:-1])/(2*dy)

    return u, v


def solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, Twall):

    # Solve energy equation
    T[1:-1, 1:-1] = Tn[1:-1, 1:-1] - (
        dt/(dx)*(np.maximum(u[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 1:-1]-Tn[1:-1, 0:-2]) +
                 np.minimum(u[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 2:]-Tn[1:-1, 1:-1])) +
        dt/(dy)*(np.maximum(v[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 1:-1]-Tn[0:-2, 1:-1]) +
                 np.minimum(v[1:-1, 1:-1], 0) *
                 (Tn[2:, 1:-1]-Tn[1:-1, 1:-1]))) + \
        a * \
        (dt/dx**2*(Tn[1:-1, 2:]-2*Tn[1:-1, 1:-1]+Tn[1:-1, 0:-2]) +
         dt/dy**2*(Tn[2:, 1:-1]-2*Tn[1:-1, 1:-1]+Tn[0:-2, 1:-1]))

    # Boundary conditions
    # West
    if np.isnan(Twall[0]):
        T[:, 0] = T[:, 1]  # adiabatic, at xmin
    else:
        T[:, 0] = Twall[0]  # temperature, at xmin
    # East
    if np.isnan(Twall[1]):
        T[:, -1] = T[:, -2]  # adiabatic, at xmax
    else:
        T[:, -1] = Twall[1]  # temperature, at xmax
    # South
    if np.isnan(Twall[2]):
        T[0, :] = T[1, :]  # adiabatic, at ymin
    else:
        T[0, :] = Twall[2]  # temperature, at ymin
    # North
    if np.isnan(Twall[3]):
        T[-1, :] = T[-2, :]  # adiabatic, at ymax
    else:
        T[-1, :] = Twall[3]  # temperature, at ymax

    return T


# Create figure
plt.close('all')
fig = plt.figure(figsize=figureSize, dpi=100)

# Time stepping
twall0 = time.time()
tOut = dtOut
t = 0
n = 0
dt = dt0

while t < tMax:

    n += 1

    t += dt

    # Update variables
    pn = p.copy()
    un = u.copy()
    vn = v.copy()
    Tn = T.copy()

    # Momentum equation with projection method
    # Intermediate velocity field u*
    [u, v] = solveMomentumEquation(u, v, un, vn, dt, dx, dy, nu,
                                   beta, T, Tref, g)
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)
    # Pressure correction
    p = solvePoissonEquation(p, pn, b, rho, dt, dx, dy, u, v, nit, pWall,
                             beta, g, T)
    [u, v] = correctPressure(u, v, p, rho, dt, dx, dy)
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)

    # Energy equation
    T = solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, Twall)

    # Output
    if (t-tOut) > -1e-6:

        # Calculate derived quantities
        # Velocities
        U = (u**2+v**2)**0.5
        uMax = np.max(np.abs(u))
        vMax = np.max(np.abs(v))
        # Peclet numbers
        Pe_u = uMax*dx/nu
        Pe_v = vMax*dy/nu
        # Courant Friedrichs Levy numbers
        CFL_u = uMax*dt0/dx
        CFL_v = vMax*dt0/dy
        # Rayleigh numbers
        dTmaxLiq = np.max(T)-np.min(T)
        RaW = g*beta*dTmaxLiq*(xmax-xmin)**3/(nu*a)
        RaH = g*beta*dTmaxLiq*(ymax-ymin)**3/(nu*a)

        print("==============================================================")
        print(" Time step n = %d, t = %8.3f, dt0 = %4.1e, t_wall = %4.1f" %
              (n, t, dt0, time.time()-twall0))
        print(" max|u| = %3.1e, CFL(u) = %3.1f, Pe(u) = %4.1f, RaW = %3.1e" %
              (uMax, CFL_u, Pe_u, RaW))
        print(" max|v| = %3.1e, CFL(v) = %3.1f, Pe(v) = %4.1f, RaH = %3.1e" %
              (vMax, CFL_v, Pe_v, RaH))

        drawnow(animateContoursAndVelocityVectors)

        tOut += dtOut
