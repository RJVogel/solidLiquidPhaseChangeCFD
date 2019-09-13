# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:20:40 2016

@author: Julian Vogel (RJVogel)

Navier Stokes solver for lid driven cavity flow
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from drawnow import drawnow
from pathlib import Path

# Constants
NAN = np.nan

# Output
saveFiguresToFile = False
outputFilename = 'anim'

# Geometry
xmin = 0.
xmax = 2.
ymin = 0.
ymax = 2.

# Spatial discretization
dx = 0.05
dy = 0.05

# Boundary conditions
# Wall x-velocity: [W, E, S, N], NAN = symmetry
uWall = [0., 0., 0., 1.]
# Wall y-velocity: [W, E, S, N], NAN = symmetry
vWall = [0., 0., 0., 0.]
# Wall pressure: [W, E, S, N], NAN = symmetry
pWall = [NAN, NAN, NAN, NAN]
# Reference pressure value and location in ONE of the corners: [SW, NW, NE, SE]
pRef = [NAN, 0, NAN, NAN]

# Material properties
rho = 1.0
mu = 0.1
nu = mu/rho

# Temporal discretization
tMax = 1.0
dt0 = 0.001
nit = 50  # iterations of pressure poisson equation

# Visualization
dtOut = 0.05  # output step length
nOut = int(round(tMax/dtOut))
figureSize = (10, 6.25)
minContour1 = -0.29
maxContour1 = 5.8
colormap1 = 'jet'
plotContourLevels1 = np.linspace(minContour1, maxContour1, num=21)
ticks1 = np.linspace(minContour1, maxContour1, num=7)

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

# Functions


def animateContoursAndVelocityVectors():

    # plot pressure and velocity
    # Axis
    ax1 = fig.add_subplot(111)
    ax1.set_aspect(1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Pressure contours and velocity vectors')
    # Filled contours for pressure
    ctf1 = ax1.contourf(X, Y, p, plotContourLevels1, extend='both',
                        alpha=1, linestyles=None, cmap=colormap1)
    # Colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True, ticks=ticks1)
    cBar1.set_label('p / Pa')
    # plot velocity vectors
    m = 1
    ax1.quiver(X[::m, ::m], Y[::m, ::m], u[::m, ::m], v[::m, ::m])

    plt.tight_layout()

    if saveFiguresToFile:
        formattedFilename = '{0}_{1:5.3f}.png'.format(outputFilename, t)
        path = Path('out') / formattedFilename
        plt.savefig(path)


def solveMomentumEquation(u, v, un, vn, dt, dx, dy, nu):

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
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + Qv

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


def solvePoissonEquation(p, pn, b, rho, dt, dx, dy, u, v, nit, pWall):

    # Right hand side
    b[1:-1, 1:-1] = rho*(
        1/dt*((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) +
              (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy)) -
        (((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 +
         2*((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy) *
            (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) +
         ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2))

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

        # Reference pressure in ONE of the corners [SW, NW, NE, SE]
        # South West
        if not np.isnan(pRef[0]):
            p[1, 1] = pRef[0]
        # North West
        if not np.isnan(pRef[1]):
            p[-2, 1] = pRef[1]
        # North East
        if not np.isnan(pRef[2]):
            p[-2, -2] = pRef[2]
        # South East
        if not np.isnan(pRef[3]):
            p[1, -2] = pRef[3]

    return p


def correctPressure(u, v, p, rho, dt, dx, dy):

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - 1/rho*dt * \
        (p[1:-1, 2:]-p[1:-1, 0:-2])/(2*dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - 1/rho*dt * \
        (p[2:, 1:-1]-p[0:-2, 1:-1])/(2*dy)

    return u, v


# Create figure
plt.close('all')
fig = plt.figure(figsize=figureSize, dpi=100)

# Time stepping
twall0 = time.time()
tOut = dtOut
t = 0
n = 0

while t < tMax:

    n += 1

    dt = dt0
    t += dt

    # Update variables
    pn = p.copy()
    un = u.copy()
    vn = v.copy()

    # Momentum equation with projection method
    # Intermediate velocity field u*
    [u, v] = solveMomentumEquation(u, v, un, vn, dt, dx, dy, nu)
    # Set velocity boundaries
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)
    # Poisson equation
    p = solvePoissonEquation(p, pn, b, rho, dt, dx, dy, u, v, nit, pWall)
    # Pressure correction
    [u, v] = correctPressure(u, v, p, rho, dt, dx, dy)
    # Set velocity boundaries
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)

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

        print("==============================================================")
        print(" Time step n = %d, t = %8.3f, dt0 = %4.1e, t_wall = %4.1f" %
              (n, t, dt0, time.time()-twall0))
        print(" max|u| = %3.1e, CFL(u) = %3.1f, Pe(u) = %4.1f" %
              (uMax, CFL_u, Pe_u))
        print(" max|v| = %3.1e, CFL(v) = %3.1f, Pe(v) = %4.1f" %
              (vMax, CFL_v, Pe_v))

        drawnow(animateContoursAndVelocityVectors)

        tOut += dtOut
