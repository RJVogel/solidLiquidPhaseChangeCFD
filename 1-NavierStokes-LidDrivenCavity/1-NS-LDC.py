# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:20:40 2016

@author: Julian Vogel (RJVogel)

Navier Stokes solver for lid driven cavity flow
"""

import numpy as np
from scipy import sparse as sps
import scipy.sparse.linalg as spla
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
xmax = 2.0
ymin = 0.
ymax = 2.0

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
pRef = [0, NAN, NAN, NAN]

# Material properties
rho = 1.0
mu = 0.1
nu = mu/rho

# Solver
poissonSolver = 'iterative'

# Temporal discretization
tMax = 1.
dt0 = 0.001
nit = 50  # iterations of pressure poisson equation

# Visualization
dtOut = 1.  # output step length
nOut = int(round(tMax/dtOut))
figureSize = (10, 6.25)
minContour1 = 0.055
maxContour1 = 14.9
colormap1 = 'jet'
plotContourLevels1 = np.linspace(minContour1, maxContour1, num=21)
ticks1 = np.linspace(minContour1, maxContour1, num=7)

# Mesh generation
# Number of inner points
nx = int((xmax-xmin)/dx)
ny = int((ymax-ymin)/dy)
# Centers with boundary nodes
x = np.linspace(xmin-dx/2, xmax+dx/2, nx+2)
y = np.linspace(ymin-dy/2, ymax+dy/2, ny+2)
X, Y = np.meshgrid(x, y)
# Faces with boundary nodes
xf = np.linspace(xmin, xmax, nx+1)
yf = np.linspace(ymin, ymax, ny+1)
Xf, Yf = np.meshgrid(xf, yf)

# Initial values
u = np.zeros((ny+2, nx+1))
v = np.zeros((ny+1, nx+2))
p = np.zeros((ny+2, nx+2))
b = np.zeros((ny+2, nx+2))

# Poisson Matrix
# Neighbor coefficients x-direction
en = np.ones((1, nx))/(dx*dx)
# Point coefficients x-direction
ep = -2*np.ones((1, nx))
ep[:, 0] += 1  # west side Neumann boundary condition
ep[:, -1] += 1  # east side Neumann boundary condition
ep = ep/(dx*dx)
# 1D coefficient matrix x-direction
Ax1d = sps.spdiags((ep.flat[:], en.flat[:], en.flat[:]),
                   [0, 1, -1], (nx), (nx))
# Identity matrix x-direction
Iy = sps.eye(ny)
# Full 2d coefficient matrix x-direction
Ax = sps.kron(Iy, Ax1d)
# Neighbor coefficients y-direction
en = np.ones((1, ny))/(dy*dy)
# Point coefficients y-direction
ep = -2*np.ones((1, ny))
ep[:, 0] += 1  # south side Neumann boundary condition
ep[:, -1] += 1  # north side Neumann boundary condition
ep = ep/(dy*dy)
# 1D coefficient matrix y-direction
Ay1d = sps.spdiags((ep.flat[:], en.flat[:], en.flat[:]),
                   [0, 1, -1], (ny), (ny))
# Identity matrix y-direction
Ix = sps.eye(nx)
# Full 2d coefficient matrix y-direction
Ay = sps.kron(Ay1d, Ix)
# Full matrix both directions
A = sps.csr_matrix(Ax + Ay)
#  Set zero pressure values at boundary nodes in south west corner
A[0, 0] = 3/2*A[0, 0]


# Functions


def avg(array, axis):

    # Compute average
    if len(array.shape) == 1:
        return (array[:-1]+array[1:])/2
    elif axis == 0:
        return (array[:-1, :]+array[1:, :])/2
    elif axis == 1:
        return (array[:, :-1]+array[:, 1:])/2
    else:
        return NAN


def animateContoursAndVelocityVectors():

    # plot pressure and velocity
    # Axis
    ax1 = fig.add_subplot(121)
    ax1.set_aspect(1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Pressure contours and velocity vectors')
    # Contours of pressure
    ctf1 = ax1.contourf(Xf, Yf, pf, 41, cmap=colormap1)
    # Colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True)
    cBar1.set_label('p / Pa')
    # plot velocity vectors
    m = 1
    ax1.quiver(X[1:-1, 1:-1][::m, ::m], Y[1:-1, 1:-1][::m, ::m],
               uc[1:-1, :][::m, ::m], vc[:, 1:-1][::m, ::m])

    # Axis
    ax2 = fig.add_subplot(122)
    ax2.set_aspect(1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Divergence of velocity')
    # Pcolor of divergence
    pc1 = ax2.pcolor(Xf, Yf, divU, cmap=colormap1)
    # Colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cBar2 = fig.colorbar(pc1, cax=cax2, extendrect=True)
    cBar2.set_label('div (U) / 1/s')

    plt.tight_layout()

    if saveFiguresToFile:
        formattedFilename = '{0}_{1:5.3f}.png'.format(outputFilename, t)
        path = Path('out') / formattedFilename
        plt.savefig(path)


def setVelocityBoundaries(u, v, uWall, vWall):

    # West
    u[:, 0] = uWall[0]
    if np.isnan(vWall[0]):
        v[:, 0] = v[:, 1]  # symmetry
    else:
        v[:, 0] = 2*vWall[0] - v[:, 1]  # wall
    # East
    u[:, -1] = uWall[1]
    if np.isnan(vWall[1]):
        v[:, -1] = v[:, -2]  # symmetry
    else:
        v[:, -1] = 2*vWall[1] - v[:, -2]  # wall
    # South
    v[0, :] = vWall[2]
    if np.isnan(uWall[2]):
        u[0, :] = u[1, :]  # symmetry
    else:
        u[0, :] = 2*uWall[2] - u[1, :]  # wall
    # North
    v[-1, :] = vWall[3]
    if np.isnan(uWall[3]):
        u[-1, :] = u[-2, :]  # symmetry
    else:
        u[-1, :] = 2*uWall[3] - u[-2, :]  # wall

    return u, v


def solveMomentumEquation(u, v, un, vn, dt, dx, dy, nu):

    # Interpolated values
    uah = avg(un, 1)  # horizontal average lives in cell centers
    uav = avg(un, 0)  # vertical average lives in cell corners
    vah = avg(vn, 1)  # horizontal average lives in cell corners
    vav = avg(vn, 0)  # vertical average lives in cell centers

    # Non-linear terms
    # u-velocity
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt*(
        1/(dx)*(uah[1:-1, 1:]*uah[1:-1, 1:] -
                uah[1:-1, :-1]*uah[1:-1, :-1]) +
        1/(dy)*(vah[1:, 1:-1]*uav[1:, 1:-1] -
                vah[:-1, 1:-1]*uav[:-1, 1:-1]))
    # v-velocity
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt*(
        1/(dx)*(uav[1:-1, 1:]*vah[1:-1, 1:] -
                uav[1:-1, :-1]*vah[1:-1, :-1]) +
        1/(dy)*(vav[1:, 1:-1]*vav[1:, 1:-1] -
                vav[:-1, 1:-1]*vav[:-1, 1:-1]))

    # Diffusive terms
    # u-velocity
    u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*(
        nu * (1/dx**2*(u[1:-1, 2:]-2*u[1:-1, 1:-1]+u[1:-1, :-2]) +
              1/dy**2*(u[2:, 1:-1]-2*u[1:-1, 1:-1]+u[:-2, 1:-1])))
    # v-velocity
    v[1:-1, 1:-1] = v[1:-1, 1:-1] + dt*(
        nu * (1/dx**2*(v[1:-1, 2:]-2*v[1:-1, 1:-1]+v[1:-1, :-2]) +
              1/dy**2*(v[2:, 1:-1]-2*v[1:-1, 1:-1]+v[:-2, 1:-1])))

    return u, v


def solvePoissonEquation(p, pn, b, rho, dt, dx, dy, u, v, nit, pWall):

    if poissonSolver == 'direct':

        # Right hand side
#        b[1:-1, 1:-1] = rho/dt*(np.diff(u[1:-1, :], axis=1)/dx +
#                                np.diff(v[:, 1:-1], axis=0)/dy)
        
        # Interpolations
        uahv = avg(avg(u, 1), 0)
        vavh = avg(avg(v, 0), 1)

        b[1:-1, 1:-1] = rho*(
            1/dt*((u[1:-1, 1:] - u[1:-1, :-1])/(dx) +
                  (v[1:, 1:-1] - v[:-1, 1:-1])/(dy)) -
            (((u[1:-1, 1:] - u[1:-1, :-1])/(dx))**2 +
             2*((uahv[1:, :] - uahv[:-1, :])/(dy) *
                (vavh[:, 1:] - vavh[:, :-1])/(dx)) +
             ((v[1:, 1:-1] - v[:-1, 1:-1])/(dy))**2))

        p[1:-1, 1:-1].flat[:] = -spla.spsolve(A, -b[1:-1, 1:-1].flat[:])
        
        # Boundary conditions
        # West
        if np.isnan(pWall[0]):
            p[:, 0] = p[:, 1]  # symmetry
        else:
            p[:, 0] = 2*pWall[0] - p[:, 1]  # wall
        # East
        if np.isnan(pWall[1]):
            p[:, -1] = p[:, -2]  # symmetry
        else:
            p[:, -1] = 2*pWall[1] - p[:, -2]  # wall
        # South
        if np.isnan(pWall[2]):
            p[0, :] = p[1, :]  # symmetry
        else:
            p[0, :] = 2*pWall[2] - p[1, :]  # wall
        # North
        if np.isnan(pWall[3]):
            p[-1, :] = p[-2, :]  # symmetry
        else:
            p[-1, :] = 2*pWall[3] - p[-2, :]  # wall

    elif (poissonSolver == 'iterative'):

        # Interpolations
        uahv = avg(avg(u, 1), 0)
        vavh = avg(avg(v, 0), 1)

        # Right hand side
        b[1:-1, 1:-1] = rho*(
            1/dt*((u[1:-1, 1:] - u[1:-1, :-1])/(dx) +
                  (v[1:, 1:-1] - v[:-1, 1:-1])/(dy)) -
            (((u[1:-1, 1:] - u[1:-1, :-1])/(dx))**2 +
             2*((uahv[1:, :] - uahv[:-1, :])/(dy) *
                (vavh[:, 1:] - vavh[:, :-1])/(dx)) +
             ((v[1:, 1:-1] - v[:-1, 1:-1])/(dy))**2))

        # Solve iteratively for pressure
        for nit in range(50):

            p[1:-1, 1:-1] = (dy**2*(p[1:-1, 2:]+p[1:-1, :-2]) +
                             dx**2*(p[2:, 1:-1]+p[:-2, 1:-1]) -
                             b[1:-1, 1:-1]*dx**2*dy**2)/(2*(dx**2+dy**2))

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

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - dt/rho * \
        (p[1:-1, 2:-1]-p[1:-1, 1:-2])/(dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - dt/rho * \
        (p[2:-1, 1:-1]-p[1:-2, 1:-1])/(dy)

    return u, v


# Create figure
# plt.close('all')
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

    # Projection method
    # Set velocity boundaries
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)
    # Intermediate velocity field u*
    [u, v] = solveMomentumEquation(u, v, un, vn, dt, dx, dy, nu)
    # Solve poisson equation
    p = solvePoissonEquation(p, pn, b, rho, dt, dx, dy, u, v, nit, pWall)
    # Pressure correction
    [u, v] = correctPressure(u, v, p, rho, dt, dx, dy)

    if (t-tOut) > -1e-6:

        # Calculate derived quantities

        # Divergence
        divU = np.diff(u[1:-1, :], axis=1)/np.diff(Xf[1:, :], axis=1) + \
            np.diff(v[:, 1:-1], axis=0)/np.diff(Yf[:, 1:], axis=0)
        # Pressure at cell corners
        pf = avg(avg(p, 0), 1)
        # Velocities
        uc = avg(u, 1)
        vc = avg(v, 0)
        ucorn = avg(u, 0)
        vcorn = avg(v, 1)
        U = (uc[1:-1, :]**2+vc[:, 1:-1]**2)**0.5
        uMax = np.max(np.abs(ucorn))
        vMax = np.max(np.abs(vcorn))
        # Peclet numbers
        Pe_u = uMax*dx/nu
        Pe_v = vMax*dy/nu
        # Courant Friedrichs Levy numbers
        CFL_u = uMax*dt0/dx
        CFL_v = vMax*dt0/dy

        print("==============================================================")
        print(" Time step n = %d, t = %8.3f, dt0 = %4.1e, t_wall = %4.1f" %
              (n, t, dt0, time.time()-twall0))
        print(" max|u| = %5.2e, CFL(u) = %5.2f, Pe(u) = %5.2f" %
              (uMax, CFL_u, Pe_u))
        print(" max|v| = %5.2e, CFL(v) = %5.2f, Pe(v) = %5.2f" %
              (vMax, CFL_v, Pe_v))

        drawnow(animateContoursAndVelocityVectors)

        tOut += dtOut
