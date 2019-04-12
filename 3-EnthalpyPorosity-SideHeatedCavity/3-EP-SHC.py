# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:20:40 2016

@author: Julian Vogel (AkaDrBird)

Enthalpy porosity solver for solid liquid phase change in side heated cavity
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from drawnow import drawnow
from pathlib import Path
from scipy import sparse as sp
import scipy.sparse.linalg as spla

# Output
saveFiguresToFile = True
outputFilename = 'anim'

# Geometry
xmin = 0.
xmax = 0.025
ymin = 0.
ymax = 0.1

# Spatial discretization
dx = 0.001
dy = 0.004

# Initial conditions
T0 = 27.

# Boundary conditions
# Wall temperature: [W, E, S, N], np.nan = symmetry
Twall = [38, np.nan, np.nan, np.nan]
# Wall x-velocity: [W, E, S, N], np.nan = symmetry
uWall = [0., 0., 0., np.nan]
# Wall y-velocity: [W, E, S, N], np.nan = symmetry
vWall = [0., 0., 0., 0.]
# Pressure: [W, E, S, N], np.nan = symmetry
pWall = [np.nan, np.nan, np.nan, 0]

# Physical constants
g = 9.81

# Material properties
rho = 820.733
c = 2078.04
k = 0.151215
a = k/(rho*c)
mu = 0.003543
nu = mu/rho
beta = 8.9e-4
Tref = 28.
Tm = 28.
L = 242454.

# Model parameters
dTm = 0.2
Cmush = 1e6

# Temporal discretization
tMax = 10800.
sigma = 0.25

# Solver settings
poissonSolver = 'iterative'
nit = 50  # iterations of pressure poisson equation

# Visualization
dtOut = 60.0  # output step length
nOut = int(round(tMax/dtOut))
figureSize = (10., 6.25)
minContour1 = 27.
maxContour1 = 38.
colormap1 = 'coolwarm'
plotContourLevels1 = np.linspace(minContour1, maxContour1, num=23)
ticks1 = np.linspace(minContour1, maxContour1, num=12)
minContour2 = 0.
maxContour2 = 0.005
colormap2 = 'jet'
plotContourLevels2 = np.linspace(minContour2, maxContour2, num=21)
ticks2 = np.linspace(minContour2, maxContour2, num=11)

# Initial time step calculation
dtVisc = min(dx**2/(2*nu), dy**2/(2*nu))  # Viscous time step
dtThrm = min(dx**2/a, dy**2/a)  # Thermal time step
dt0 = sigma*min(dtVisc, dtThrm)
nsignifdigs = -np.floor(np.log10(abs(dtOut/dt0)))
dt = dtOut/(np.ceil(dtOut/dt0*10**nsignifdigs)/(10**nsignifdigs))

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
f = T > Tm
f = f.astype(float)
cMod = np.ones((ny, nx))
B = np.zeros((ny, nx))

CP = np.zeros((ny, nx))
CW = np.zeros((ny, nx))
CE = np.zeros((ny, nx))
CS = np.zeros((ny, nx))
CN = np.zeros((ny, nx))
A = sp.csr_matrix((nx*ny, nx*ny))

# Functions


def animateContoursAndVelocityVectors():

    # plot temperature and pressure
    # Axis
    ax1 = fig.add_subplot(121)
    ax1.set_aspect(1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Temperature and pressure contours')
    # Filled contours for temperature
    ctf1 = ax1.contourf(X, Y, T, plotContourLevels1, extend='both',
                        alpha=1, linestyles=None, cmap=colormap1)
    # Contours for pressure
    ct1 = ax1.contour(X, Y, p, levels=20,
                      colors='black', linewidths=1, linestyles='dotted')
    plt.clabel(ct1, ct1.levels[::2], fmt='%1.1e', fontsize='smaller')
    # Colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True, ticks=ticks1)
    cBar1.set_label('T / °C')

    # Plot liquid fraction and velocity
    ax2 = fig.add_subplot(122)
    ax2.set_aspect(1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Liquid fraction and velocity')
    # Filled contours
    ax2.contourf(X, Y, f, np.linspace(0, 1, num=11), extend='both',
                 alpha=1, linestyles=None, cmap='gray')
    # plot velocity
    m = 1
    qv = ax2.quiver(X,
                    Y,
                    u,
                    v,
                    U,
                    clim=np.array([min(plotContourLevels2),
                                   max(plotContourLevels2)]),
                    cmap=colormap2)
    # Colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cBar2 = fig.colorbar(qv, cax=cax2, extendrect=True, ticks=ticks2)
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


def calcVelocitySwitchOff(u, v, B, dt, f, Cmush):

    # Velocity switch-off constant
    q = 1e-3
    B[1:-1, 1:-1] = Cmush*(1-f[1:-1, 1:-1])**2/(f[1:-1, 1:-1]**3+q)

    # Velocity switch off
    u[1:-1, 1:-1] = u[1:-1, 1:-1]/(1+dt*B[1:-1, 1:-1])
    v[1:-1, 1:-1] = v[1:-1, 1:-1]/(1+dt*B[1:-1, 1:-1])

    return u, v, B


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


def buildPoissonRightHandSide(b, rho, dt, u, v, dx, dy, beta, g, T):

    # Right hand side
    b[1:-1, 1:-1] = rho*(
        1/dt*((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) +
              (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))
        - (((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 +
           2*((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy) *
              (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) +
           ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2)
        + beta*g*(T[2:, 1:-1] - T[0:-2, 1:-1])/(2*dy))

    return b


def solvePoissonEquation(p, b, dx, dy, nit, pWall, solver,
                         CP, CW, CE, CS, CN, A, nx, ny):

    if solver == 'direct':

        # Inner nodes
        CP[1:-1, 1:-1] = 2*(1/dx**2+1/dy**2)
        CW[1:-1, 1:-1] = -1/dx**2
        CE[1:-1, 1:-1] = -1/dx**2
        CS[1:-1, 1:-1] = -1/dy**2
        CN[1:-1, 1:-1] = -1/dy**2

        # Boundary conditions
        CP[:, 0] = 1
        CP[:, -1] = 1
        CP[0, :] = 1
        CP[-1, :] = 1
        CW[1:-1, -1] = -1
        CE[1:-1, 0] = -1
        CS[-1, 1:-1] = 0
        CN[0, 1:-1] = -1
        CW[0, -1] = -0.5
        CW[-1, -1] = 0
        CE[0, 0] = -0.5
        CE[-1, 0] = 0
        CS[[-1, -1], [0, -1]] = 0
        CN[[0, 0], [0, -1]] = -0.5

        A = sp.csr_matrix(sp.spdiags((CP.flat[:], CW.flat[:], CE.flat[:],
                                      CS.flat[:], CN.flat[:]),
                                     [0, 1, -1, nx, -(nx)],
                                     ((nx)*(ny)), ((nx)*(ny))).T)

        p.flat[:] = spla.spsolve(A, -b.flat[:], use_umfpack=True)

    elif (solver == 'iterative'):

        for nit in range(50):

            # Reference pressure in upper left corner
            # p[-1, 1] = 0

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


def correctPressure(u, v, p, rho, dt, dx, dy, B):

    u[1:-1, 1:-1] = u[1:-1, 1:-1] - 1/rho*dt/(1+dt*B[1:-1, 1:-1]) * \
        (p[1:-1, 2:]-p[1:-1, 0:-2])/(2*dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - 1/rho*dt/(1+dt*B[1:-1, 1:-1]) * \
        (p[2:, 1:-1]-p[0:-2, 1:-1])/(2*dy)

    return u, v


def calcEnthalpyPorosity(T, Tm, dTm, L, c):

    # Calculate liquid phase fraction
    f = (T-Tm)/dTm+0.5
    f = np.minimum(f, 1)
    f = np.maximum(f, 0)

    # Find phase change cells
    pc = np.logical_and(f > 0, f < 1)

    # Set heat capacity modifier
    cMod[pc] = 1+L/(c*dTm)
    cMod[np.logical_not(pc)] = 1

    return f, pc, cMod


def solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, Twall, cMod):

    # Solve energy equation with modified heat capacity
    T[1:-1, 1:-1] = Tn[1:-1, 1:-1] - 1/cMod[1:-1, 1:-1] * (
        dt/(dx)*(np.maximum(u[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 1:-1]-Tn[1:-1, 0:-2]) +
                 np.minimum(u[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 2:]-Tn[1:-1, 1:-1])) +
        dt/(dy)*(np.maximum(v[1:-1, 1:-1], 0) *
                 (Tn[1:-1, 1:-1]-Tn[0:-2, 1:-1]) +
                 np.minimum(v[1:-1, 1:-1], 0) *
                 (Tn[2:, 1:-1]-Tn[1:-1, 1:-1]))) + \
        1/cMod[1:-1, 1:-1]*a * \
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
uMax = 0
vMax = 0

while t < tMax:

    n += 1

    # Automatic time stepping
    if np.logical_and(uMax > 0, vMax > 0):
        dtConv = min(dx/uMax, dy/vMax)
        dtVisCon = 1/(1/dtVisc+1/dtConv)
        dt0 = sigma*min(dtThrm, dtVisCon)
        nsigdigs = -np.floor(np.log10(abs(dtOut/dt0)))
        dt = dtOut/(np.ceil(dtOut/dt0*10**nsigdigs)/(10**nsigdigs))
        dt = min(dt, tOut-t)

    t += dt

    # Update variables
    pn = p.copy()
    un = u.copy()
    vn = v.copy()
    fn = f.copy()
    Tn = T.copy()

    # Momentum equation with projection method
    # Intermediate velocity field u*
    [u, v] = solveMomentumEquation(u, v, un, vn, dt, dx, dy,
                                   nu, beta, T, Tref, g)
    # Switch off velocities at solid phase
    [u, v, B] = calcVelocitySwitchOff(u, v, B, dt, f, Cmush)
    # Set velocity boundaries
    [u, v] = setVelocityBoundaries(u, v, uWall, vWall)
    # Pressure correction
    b = buildPoissonRightHandSide(b, rho, dt, u, v, dx, dy, beta, g, T)
    p = solvePoissonEquation(p, b, dx, dy, nit, pWall, poissonSolver,
                             CP, CW, CE, CS, CN, A, nx, ny)
    [u, v] = correctPressure(u, v, p, rho, dt, dx, dy, B)
    # [u, v] = setVelocityBoundaries(u, v, uWall, vWall)

    # Energy equation
    T = solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, Twall, cMod)

    # Enthalpy porosity method
    [f, pc, cMod] = calcEnthalpyPorosity(T, Tm, dTm, L, c)

    # Output
    if (t-tOut) > -1e-6:

        t = tOut

        # Calculate derived quantities
        # Velocities
        U = (u**2+v**2)**0.5
        uMax = np.max(np.abs(u))
        vMax = np.max(np.abs(v))
        # Peclet numbers
        Pe_u = uMax*dx/nu
        Pe_v = vMax*dy/nu
        # Courant Friedrichs Levy numbers
        CFL_u = uMax*dt/dx
        CFL_v = vMax*dt/dy
        # Rayleigh numbers
        if np.any(f > 0):
            dTmaxLiq = np.max(T[f > 0])-np.min(T[f > 0])
        else:
            dTmaxLiq = 0
        RaW = g*beta*dTmaxLiq*(xmax-xmin)**3/(nu*a)
        RaH = g*beta*dTmaxLiq*(ymax-ymin)**3/(nu*a)

        print("==============================================================")
        print(" Time step n = %d, t = %8.3f, dt = %4.1e, t_wall = %4.1f" %
              (n, t, dt, time.time()-twall0))
        print(" max|u| = %3.1e, CFL(u) = %3.1f, Pe(u) = %4.1f, RaW = %3.1e" %
              (uMax, CFL_u, Pe_u, RaW))
        print(" max|v| = %3.1e, CFL(v) = %3.1f, Pe(v) = %4.1f, RaH = %3.1e" %
              (vMax, CFL_v, Pe_v, RaH))

        drawnow(animateContoursAndVelocityVectors)

        tOut += dtOut
