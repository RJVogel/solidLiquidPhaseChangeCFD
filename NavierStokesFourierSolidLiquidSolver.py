# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:20:40 2016

@author: Julian Vogel (RJVogel)

Navier Stokes Fourier Solid Liquid Phase Change Solver
"""


# Libraries ====================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sps
import scipy.sparse.linalg as spla
import time
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Configuration and initialization =============================================

# Constants
NAN = np.nan

# Discretization ---------------------------------------------------------------

# Geometry
xmin = 0.
xmax = 0.02
ymin = 0.
ymax = 0.02

# Spatial discretization
dx = 0.0005
dy = 0.0005

# Temporal discretization
tMax = 1
dt = 0.00025

# Upwind discretization
gamma = 1


# Momentum equations -----------------------------------------------------------

# Solver momentum equations?
solveMomentum = True

# Initial conditions
p0 = 0
u0 = 0
v0 = 0

# Boundary conditions

# Wall x-velocity: [[S, N]
#                   [W, E]], NAN = symmetry
uWall = [[0., 0.],
         [0., 0.]]
# Wall y-velocity: [[S, N]
#                   [W, E]], NAN = symmetry
vWall = [[0., 0.],
         [1., 0.]]
# Wall pressure: [[S, N]
#                 [W, E]], NAN = symmetry
pWall = [[NAN, NAN],
         [NAN, NAN]]

# Material properties

# Density
rho = 820.733
# Kinematic viscosity
mu = 0.003543

# Solver: amg, amg_precond_bicgstab, direct
solver = 'amg_precond_bicgstab'
tol_p = 1e-6
tol_U = 1e-6


# Energy equation --------------------------------------------------------------

# Solve energy equation?
solveEnergy = False

# Initial conditions
T0 = 27

# Boundary conditions
# Wall temperature: [[S, N]
#                   [W, E]], NAN = symmetry
Twall = [[NAN, NAN],
         [38, 28]]

# Material properties

# Specific heat capacity
c = 2078.04
# Thermal conductivity
k = 0.151215
# Thermal expansion coefficient
beta = 8.9e-4
# Reference temperature for linearized buoyancy term (Boussinesq)
Tref = 28.

# Physical constants
g = 9.81


# Solid-liquid phase change ----------------------------------------------------

# Solve phase change?
solvePhaseChange = False

# Boundary conditions
# Wall liquid fraction: [[S, N]
#                        [W, E]], NAN = symmetry
fWall = [[NAN, NAN],
         [1, NAN]]

# Material properties

# Melting temperature
Tm = 28.
# Latent heat
L = 242454.

# Model parameters

# Mushy region temperature range
dTm = 0.2
# Mushy region constant
Cmush = 1e9


# Visualization ----------------------------------------------------------------

# Output control

# Print step
dtOut = max(dt, tMax/10)
# Plot step
dtPlot = dtOut

# Plots

# Figure size on screen
figureSize = (7, 6)
# Use inline graphics
inlineGraphics = True

# Plot definition
plotContourVar = 'p'  # 'p', u, v, 'U', 'divU', 'T'
plotFaceValues = False
plotVelocityVectorsEvery = 1
plotLevels = (None, None)
colormap = 'bwr'

# Save to file
saveFiguresToFile = False
figureFilename = 'liddrivencavity_Re20'

# Profile
writeProfile = False
profileFilename = ''
profilePoint = [0.5, 0.5]
profileDirection = 0


# Functions ====================================================================


def avg(array, axis):

    # Compute average
    if len(array.shape) == 1:
        return (array[:-1]+array[1:])/2
    elif axis == 0:
        return (array[:-1, :]+array[1:, :])/2
    elif axis == 1:
        return (array[:, :-1]+array[:, 1:])/2
    else:
        return np.nan


def animateContoursAndVelocityVectors(inlineGraphics, plotContourVar,
                                      plotFaceValues, plotLevels,
                                      plotVelocityVectorsEvery, figureSize,
                                      fig=None, ax=None):

    # Font
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 12}
    plt.rc('font', **font)

    if fig is None or inlineGraphics:
        # Create figure
        fig = plt.figure(figsize=figureSize, dpi=72)
        # Create axis
        ax1 = fig.add_subplot(111)
        ax1.set_aspect(1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

    # Select plot variable
    if plotContourVar == 'p':
        if plotFaceValues:
            var = pcorn
        else:
            var = p[1:-1, 1:-1]
        label = 'p / Pa'
    elif plotContourVar == 'u':
        if plotFaceValues:
            var = ucorn
        else:
            var = uc[1:-1, :]
        label = 'u / m/s'
    elif plotContourVar == 'v':
        if plotFaceValues:
            var = vcorn
        else:
            var = vc[:, 1:-1]
        label = 'v / m/s'
    elif plotContourVar == 'U':
        if plotFaceValues:
            var = Ucorn
        else:
            var = Uc
        label = 'U / m/s'
    elif plotContourVar == 'divU':
        var = divU
        label = 'div (U) / 1/s'
        plotFaceValues = False
    elif plotContourVar == 'T':
        if plotFaceValues:
            var = Tcorn
        else:
            var = T[1:-1, 1:-1]
        label = 'T / °C'
    elif plotContourVar == 'f':
        if plotFaceValues:
            var = fcorn
        else:
            var = f[1:-1, 1:-1]
        label = 'f'

    # Plot levels
    if plotLevels == (None, None):
        plotLevels = (np.min(var)-1e-12, np.max(var)+1e-12)

    # Plot contours/pcolor
    plotContourLevels = np.linspace(plotLevels[0], plotLevels[1], num=40)
    if plotFaceValues:
        ctf1 = ax1.contourf(Xf, Yf, var, plotContourLevels, cmap=colormap,
                            extend='both')
    else:
        ctf1 = ax1.pcolor(Xf, Yf, var, vmin=plotLevels[0], vmax=plotLevels[1],
                          cmap=colormap)

    # Velocity vectors
    if plotVelocityVectorsEvery >= 1 and \
            np.any(np.abs(uc) > 1e-6) and np.any(np.abs(vc) > 1e-6):
        m = plotVelocityVectorsEvery
        m0 = int(np.ceil(m/2)-1)
        ax1.quiver(X[1:-1, 1:-1][m0::m, m0::m], Y[1:-1, 1:-1][m0::m, m0::m],
                   uc[1:-1, :][m0::m, m0::m],
                   vc[:, 1:-1][m0::m, m0::m])

    if fig is None or inlineGraphics:
        # Colorbar
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        ticks1 = np.linspace(plotLevels[0], plotLevels[1], num=11)
        cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True, ticks=ticks1)
        cBar1.set_label(label)

    plt.tight_layout()
    plt.show()

    return fig, ax


def buildPoissonEquation(nx, ny, dx, dy, wallBC,
                         dirichletBetweenNodes=(False, False)):

    # System matrix

    # Identity matrices
    Ix = sps.eye(nx)
    Iy = sps.eye(ny)
    # 1D coefficient matrices x- and y-direction
    Ax1d = buildSecondDerivative1d(nx, 1, wallBC, dirichletBetweenNodes)/(dx*dx)
    Ay1d = buildSecondDerivative1d(ny, 0, wallBC, dirichletBetweenNodes)/(dy*dy)
    # Full 2d coefficient matrices x- and y-direction
    Ax = sps.kron(Iy, Ax1d)
    Ay = sps.kron(Ay1d, Ix)
    # Full 2d coefficient matrix
    A = sps.csr_matrix(Ax + Ay)

    # Set Dirichlet BC: Constant part goes in RHS
    rhs_bound = np.zeros((ny, nx))
    # West
    if not np.isnan(wallBC[1][0]):
        if dirichletBetweenNodes[1]:
            rhs_bound[:, 0] = 2*wallBC[1][0]/(dx*dx)
        else:
            rhs_bound[:, 0] = 1*wallBC[1][0]/(dx*dx)
    # East
    if not np.isnan(wallBC[1][1]):
        if dirichletBetweenNodes[1]:
            rhs_bound[:, -1] = 2*wallBC[1][1]/(dx*dx)
        else:
            rhs_bound[:, -1] = 1*wallBC[1][1]/(dx*dx)
    # South
    if not np.isnan(wallBC[0][0]):
        if dirichletBetweenNodes[0]:
            rhs_bound[0, :] = 2*wallBC[0][0]/(dy*dy)
        else:
            rhs_bound[0, :] = 1*wallBC[0][0]/(dy*dy)
    # North
    if not np.isnan(wallBC[0][1]):
        if dirichletBetweenNodes[0]:
            rhs_bound[-1, :] = 2*wallBC[0][1]/(dy*dy)
        else:
            rhs_bound[-1, :] = 1*wallBC[0][1]/(dy*dy)

    return A, rhs_bound


def buildSecondDerivative1d(n, direction, wallBC, dirichletBetweenNodes):
    # Poisson Matrix 1D

    # Neighbor coefficients
    en = -1*np.ones((1, n))
    # Point coefficients
    ep = 2*np.ones((1, n))
    # South/West boundary
    if np.isnan(wallBC[direction][0]):
        ep[:, 0] += -1  # Neumann boundary condition
    elif dirichletBetweenNodes[direction]:
        ep[:, 0] += 1  # Dirichlet BC for boundary between nodes
    # North/East boundary
    if np.isnan(wallBC[direction][1]):
        ep[:, -1] += -1  # Neumann boundary condition
    elif dirichletBetweenNodes[direction]:
        ep[:, -1] += 1  # Dirichlet BC for boundary between nodes
    # return 1D coefficient matrix
    return sps.spdiags((ep.ravel(), en.ravel(), en.ravel()),
                       [0, 1, -1], (n), (n))


def solveMomentumEquation(
        us, vs, un, vn, uc, vc, ucorn, vcorn, T, Tref, dt, dx, dy,
        nu, beta, g, A_u, A_v, rhs_u, rhs_v, rhs_u_bound, rhs_v_bound,
        solver, tol, ml_u, ml_v, M_u, M_v, solveEnergy, gamma,
        solvePhaseChange, Bu, Bv):

    udh = np.diff(us, axis=1)/2
    udv = np.diff(us, axis=0)/2
    vdv = np.diff(vs, axis=0)/2
    vdh = np.diff(vs, axis=1)/2

    # Non-linear terms
    Fu = 1/(dx)*np.diff(uc[1:-1, :]*uc[1:-1, :] -
                        gamma*np.abs(uc[1:-1, :])*udh[1:-1, :], axis=1) + \
        1/(dy)*np.diff(vcorn[:, 1:-1]*ucorn[:, 1:-1] -
                       gamma*np.abs(vcorn[:, 1:-1]*udv[:, 1:-1]), axis=0)
    Fv = 1/(dx)*np.diff(ucorn[1:-1, :]*vcorn[1:-1, :] -
                        gamma*np.abs(ucorn[1:-1, :])*vdh[1:-1, :], axis=1) + \
        1/(dy)*np.diff(vc[:, 1:-1]*vc[:, 1:-1] -
                       gamma*np.abs(vc[:, 1:-1])*vdv[:, 1:-1], axis=0)

    if solvePhaseChange:
        # Velocity switch off
        A_u = A_u + sps.spdiags(
            dt*Bu[1:-1, 1:-1].ravel(), 0, ny*(nx-1), ny*(nx-1))
        A_v = A_v + sps.spdiags(
            dt*Bv[1:-1, 1:-1].ravel(), 0, (ny-1)*nx, (ny-1)*nx)

    # Right hand side
    rhs_u[:, :] = un[1:-1, 1:-1] - dt*Fu
    rhs_v[:, :] = vn[1:-1, 1:-1] - dt*Fv

    # Boundary conditions
    rhs_u = rhs_u + rhs_u_bound
    rhs_v = rhs_v + rhs_v_bound

#    if solvePhaseChange:
#        rhs_u[:, :] = rhs_u[:, :] + un[1:-1, 1:-1]*dt*Bu[1:-1, 1:-1]
#        rhs_v[:, :] = rhs_v[:, :] + vn[1:-1, 1:-1]*dt*Bv[1:-1, 1:-1]

    if solveEnergy:
        # Interpolate temperature on staggered y face positions
        Tfy = avg(T, 0)
        # Buoyancy
        rhs_v[:, :] = rhs_v[:, :] + dt*beta*(Tfy[1:-1, 1:-1]-Tref)*g

    # Solve linear systems
    if solver == 'amg_precond_bicgstab':
        [us[1:-1, 1:-1].flat[:], info] = spla.bicgstab(
            A_u, rhs_u.ravel(), x0=u[1:-1, 1:-1].ravel(), tol=tol, M=M_u)
        [vs[1:-1, 1:-1].flat[:], info] = spla.bicgstab(
            A_v, rhs_v.ravel(), x0=v[1:-1, 1:-1].ravel(), tol=tol, M=M_v)
    elif solver == 'amg':
        us[1:-1, 1:-1].flat[:] = ml_u.solve(rhs_u.ravel(), tol=tol)
        vs[1:-1, 1:-1].flat[:] = ml_v.solve(rhs_v.ravel(), tol=tol)
    else:
        us[1:-1, 1:-1].flat[:] = spla.spsolve(A_u, rhs_u.ravel())
        vs[1:-1, 1:-1].flat[:] = spla.spsolve(A_v, rhs_v.ravel())

    return us, vs, rhs_u, rhs_v


def correctPressure(us, vs, p, T, A, rhs_p, rhs_p_bound, rho, g, beta,
                    dt, dx, dy, solver, tol, ml, M, solveEnergy,
                    solvePhaseChange, Bu, Bv):

    # Poisson equation
    # Right hand side
    rhs_p[:, :] = -rho/dt*(np.diff(us[1:-1, :], axis=1)/dx +
                           np.diff(vs[:, 1:-1], axis=0)/dy)

    # Boundary conditions
    rhs_p[:, :] = rhs_p[:, :] + rhs_p_bound[:, :]

    # Solve linear system
    if solver == 'amg_precond_bicgstab':
        [p[1:-1, 1:-1].flat[:], info] = spla.bicgstab(
            A, rhs_p.ravel(), x0=p[1:-1, 1:-1].ravel(), tol=tol, M=M)
    elif solver == 'amg':
        p[1:-1, 1:-1].flat[:] = ml.solve(rhs_p.ravel(), tol=tol)
    else:
        p[1:-1, 1:-1].flat[:] = spla.spsolve(A, rhs_p.ravel())

    # Pressure gradients
    dpdx = - dt/rho * (p[1:-1, 2:-1]-p[1:-1, 1:-2])/(dx)
    dpdy = - dt/rho * (p[2:-1, 1:-1]-p[1:-2, 1:-1])/(dy)

#    if solvePhaseChange:
#        # Correct for velocity switch-off
#        dpdx = dpdx/(1+dt*Bu[1:-1, 1:-1])
#        dpdy = dpdy/(1+dt*Bv[1:-1, 1:-1])

    # Project corrected pressure onto velocity field
    u[1:-1, 1:-1] = us[1:-1, 1:-1] + dpdx
    v[1:-1, 1:-1] = vs[1:-1, 1:-1] + dpdy

    return u, v, p, rhs_p


def solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, gamma,
                        solvePhaseChange, cMod):

    # Interpolate temperature on staggered x and y face positions
    Tfx = avg(T, 1)
    Tfy = avg(T, 0)
    Tdx = np.diff(T, axis=1)/2
    Tdy = np.diff(T, axis=0)/2

    uT = u[1:-1, :]*Tfx[1:-1, :] - gamma*np.abs(u[1:-1, :])*Tdx[1:-1, :]
    vT = v[:, 1:-1]*Tfy[:, 1:-1] - gamma*np.abs(v[:, 1:-1])*Tdy[:, 1:-1]

    dT = - dt*(1/dx*np.diff(uT, axis=1) + 1/dy*np.diff(vT, axis=0)) \
        + dt*a*(1/dx**2*(Tn[1:-1, 2:]-2*Tn[1:-1, 1:-1]+Tn[1:-1, :-2]) +
                1/dy**2*(Tn[2:, 1:-1]-2*Tn[1:-1, 1:-1]+Tn[:-2, 1:-1]))

    if solvePhaseChange:
        dT = dT/cMod[1:-1, 1:-1]

    # Solve energy equation
    T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + dT

    return T


def calcEnthalpyPorosity(T, Tm, dTm, L, c, solveMomentum, Cmush, Bu, Bv):

    # Calculate liquid phase fraction
    f = (T-Tm)/dTm+0.5
    f = np.minimum(f, 1)
    f = np.maximum(f, 0)

    # Find phase change cells
    pc = np.logical_and(f > 0, f < 1)

    # Set heat capacity modifier
    cMod[pc] = 1+L/(c*dTm)
    cMod[np.logical_not(pc)] = 1

    if solveMomentum:
        # Velocity switch-off constant
        q = 1e-3
        B = Cmush*(1-f)**2/(f**3+q)
        Bu, Bv = avg(B, 1), avg(B, 0)
    else:
        Bu[:, :] = np.nan
        Bv[:, :] = np.nan

    return f, pc, cMod, Bu, Bv


def interpolateVelocities(u, v, uWall, vWall):

    # West
    if np.isnan(vWall[1][0]):
        v[:, 0] = v[:, 1]  # symmetry
    else:
        v[:, 0] = 2*vWall[1][0] - v[:, 1]  # wall
    # East
    if np.isnan(vWall[1][1]):
        v[:, -1] = v[:, -2]  # symmetry
    else:
        v[:, -1] = 2*vWall[1][1] - v[:, -2]  # wall
    # South
    if np.isnan(uWall[0][0]):
        u[0, :] = u[1, :]  # symmetry
    else:
        u[0, :] = 2*uWall[0][0] - u[1, :]  # wall
    # North
    if np.isnan(uWall[0][1]):
        u[-1, :] = u[-2, :]  # symmetry
    else:
        u[-1, :] = 2*uWall[0][1] - u[-2, :]  # wall

    # Internal cell centered and cell corner values
    uc = avg(u, 1)  # horizontal average lives in cell centers
    vc = avg(v, 0)  # vertical average lives in cell centers
    ucorn = avg(u, 0)  # vertical average lives in cell corners
    vcorn = avg(v, 1)  # horizontal average lives in cell corners

    return u, v, uc, vc, ucorn, vcorn


def interpolateCellCenteredOnBoundary(var, varWall):

    # West
    if np.isnan(varWall[1][0]):
        var[1:-1, 0] = var[1:-1, 1]
    else:
        var[:, 0] = 2*varWall[1][0] - var[:, 1]
    # East
    if np.isnan(varWall[1][1]):
        var[1:-1, -1] = var[1:-1, -2]
    else:
        var[:, -1] = 2*varWall[1][1] - var[:, -2]
    # South
    if np.isnan(varWall[0][0]):
        var[0, 1:-1] = var[1, 1:-1]
    else:
        var[0, :] = 2*varWall[0][0] - var[1, :]
    # North
    if np.isnan(varWall[0][1]):
        var[-1, 1:-1] = var[-2, 1:-1]
    else:
        var[-1, :] = 2*varWall[0][1] - var[-2, :]

    return var


def interpolateCellCenteredOnCorners(var, varWall):

    # Boundary corners interpolated with mean of x- and y- 2nd order backward
    var[0, 0] = (2*var[0, 1] - var[0, 2] + 2*var[1, 0] - var[2, 0])/2
    var[0, -1] = (2*var[0, -2] - var[0, -3] + 2*var[1, -1] - var[2, -1])/2
    var[-1, 0] = (2*var[-1, 1] - var[-1, 2] + 2*var[-2, 0] - var[-3, 0])/2
    var[-1, -1] = (2*var[-1, -2] - var[-1, -3] + 2*var[-2, -1] - var[-3, -1])/2

    # All cell corners except boundary corners
    varcorn = avg(avg(var, 0), 1)

    return var, varcorn


def calcDerived(Xf, Yf, p, u, v, dx, dy, nu, a, dt):

    # Velocity magnitudes
    Uc = (uc[1:-1, :]**2+vc[:, 1:-1]**2)**0.5
    Ucorn = (ucorn**2+vcorn**2)**0.5
    uMax = np.max(np.abs(ucorn))
    vMax = np.max(np.abs(vcorn))
    # Divergence
    divU = np.diff(u[1:-1, :], axis=1)/np.diff(Xf[1:, :], axis=1) + \
        np.diff(v[:, 1:-1], axis=0)/np.diff(Yf[:, 1:], axis=0)
    # Peclet number viscous
    Pe_nu_u = uMax*dx/nu
    Pe_nu_v = vMax*dy/nu
    # Courant Friedrichs Levy number
    CFL_u = uMax*dt/dx
    CFL_v = vMax*dt/dy
    # Viscous time step constraint
    Vis_x = dt*(2*nu)/dx**2
    Vis_y = dt*(2*nu)/dy**2
    # Peclet number temp diff
    Pe_a_u = uMax*dx/a
    Pe_a_v = vMax*dy/a
    # Fourier numbers
    Fo_x = dt*(2*a)/dx**2
    Fo_y = dt*(2*a)/dy**2

    return Uc, Ucorn, uMax, vMax, divU, Pe_nu_u, Pe_nu_v, \
        Pe_a_u, Pe_a_v, CFL_u, CFL_v, Vis_x, Vis_y, Fo_x, Fo_y


# PREPROCESSING ================================================================

# Mesh generation---------------------------------------------------------------

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
# Print
print("Mesh generated with %d x %d = %d internal centered nodes" %
      (nx, ny, nx*ny))

# Initial values----------------------------------------------------------------

# Variables
p = p0*np.ones((ny+2, nx+2))
u = u0*np.ones((ny+2, nx+1))
v = v0*np.ones((ny+1, nx+2))
T = T0*np.ones((ny+2, nx+2))
# Material properties
nu = mu/rho
a = k/(rho*c)
# Dimensionless numbers
Pr = nu/a
Tb = np.array(Twall)[~np.isnan(Twall)]
if solvePhaseChange:
    Tb = np.append(Tb, Tm)
RaLx = g*beta*(np.max(Tb)-np.min(Tb))*(xmax-xmin)**3/(nu*a)
RaLy = g*beta*(np.max(Tb)-np.min(Tb))*(ymax-ymin)**3/(nu*a)
# Constant boundaries
# West
u[:, 0] = uWall[1][0]
# East
u[:, -1] = uWall[1][1]
# South
v[0, :] = vWall[0][0]
# North
v[-1, :] = vWall[0][1]
# Interpolate velocity internal and boundary values for non-linear terms
[u, v, uc, vc, ucorn, vcorn] = interpolateVelocities(u, v, uWall, vWall)
# Interpolate temperature on boundaries for non-linear and source terms
T = interpolateCellCenteredOnBoundary(T, Twall)
# Liquid phase fraction
f = (T > Tm).astype(float)
# Modified heat capacity
cMod = np.ones((ny+2, nx+2))
# Velocity switch off terms
Bu = np.zeros((ny+2, nx+1))
Bv = np.zeros((ny+1, nx+2))
# Right hand sides
rhs_p = np.zeros((ny, nx))
rhs_u = np.zeros((ny, nx-1))
rhs_v = np.zeros((ny-1, nx))
# Right hand sides boundary part
rhs_p_bound = np.zeros((ny, nx))
rhs_u_bound = np.zeros((ny, nx-1))
rhs_v_bound = np.zeros((ny-1, nx))

# Generate system matrices------------------------------------------------------

if solveMomentum:
    # Poisson equation for pressure
    [A_p, rhs_p_bound] = buildPoissonEquation(nx, ny, dx, dy,
                                              pWall, (True, True))
    # Zero pressure at boundary nodes in SW corner, if only Neumann boundaries
    if np.all(np.isnan(pWall)):
        A_p[0, 0] = 3/2*A_p[0, 0]

    # Poisson equation for u velocity
    [A_u, rhs_u_bound] = buildPoissonEquation(nx-1, ny, dx, dy,
                                              uWall, (True, False))
    # Build equation system u** - nu*dt*laplace(u**) = u*
    A_u = sps.eye((nx-1)*ny) + dt*nu*A_u
    rhs_u_bound = dt*nu*rhs_u_bound

    # Poisson equation for v velocity
    [A_v, rhs_v_bound] = buildPoissonEquation(nx, ny-1, dx, dy,
                                              vWall, (False, True))
    # Build equation system v** - nu*dt*laplace(v**) = v*
    A_v = sps.eye(nx*(ny-1)) + dt*nu*A_v
    rhs_v_bound = dt*nu*rhs_v_bound

    # Algebraic Multigrid (AMG) as preconditioner
    if solver in ['amg', 'amg_precond_bicgstab']:
        import pyamg
    if solver in ['amg', 'amg_precond_bicgstab']:
        ml_p = pyamg.smoothed_aggregation_solver(A_p, max_coarse=10)
        ml_u = pyamg.smoothed_aggregation_solver(A_u, max_coarse=10)
        ml_v = pyamg.smoothed_aggregation_solver(A_v, max_coarse=10)
    if solver == 'amg_precond_bicgstab':
        M_p = ml_p.aspreconditioner(cycle='V')
        M_u = ml_u.aspreconditioner(cycle='V')
        M_v = ml_v.aspreconditioner(cycle='V')


# Time stepping ================================================================

twall0 = time.time()
tOut = dtOut
tPlot = dtPlot
fig, ax = None, None
t, n = 0, 0

while t - tMax < -1e-9:

    n += 1
    t += dt

    if solveMomentum:

        # Update variables
        pn = p.copy()
        un, vn, us, vs = u.copy(), v.copy(), u.copy(), v.copy()

        # Projection method: Intermediate velocity field U*
        [us, vs, rhs_u, rhs_v] = solveMomentumEquation(
            us, vs, un, vn, uc, vc, ucorn, vcorn, T, Tref, dt, dx, dy,
            nu, beta, g, A_u, A_v, rhs_u, rhs_v, rhs_u_bound, rhs_v_bound,
            solver, tol_U, ml_u, ml_v, M_u, M_v, solveEnergy, gamma,
            solvePhaseChange, Bu, Bv)

        # Projection method: Pressure correction
        [u, v, p, rhs_p] = correctPressure(
            us, vs, p, T, A_p, rhs_p, rhs_p_bound,
            rho, g, beta, dt, dx, dy, solver, tol_p, ml_p, M_p, solveEnergy,
            solvePhaseChange, Bu, Bv)

        # Interpolate velocity on internal & boundary values f. non-linear terms
        [u, v, uc, vc, ucorn, vcorn] = interpolateVelocities(u, v, uWall, vWall)

    if solveEnergy:

        # Advance temperature
        Tn = T.copy()

        # Solve energy equation
        T = solveEnergyEquation(T, Tn, u, v, dt, dx, dy, a, gamma,
                                solvePhaseChange, cMod)

        # Interpolate temperature on boundaries for non-linear and source terms
        T = interpolateCellCenteredOnBoundary(T, Twall)

        if solvePhaseChange:

            # Enthalpy porosity method
            [f, pc, cMod, Bu, Bv] = calcEnthalpyPorosity(
                T, Tm, dTm, L, c, solveMomentum, Cmush, Bu, Bv)

    # Print output step --------------------------------------------------------

    if (t-tOut) > -1e-6:

        if solveMomentum:
            # Calculate residuals
            Res_u = np.linalg.norm(rhs_u.ravel()-A_u*us[1:-1, 1:-1].ravel()) / (
                np.linalg.norm(rhs_u.ravel())+1e-12)
            Res_v = np.linalg.norm(rhs_v.ravel()-A_v*vs[1:-1, 1:-1].ravel()) / (
                np.linalg.norm(rhs_v.ravel())+1e-12)
            Res_p = np.linalg.norm(rhs_p.ravel()-A_p*p[1:-1, 1:-1].ravel()) / (
                np.linalg.norm(rhs_p.ravel())+1e-12)

        # Calculate derived quantities
        [Uc, Ucorn, uMax, vMax, divU, Pe_nu_u, Pe_nu_v,
         Pe_a_u, Pe_a_v, CFL_u, CFL_v, Vis_x, Vis_y, Fo_x, Fo_y] = \
            calcDerived(Xf, Yf, p, u, v, dx, dy, nu, a, dt)

        print("==============================================================")
        print("Time step n = %d, t = %8.3f, dt = %4.1e, t_wall = %4.1f" %
              (n, t, dt, time.time()-twall0))
        if solveMomentum:
            print("max|u| = %4.2e, CFL(u) = %5.2f, "
                  "Pe(u) = %5.2f, Vis(x) = %5.2f" %
                  (uMax, CFL_u, Pe_nu_u, Vis_x))
            print("max|v| = %4.2e, CFL(v) = %5.2f, "
                  "Pe(v) = %5.2f, Vis(y) = %5.2f" %
                  (vMax, CFL_v, Pe_nu_v, Vis_y))
            print("Res(p) = %5.2e, Res(u) = %5.2e, Res(v) = %5.2e" %
                  (Res_p, Res_u, Res_v))
            print("ddt(p) = %5.2e, ddt(u) = %5.2e, ddt(v) = %5.2e" %
                  (np.linalg.norm((p-pn)/dt),
                   np.linalg.norm((u-un)/dt),
                   np.linalg.norm((v-vn)/dt)))
        if solveEnergy:
            print("Fo(x) = %5.2f, Fo(y) = %5.2f, "
                  "Pe(a,u) = %5.2f, Pe(a,v) = %5.2f" %
                  (Fo_x, Fo_y, Pe_a_u, Pe_a_v))
            print("Pr = %5.2f, Ra(Lx) = %4.1e, Ra(Ly) = %4.1e" %
                  (Pr, RaLx, RaLy))

        tOut += dtOut

    # Plot and save graphs and profiles ----------------------------------------

    if (t-tPlot) > -1e-6:

        if solveMomentum:
            # Interpolate pressure on corners and boundaries for plotting
            p = interpolateCellCenteredOnBoundary(p, pWall)
            [p, pcorn] = interpolateCellCenteredOnCorners(p, pWall)

        if solveEnergy:
            # Interpolate temperature on corners for plotting
            [T, Tcorn] = interpolateCellCenteredOnCorners(T, Twall)

        if solvePhaseChange:
            # Interpolate temperature on corners for plotting
            [f, fcorn] = interpolateCellCenteredOnCorners(f, fWall)

        # Plot
        [fig, ax] = animateContoursAndVelocityVectors(
            inlineGraphics, plotContourVar, plotFaceValues, plotLevels,
            plotVelocityVectorsEvery, figureSize, fig, ax)
        fig.canvas.draw()
        fig.canvas.flush_events()

        if saveFiguresToFile:
            formattedFilename = '{0}_{1:5.3f}.png'.format(figureFilename, t)
            path = Path('out') / formattedFilename
            fig.savefig(path, dpi=144)

        if writeProfile:
            if profileDirection == 0:
                dataIndices = Xf == profilePoint[1]
            else:
                dataIndices = Yf == profilePoint[0]
            prof = np.stack((Xf[dataIndices], Yf[dataIndices],
                             pcorn[dataIndices], ucorn[dataIndices],
                             vcorn[dataIndices]), axis=-1)
            np.savetxt('out/' + profileFilename + '.xy', prof,
                       fmt='%.9f', delimiter="\t")

        tPlot += dtPlot
