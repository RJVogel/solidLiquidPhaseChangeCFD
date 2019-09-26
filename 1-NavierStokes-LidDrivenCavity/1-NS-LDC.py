# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:20:40 2016

@author: Julian Vogel (RJVogel)

Navier Stokes solver for lid driven cavity flow
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
xmax = 1.0
ymin = 0.
ymax = 1.0

# Spatial discretization
dx = 0.05
dy = 0.05

# Temporal discretization
tMax = 5.
dt = 0.05


# Momentum equations -----------------------------------------------------------

# Material properties

# Density
rho = 1.0
# Kinematic viscosity
nu = 0.1/rho

# Boundary conditions

# Wall x-velocity: [[S, N]
#                   [W, E]], NAN = symmetry
uWall = [[0., 1.],
         [0., 0.]]
# Wall y-velocity: [[S, N]
#                   [W, E]], NAN = symmetry
vWall = [[0., 0.],
         [0., 0.]]
# Wall pressure: [[S, N]
#                 [W, E]], NAN = symmetry
pWall = [[NAN, NAN],
         [NAN, NAN]]

# Solver: Use AMG preconditioner and BICGSTAB solver? Otherwise direct solver
# Viscous laplace term in momentum equation
precond_viscous = True
# Poisson equation for pressure
precond_pressure = True


# Visualization ----------------------------------------------------------------

# Output control

# Print step
dtOut = 1.0
# Plot step
dtPlot = tMax  # Plot step length

# Plots

# Figure size on screen
figureSize = (7, 6)
# Use inline graphics
inlineGraphics = True

# Plot definition
plotContourVar = 'p'  # 'p', u, v, 'U', 'divU'
plotFaceValues = True
plotVelocityVectorsEvery = 1
plotLevels = (-2, 2)
colormap = 'bwr'

# Save to file
saveFiguresToFile = False
figureFilename = 'liddrivencavity_Re100'

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
    if plotVelocityVectorsEvery >= 1:
        m = plotVelocityVectorsEvery
        m0 = int(np.ceil(m/2)-1)
        ax1.quiver(X[1:-1, 1:-1][m0::m, m0::m], Y[1:-1, 1:-1][m0::m, m0::m],
                   uc[1:-1, :][m0::m, m0::m]+1e-12,
                   vc[:, 1:-1][m0::m, m0::m]+1e-12)

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


def solveMomentumNonLinearTerms(u, v, un, vn, dt, dx, dy):

    # Interpolated values
    uah = avg(un, 1)  # horizontal average lives in cell centers
    uav = avg(un, 0)  # vertical average lives in cell corners
    vah = avg(vn, 1)  # horizontal average lives in cell corners
    vav = avg(vn, 0)  # vertical average lives in cell centers
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

    return u, v


def solveMomentumViscousTerms(u, v, dt, dx, dy, nu, A_u, A_v, rhs_u, rhs_v,
                              rhs_u_bound, rhs_v_bound, amg_pre, M_u, M_v):

    # Right hand sides
    rhs_u[:, :] = u[1:-1, 1:-1] + rhs_u_bound[:, :]
    rhs_v[:, :] = v[1:-1, 1:-1] + rhs_v_bound[:, :]

    # Solve linear systems
    if amg_pre:
        [u[1:-1, 1:-1].flat[:], info] = spla.bicgstab(
            A_u, rhs_u.ravel(), x0=u[1:-1, 1:-1].ravel(),
            tol=1e-6, maxiter=100, M=M_u)
        [v[1:-1, 1:-1].flat[:], info] = spla.bicgstab(
            A_v, rhs_v.ravel(), x0=v[1:-1, 1:-1].ravel(),
            tol=1e-6, maxiter=100, M=M_v)
    else:
        u[1:-1, 1:-1].flat[:] = spla.spsolve(A_u, rhs_u.ravel())
        v[1:-1, 1:-1].flat[:] = spla.spsolve(A_v, rhs_v.ravel())

#    # u-velocity
#    u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*(
#        nu * (1/dx**2*(u[1:-1, 2:]-2*u[1:-1, 1:-1]+u[1:-1, :-2]) +
#              1/dy**2*(u[2:, 1:-1]-2*u[1:-1, 1:-1]+u[:-2, 1:-1])))
#    # v-velocity
#    v[1:-1, 1:-1] = v[1:-1, 1:-1] + dt*(
#        nu * (1/dx**2*(v[1:-1, 2:]-2*v[1:-1, 1:-1]+v[1:-1, :-2]) +
#              1/dy**2*(v[2:, 1:-1]-2*v[1:-1, 1:-1]+v[:-2, 1:-1])))

    return u, v, rhs_u, rhs_v


def correctPressure(u, v, p, A, rhs, rhs_bound, rho, dt, dx, dy, amg_pre, M):

    # Poisson equation
    # Right hand side
    rhs[:, :] = -rho/dt*(np.diff(u[1:-1, :], axis=1)/dx +
                         np.diff(v[:, 1:-1], axis=0)/dy) + rhs_bound[:, :]
    # Solve linear system
    if amg_pre:
        [p[1:-1, 1:-1].flat[:], info] = spla.bicgstab(
            A, rhs.ravel(), x0=p[1:-1, 1:-1].ravel(),
            tol=1e-9, maxiter=100, M=M)
    else:
        p[1:-1, 1:-1].flat[:] = spla.spsolve(A, rhs.ravel())

    # Project corrected pressure onto velocity field
    u[1:-1, 1:-1] = u[1:-1, 1:-1] - dt/rho * \
        (p[1:-1, 2:-1]-p[1:-1, 1:-2])/(dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - dt/rho * \
        (p[2:-1, 1:-1]-p[1:-2, 1:-1])/(dy)

    return u, v, p, rhs


def setBoundaryConditions(u, v, p, uWall, vWall, pWall):

    # Velocities

    # West
    u[:, 0] = uWall[1][0]
    if np.isnan(vWall[1][0]):
        v[:, 0] = v[:, 1]  # symmetry
    else:
        v[:, 0] = 2*vWall[1][0] - v[:, 1]  # wall
    # East
    u[:, -1] = uWall[1][1]
    if np.isnan(vWall[1][1]):
        v[:, -1] = v[:, -2]  # symmetry
    else:
        v[:, -1] = 2*vWall[1][1] - v[:, -2]  # wall
    # South
    v[0, :] = vWall[0][0]
    if np.isnan(uWall[0][0]):
        u[0, :] = u[1, :]  # symmetry
    else:
        u[0, :] = 2*uWall[0][0] - u[1, :]  # wall
    # North
    v[-1, :] = vWall[0][1]
    if np.isnan(uWall[0][1]):
        u[-1, :] = u[-2, :]  # symmetry
    else:
        u[-1, :] = 2*uWall[0][1] - u[-2, :]  # wall

    # Pressure

    # West
    if np.isnan(pWall[1][0]):
        p[1:-1, 0] = p[1:-1, 1]
    else:
        p[1:-1, 0] = 2*pWall[1][0] - p[1:-1, 1]
    # East
    if np.isnan(pWall[1][1]):
        p[1:-1, -1] = p[1:-1, -2]
    else:
        p[1:-1, -1] = 2*pWall[1][1] - p[1:-1, -2]
    # South
    if np.isnan(pWall[0][0]):
        p[0, 1:-1] = p[1, 1:-1]
    else:
        p[0, 1:-1] = 2*pWall[0][0] - p[1, 1:-1]
    # North
    if np.isnan(pWall[0][1]):
        p[-1, 1:-1] = p[-2, 1:-1]
    else:
        p[-1, 1:-1] = 2*pWall[0][1] - p[-2, 1:-1]
    # Corners interpolated with mean of x- and y- 2nd order backward stencil
    p[0, 0] = (2*p[0, 1] - p[0, 2] + 2*p[1, 0] - p[2, 0]) / 2  # SW
    p[0, -1] = (2*p[0, -2] - p[0, -3] + 2*p[1, -1] - p[2, -1]) / 2  # SE
    p[-1, 0] = (2*p[-1, 1] - p[-1, 2] + 2*p[-2, 0] - p[-3, 0]) / 2  # NW
    p[-1, -1] = (2*p[-1, -2] - p[-1, -3] + 2*p[-2, -1] - p[-3, -1]) / 2  # NE

    return u, v, p


def calcDerived(Xf, Yf, p, u, v, dx, dy, nu, dt):

    # Pressure at cell corners
    pcorn = avg(avg(p, 0), 1)
    # Velocities
    uc = avg(u, 1)
    vc = avg(v, 0)
    Uc = (uc[1:-1, :]**2+vc[:, 1:-1]**2)**0.5
    ucorn = avg(u, 0)
    vcorn = avg(v, 1)
    Ucorn = (ucorn**2+vcorn**2)**0.5
    uMax = np.max(np.abs(ucorn))
    vMax = np.max(np.abs(vcorn))
    # Divergence
    divU = np.diff(u[1:-1, :], axis=1)/np.diff(Xf[1:, :], axis=1) + \
        np.diff(v[:, 1:-1], axis=0)/np.diff(Yf[:, 1:], axis=0)
    # Peclet number
    Pe_u = uMax*dx/nu
    Pe_v = vMax*dy/nu
    # Courant Friedrichs Levy number
    CFL_u = uMax*dt/dx
    CFL_v = vMax*dt/dy
    # Viscous time step constraint
    Vis_x = dt*(2*nu)/dx**2
    Vis_y = dt*(2*nu)/dy**2

    return pcorn, uc, vc, ucorn, vcorn, Uc, Ucorn, uMax, vMax, divU, \
        Pe_u, Pe_v, CFL_u, CFL_v, Vis_x, Vis_y


# PREPROCESSING ================================================================

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

# Variables
p = np.zeros((ny+2, nx+2))
u = np.zeros((ny+2, nx+1))
v = np.zeros((ny+1, nx+2))
# Right hand sides
rhs_p = np.zeros((ny, nx))
rhs_u = np.zeros((ny, nx-1))
rhs_v = np.zeros((ny-1, nx))
# Right hand sides boundary part
rhs_p_bound = np.zeros((ny, nx))
rhs_u_bound = np.zeros((ny, nx-1))
rhs_v_bound = np.zeros((ny-1, nx))

# Generate system matrices

# Poisson equation for pressure
[A_p, rhs_p_bound] = buildPoissonEquation(nx, ny, dx, dy,
                                          pWall, (True, True))
# Set zero pressure at boundary nodes in SW corner, if only Neumann boundaries
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
if precond_pressure or precond_viscous:
    import pyamg
if precond_pressure:
    ml = pyamg.smoothed_aggregation_solver(A_p, max_coarse=10)
    M_p = ml.aspreconditioner(cycle='V')
if precond_viscous:
    ml = pyamg.smoothed_aggregation_solver(A_u, max_coarse=10)
    M_u = ml.aspreconditioner(cycle='V')
    ml = pyamg.smoothed_aggregation_solver(A_v, max_coarse=10)
    M_v = ml.aspreconditioner(cycle='V')

# Calculate derived quantities
[pcorn, uc, vc, ucorn, vcorn, Uc, Ucorn, uMax, vMax, divU,
 Pe_u, Pe_v, CFL_u, CFL_v, Vis_x, Vis_y] = calcDerived(
     Xf, Yf, p, u, v, dx, dy, nu, dt)

# Plot
[fig, ax] = animateContoursAndVelocityVectors(
    inlineGraphics, plotContourVar, plotFaceValues, plotLevels,
    plotVelocityVectorsEvery, figureSize)


# Time stepping ================================================================

twall0 = time.time()
tOut = dtOut
tPlot = dtPlot
t = 0
n = 0

while t < tMax:

    n += 1

    dt = dt
    t += dt

    # Update variables
    pn = p.copy()
    un = u.copy()
    vn = v.copy()

    # Projection method

    # Intermediate velocity field u*
    [u, v] = solveMomentumNonLinearTerms(u, v, un, vn, dt, dx, dy)
    [u, v, rhs_u, rhs_v] = solveMomentumViscousTerms(
        u, v, dt, dx, dy, nu, A_u, A_v, rhs_u, rhs_v,
        rhs_u_bound, rhs_v_bound, precond_viscous, M_u, M_v)
    Res_u = np.linalg.norm(rhs_u.ravel()-A_u*u[1:-1, 1:-1].ravel())
    Res_v = np.linalg.norm(rhs_v.ravel()-A_v*v[1:-1, 1:-1].ravel())
    # Pressure correction
    [u, v, p, rhs_p] = correctPressure(u, v, p, A_p, rhs_p, rhs_p_bound,
                                       rho, dt, dx, dy, precond_pressure, M_p)
    Res_p = np.linalg.norm(rhs_p.ravel()-A_p*p[1:-1, 1:-1].ravel())
    # Update boundary values
    [u, v, p] = setBoundaryConditions(u, v, p, uWall, vWall, pWall)

    if (t-tOut) > -1e-6:

        # Calculate derived quantities
        [pcorn, uc, vc, ucorn, vcorn, Uc, Ucorn, uMax, vMax, divU,
         Pe_u, Pe_v, CFL_u, CFL_v, Vis_x, Vis_y] = calcDerived(
             Xf, Yf, p, u, v, dx, dy, nu, dt)

        print("==============================================================")
        print(" Time step n = %d, t = %8.3f, dt = %4.1e, t_wall = %4.1f" %
              (n, t, dt, time.time()-twall0))
        print(" max|u| = %5.2e, CFL(u) = %5.2f, Pe(u) = %5.2f, Vis(x) = %5.2f" %
              (uMax, CFL_u, Pe_u, Vis_x))
        print(" max|v| = %5.2e, CFL(v) = %5.2f, Pe(v) = %5.2f, Vis(y) = %5.2f" %
              (vMax, CFL_v, Pe_v, Vis_y))
        print(" Res(p) = %5.2e, Res(u) = %5.2e, Res(v) = %5.2e" %
              (Res_p, Res_u, Res_v))
        print(" ddt(p) = %5.2e, ddt(u) = %5.2e, ddt(v) = %5.2e" %
              (np.linalg.norm((p-pn)/dt),
               np.linalg.norm((u-un)/dt),
               np.linalg.norm((v-vn)/dt)))

        tOut += dtOut

    if (t-tPlot) > -1e-6:

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
