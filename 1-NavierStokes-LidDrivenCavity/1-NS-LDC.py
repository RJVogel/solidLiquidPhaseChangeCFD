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
nu = 0.05/rho

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

# Solver: amg, amg_precond_bicgstab, direct
solver = 'amg_precond_bicgstab'
tol_p = 1e-6
tol_U = 1e-6


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
plotLevels = (-1, 1)
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
        plotLevels = (np.min(var), np.max(var))

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


def solveMomentumNonLinearTerms(us, vs, un, vn, uc, vc, ucorn, vcorn,
                                dt, dx, dy):

    # u*-velocity
    us[1:-1, 1:-1] = un[1:-1, 1:-1] - dt*(
        1/(dx)*(uc[1:-1, 1:]*uc[1:-1, 1:] -
                uc[1:-1, :-1]*uc[1:-1, :-1]) +
        1/(dy)*(vcorn[1:, 1:-1]*ucorn[1:, 1:-1] -
                vcorn[:-1, 1:-1]*ucorn[:-1, 1:-1]))
    # v*-velocity
    vs[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt*(
        1/(dx)*(ucorn[1:-1, 1:]*vcorn[1:-1, 1:] -
                ucorn[1:-1, :-1]*vcorn[1:-1, :-1]) +
        1/(dy)*(vc[1:, 1:-1]*vc[1:, 1:-1] -
                vc[:-1, 1:-1]*vc[:-1, 1:-1]))

    return us, vs


def solveMomentumViscousTerms(us, vs, dt, dx, dy, nu, A_u, A_v, rhs_u, rhs_v,
                              rhs_u_bound, rhs_v_bound, solver, tol, ml_u, ml_v,
                              M_u, M_v):

    # Right hand sides
    rhs_u[:, :] = us[1:-1, 1:-1] + rhs_u_bound[:, :]
    rhs_v[:, :] = vs[1:-1, 1:-1] + rhs_v_bound[:, :]

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


def correctPressure(us, vs, p, A, rhs, rhs_bound, rho, dt, dx, dy,
                    solver, tol, ml, M):

    # Poisson equation
    # Right hand side
    rhs[:, :] = -rho/dt*(np.diff(us[1:-1, :], axis=1)/dx +
                         np.diff(vs[:, 1:-1], axis=0)/dy) + rhs_bound[:, :]
    # Solve linear system
    if solver == 'amg_precond_bicgstab':
        [p[1:-1, 1:-1].flat[:], info] = spla.bicgstab(
            A, rhs.ravel(), x0=p[1:-1, 1:-1].ravel(), tol=tol, M=M)
    elif solver == 'amg':
        p[1:-1, 1:-1].flat[:] = ml.solve(rhs.ravel(), tol=tol)
    else:
        p[1:-1, 1:-1].flat[:] = spla.spsolve(A, rhs.ravel())

    # Project corrected pressure onto velocity field
    u[1:-1, 1:-1] = us[1:-1, 1:-1] - dt/rho * \
        (p[1:-1, 2:-1]-p[1:-1, 1:-2])/(dx)
    v[1:-1, 1:-1] = vs[1:-1, 1:-1] - dt/rho * \
        (p[2:-1, 1:-1]-p[1:-2, 1:-1])/(dy)

    return u, v, p, rhs


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

    # Interpolated values
    uc = avg(u, 1)  # horizontal average lives in cell centers
    vc = avg(v, 0)  # vertical average lives in cell centers
    ucorn = avg(u, 0)  # vertical average lives in cell corners
    vcorn = avg(v, 1)  # horizontal average lives in cell corners

    return u, v, uc, vc, ucorn, vcorn


def interpolatePressure(p, pWall):

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
    # Boundary corners interpolated with mean of x- and y- 2nd order backward
    p[0, 0] = (2*p[0, 1] - p[0, 2] + 2*p[1, 0] - p[2, 0]) / 2  # SW
    p[0, -1] = (2*p[0, -2] - p[0, -3] + 2*p[1, -1] - p[2, -1]) / 2  # SE
    p[-1, 0] = (2*p[-1, 1] - p[-1, 2] + 2*p[-2, 0] - p[-3, 0]) / 2  # NW
    p[-1, -1] = (2*p[-1, -2] - p[-1, -3] + 2*p[-2, -1] - p[-3, -1]) / 2  # NE

    # Pressure in corners
    pcorn = avg(avg(p, 0), 1)

    return p, pcorn


def calcDerived(Xf, Yf, p, u, v, dx, dy, nu, dt):

    # Velocity magnitudes
    Uc = (uc[1:-1, :]**2+vc[:, 1:-1]**2)**0.5
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

    return Uc, Ucorn, uMax, vMax, divU, Pe_u, Pe_v, CFL_u, CFL_v, Vis_x, Vis_y


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
p = np.zeros((ny+2, nx+2))
u = np.zeros((ny+2, nx+1))
v = np.zeros((ny+1, nx+2))
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
# Right hand sides
rhs_p = np.zeros((ny, nx))
rhs_u = np.zeros((ny, nx-1))
rhs_v = np.zeros((ny-1, nx))
# Right hand sides boundary part
rhs_p_bound = np.zeros((ny, nx))
rhs_u_bound = np.zeros((ny, nx-1))
rhs_v_bound = np.zeros((ny-1, nx))

# Generate system matrices------------------------------------------------------

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

while t < tMax:

    n += 1
    t += dt

    # Update variables
    pn, un, vn, us, vs = p.copy(), u.copy(), v.copy(), u.copy(), v.copy()

    # Projection method: Intermediate velocity field u*
    [us, vs] = solveMomentumNonLinearTerms(us, vs, un, vn, uc, vc, ucorn, vcorn,
                                           dt, dx, dy)
    [us, vs, rhs_u, rhs_v] = solveMomentumViscousTerms(
        us, vs, dt, dx, dy, nu, A_u, A_v, rhs_u, rhs_v,
        rhs_u_bound, rhs_v_bound, solver, tol_U, ml_u, ml_v, M_u, M_v)

    # Projection method: Pressure correction
    [u, v, p, rhs_p] = correctPressure(us, vs, p, A_p, rhs_p, rhs_p_bound, rho,
                                       dt, dx, dy, solver, tol_p, ml_p, M_p)

    # Interpolate velocity internal and boundary values for non-linear terms
    [u, v, uc, vc, ucorn, vcorn] = interpolateVelocities(u, v, uWall, vWall)

    # Print output step --------------------------------------------------------

    if (t-tOut) > -1e-6:

        # Calculate residuals
        Res_u = np.linalg.norm(rhs_u.ravel()-A_u*us[1:-1, 1:-1].ravel()) / \
            np.linalg.norm(rhs_u.ravel())
        Res_v = np.linalg.norm(rhs_v.ravel()-A_v*vs[1:-1, 1:-1].ravel()) / \
            np.linalg.norm(rhs_v.ravel())
        Res_p = np.linalg.norm(rhs_p.ravel()-A_p*p[1:-1, 1:-1].ravel()) / \
            np.linalg.norm(rhs_p.ravel())

        # Calculate derived quantities
        [Uc, Ucorn, uMax, vMax, divU,
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

    # Plot and save graphs and profiles ----------------------------------------

    if (t-tPlot) > -1e-6:

        # Interpolate pressure on corners and boundaries for plotting
        [p, pcorn] = interpolatePressure(p, pWall)

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
