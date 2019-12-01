# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:20:40 2016

@author: Julian Vogel (RJVogel)

Navier Stokes solver for lid driven cavity flow
"""


# Libraries ====================================================================

import numpy as np
import matplotlib.pyplot as plt
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
dt = 0.005

# Solver settings
nit = 50  # iterations of pressure poisson equation

# Upwind discretization
gamma = 1.


# Momentum equations -----------------------------------------------------------

# Initial conditions
p0 = 0.
u0 = 0.
v0 = 0.

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

# Material properties

# Density
rho = 1.0
# Kinematic viscosity
mu = 0.05


# Visualization ----------------------------------------------------------------

# Output control

# Print step
dtOut = max(dt, tMax/10)
# Plot step
dtPlot = tMax

# Plots

# Figure size on screen
figureSize = (7, 6)
# Use inline graphics
inlineGraphics = True

# Plot definition
plotContourVar = 'p'  # 'p', u, v, 'U', 'divU'
plotFaceValues = True
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


def solveMomentumEquation(
        u, v, uc, vc, ucorn, vcorn, dt, dx, dy, nu, gamma):

    us, vs = u.copy(), v.copy()

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

    # Diffusive terms
    Du = nu * (
        1/dx**2*(u[1:-1, 2:]-2*u[1:-1, 1:-1]+u[1:-1, :-2]) +
        1/dy**2*(u[2:, 1:-1]-2*u[1:-1, 1:-1]+u[:-2, 1:-1]))
    Dv = nu * (
        1/dx**2*(v[1:-1, 2:]-2*v[1:-1, 1:-1]+v[1:-1, :-2]) +
        1/dy**2*(v[2:, 1:-1]-2*v[1:-1, 1:-1]+v[:-2, 1:-1]))

    # Update velocities
    us[1:-1, 1:-1] = u[1:-1, 1:-1] - dt*Fu + dt*Du
    vs[1:-1, 1:-1] = v[1:-1, 1:-1] - dt*Fv + dt*Dv

    return us, vs


def correctPressure(us, vs, p, rho, dt, dx, dy, rhs_p, pWall, nit):

    # Interpolations
    uahv = avg(avg(us, 1), 0)
    vavh = avg(avg(vs, 0), 1)

    # Right hand side for iterative solution
    rhs_p = rho*(
        1/dt*((u[1:-1, 1:] - u[1:-1, :-1])/dx +
              (v[1:, 1:-1] - v[:-1, 1:-1])/dy) -
        (((u[1:-1, 1:] - u[1:-1, :-1])/dx)**2 +
         2*((uahv[1:, :] - uahv[:-1, :])/dy *
            (vavh[:, 1:] - vavh[:, :-1])/dx) +
         ((v[1:, 1:-1] - v[:-1, 1:-1])/dy)**2))

    # Solve iteratively for pressure
    for nit in range(50):

        # Force constant pressure in SW corner
        p[1, 1] = 0

        # Iterative solution of pressure
        p[1:-1, 1:-1] = (dy**2*(p[1:-1, 2:]+p[1:-1, :-2]) +
                         dx**2*(p[2:, 1:-1]+p[:-2, 1:-1]) -
                         rhs_p*dx**2*dy**2)/(2*(dx**2+dy**2))

        # Interpolate solution on boundaries
        p = interpolateCellCenteredOnBoundary(p, pWall)

    # Pressure gradients
    dpdx = - dt/rho * (p[1:-1, 2:-1]-p[1:-1, 1:-2])/dx
    dpdy = - dt/rho * (p[2:-1, 1:-1]-p[1:-2, 1:-1])/dy

    # Project corrected pressure onto velocity field
    u[1:-1, 1:-1] = us[1:-1, 1:-1] + dpdx
    v[1:-1, 1:-1] = vs[1:-1, 1:-1] + dpdy

    return u, v, p


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


def calcDerived(Xf, Yf, p, u, v, dx, dy, nu, dt):

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

    return Uc, Ucorn, uMax, vMax, divU, Pe_nu_u, Pe_nu_v, CFL_u, CFL_v, \
        Vis_x, Vis_y


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
# Material properties
nu = mu/rho
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


# Time stepping ================================================================

twall0 = time.time()
tOut = dtOut
tPlot = dtPlot
fig, ax = None, None
t, n = 0, 0

while t - tMax < -1e-9:

    n += 1
    t += dt

    # Update variables
    pn = p.copy()
    un, vn = u.copy(), v.copy()

    # Projection method: Intermediate velocity field U*
    [us, vs] = solveMomentumEquation(
        un, vn, uc, vc, ucorn, vcorn, dt, dx, dy, nu, gamma)

    # Projection method: Pressure correction
    [u, v, p] = correctPressure(us, vs, p, rho, dt, dx, dy, rhs_p, pWall, nit)

    # Interpolate velocity on internal & boundary values f. non-linear terms
    [u, v, uc, vc, ucorn, vcorn] = interpolateVelocities(u, v, uWall, vWall)

    # Print output step --------------------------------------------------------

    if (t-tOut) > -1e-6:

        # Calculate derived quantities
        [Uc, Ucorn, uMax, vMax, divU, Pe_nu_u, Pe_nu_v,
         CFL_u, CFL_v, Vis_x, Vis_y] = \
            calcDerived(Xf, Yf, p, u, v, dx, dy, nu, dt)

        print("==============================================================")
        print("Time step n = %d, t = %8.3f, dt = %4.1e, t_wall = %4.1f" %
              (n, t, dt, time.time()-twall0))
        print("max|u| = %4.2e, CFL(u) = %5.2f, "
              "Pe(u) = %5.2f, Vis(x) = %5.2f" %
              (uMax, CFL_u, Pe_nu_u, Vis_x))
        print("max|v| = %4.2e, CFL(v) = %5.2f, "
              "Pe(v) = %5.2f, Vis(y) = %5.2f" %
              (vMax, CFL_v, Pe_nu_v, Vis_y))
        print("ddt(p) = %5.2e, ddt(u) = %5.2e, ddt(v) = %5.2e" %
              (np.linalg.norm((p-pn)/dt),
               np.linalg.norm((u-un)/dt),
               np.linalg.norm((v-vn)/dt)))

        tOut += dtOut

    # Plot and save graphs and profiles ----------------------------------------

    if (t-tPlot) > -1e-6:

        # Interpolate pressure on corners and boundaries for plotting
        p = interpolateCellCenteredOnBoundary(p, pWall)
        [p, pcorn] = interpolateCellCenteredOnCorners(p, pWall)

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
