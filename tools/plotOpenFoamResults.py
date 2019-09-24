# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:06:34 2019

@author: Julian Vogel (RJVogel)
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configuration
saveFiguresToFile = True
outputFilename = 'openFoam_liddrivencavity_Re20'
t = 1
plt.close('all')
figureSize = (6, 6)
plotLevels1 = (-5, 5)
plotLevels2 = (-1, 1)
colormap1 = 'bwr'
colormap2 = 'seismic'
plotEveryMthVector = 1
plotDivergence = False


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


# grid is the central object in VTK where every field is added on to grid
grid = pv.UnstructuredGrid('liddrivencavity_1000.vtk')

# grid points
points = grid.points
points2D = np.array(points[points[:, 2] == 0][:, :2])

# get unique grid values
xf = np.unique(points2D[:, 0])
yf = np.unique(points2D[:, 1])
Xf, Yf = np.meshgrid(xf, yf)
x = avg(xf, 0)
y = avg(yf, 0)
X, Y = np.meshgrid(x, y)

# Create indexing matrix
indices = np.ones((len(yf), len(xf)), dtype=int)
for j in range(len(yf)):
    for i in range(len(xf)):
        indices[j, i] = int(np.array(np.where(np.all(points2D == (xf[i], yf[j]),
                            axis=1))))
# indices = np.reshape(indices, len(xf)*len(yf), 0)

# get fields
p0 = grid.point_arrays['p'][points[:, 2] == 0]
u0 = grid.point_arrays['U'][points[:, 2] == 0][:, 0]
v0 = grid.point_arrays['U'][points[:, 2] == 0][:, 1]

# Write data into new fields
pf = np.zeros(len(indices), dtype=float)
pf = p0[indices]
pf = np.reshape(pf, (len(yf), len(xf)))

uf = np.zeros(len(indices), dtype=float)
uf = u0[indices]
uf = np.reshape(uf, (len(yf), len(xf)))

vf = np.zeros(len(indices), dtype=float)
vf = v0[indices]
vf = np.reshape(vf, (len(yf), len(xf)))

# Cell centre values
pc = avg(avg(pf, 0), 1)
uc = avg(avg(uf, 0), 1)
vc = avg(avg(vf, 0), 1)

# Divergence
divU = np.diff(avg(uf, 0), axis=1)/np.diff(Xf[1:, :], axis=1) + \
    np.diff(avg(vf, 1), axis=0)/np.diff(Yf[:, 1:], axis=0)

# Plot

# Figure
fig = plt.figure(figsize=figureSize, dpi=100)

# Levels
if plotLevels1 == (None, None):
    plotLevels1 = (np.min(pf)-1e-12, np.max(pf)+1e-12)
if plotLevels2 == (None, None):
    plotLevels2 = (np.min(divU)-1e-12, np.max(divU)+1e-12)

plotContourLevels1 = np.linspace(plotLevels1[0], plotLevels1[1], num=40)
plotContourLevels2 = np.linspace(plotLevels2[0], plotLevels2[1], num=40)

m = plotEveryMthVector

# Create figure
fig = plt.figure(figsize=figureSize, dpi=100)
# Create axis 1
# Axis
if plotDivergence:
    ax1 = fig.add_subplot(121)
else:
    ax1 = fig.add_subplot(111)
ax1.set_aspect(1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('OpenFOAM: Pressure contours and velocity vectors')

# Contours of pressure
ctf1 = ax1.contourf(Xf, Yf, pf, plotContourLevels1,
                    extend='both', cmap=colormap1)

# Colorbar 1
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
ticks1 = np.linspace(plotLevels1[0], plotLevels1[1], num=11)
cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True, ticks=ticks1)
cBar1.set_label('p / Pa')

# plot velocity vectors
ax1.quiver(X[::m, ::m], Y[::m, ::m],
           uc[::m, ::m]+1e-12, vc[::m, ::m]+1e-12)

if plotDivergence:
    # Create axis 2
    # Axis
    ax2 = fig.add_subplot(122)
    ax2.set_aspect(1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('OpenFOAM: Divergence of velocity')

    # Contours of divergence
    ctf2 = ax2.contourf(X, Y, divU,
                        plotContourLevels2, extend='both',
                        cmap=colormap2)

    # Colorbar 2
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    ticks2 = np.linspace(plotLevels2[0], plotLevels2[1], num=11)
    cBar2 = fig.colorbar(ctf2, cax=cax2, extendrect=True, ticks=ticks2)
    cBar2.set_label('div (U) / 1/s')

if saveFiguresToFile:
    formattedFilename = '{0}_{1:5.3f}.png'.format(outputFilename, t)
    path = formattedFilename
    fig.savefig(path)

plt.tight_layout()
plt.show()
