# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:06:34 2019

@author: Julian Vogel (RJVogel)
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
grid = pv.UnstructuredGrid('cavity_1000.vtk')

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
plt.close('all')
figureSize = (10, 6.25)
colormap1 = 'jet'
minContour1 = -5
maxContour1 = 5
minContour2 = -1
maxContour2 = 1
fig = plt.figure(figsize=figureSize, dpi=100)

# Plot pressure and velocity
# Axis
ax1 = fig.add_subplot(121)
ax1.set_aspect(1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Pressure contours and velocity vectors')
# Contours of pressure
plotContourLevels1 = np.linspace(minContour1, maxContour1, num=41)
ctf1 = ax1.contourf(Xf, Yf, pf, plotContourLevels1,
                    extend='both', cmap=colormap1)
# Colorbar
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
ticks1 = np.linspace(minContour1, maxContour1, num=7)
cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True, ticks=ticks1)
cBar1.set_label('p / Pa')
# plot velocity vectors
m = 1
ax1.quiver(X[::m, ::m], Y[::m, ::m],
           uc[::m, ::m], vc[::m, ::m])

# Plot divergence of velocity
# Axis
ax2 = fig.add_subplot(122)
ax2.set_aspect(1)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Divergence of velocity')
# Pcolor of divergence
plotContourLevels2 = np.linspace(minContour2, maxContour2, num=41)
ctf2 = ax2.contourf(X, Y, divU,
                    plotContourLevels2, extend='both', cmap=colormap1)
# Colorbar
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
ticks2 = np.linspace(minContour2, maxContour2, num=7)
cBar2 = fig.colorbar(ctf2, cax=cax2, extendrect=True, ticks=ticks2)
cBar2.set_label('div (U) / 1/s')

plt.tight_layout()
