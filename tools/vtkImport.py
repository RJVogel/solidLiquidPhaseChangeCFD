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
fig = plt.figure(figsize=figureSize, dpi=100)

# Axis
ax1 = fig.add_subplot(111)
ax1.set_aspect(1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Pressure contours and velocity vectors')
# Filled contours for pressure at faces

ctf1 = ax1.contourf(Xf, Yf, pf, 41, cmap=colormap1)
# Colorbar
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
cBar1 = fig.colorbar(ctf1, cax=cax1, extendrect=True)
cBar1.set_label('p / Pa')
# plot velocity vectors
m = 1
ax1.quiver(X[::m, ::m], Y[::m, ::m],
           uc[::m, ::m], vc[::m, ::m])

plt.tight_layout()
