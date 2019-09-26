# solidLiquidPhaseChangeCFD
Basic CFD simulation of thermally driven fluid flow and solid-liquid phase change problems

## Preface

The CFD simulation framework presented here builds upon learning ressources established by others. Lorena Barba published a wonderful course on how to write a basic solver for the Navier-Stokes equations written in Python: [12 steps to Navier Stokes](https://github.com/barbagroup/CFDPython). The course ends with solving a popular 2d test case, the lid-driven cavity. Another brilliant course explaining linear algebra and CFD methods with examples written in Matlab is [Computational Science and Engineering I](https://ocw.mit.edu/courses/mathematics/18-085-computational-science-and-engineering-i-fall-2008/index.htm) by Gilbert Strang. Many resources and examples can also be found on the web page of the book [Computational Science and Engineering](http://math.mit.edu/~gs/cse/). There is also a nice example code of a [Navier-Stokes solver for the lid-driven cavity](http://math.mit.edu/~gs/cse/codes/mit18086_navierstokes.pdf). All this learning material helped me to develop my own solver for solid-liquid phase change problems with natural convection. In the following, this solver is constructed in three steps.

## Three steps to solid-liquid phase change

The CFD method presented here starts with the lid-driven cavity flow problem in a first step. In the second step, the code is extended to thermally driven fluid flow (natural convection). Finally, in the third step a model for solid-liquid phase change is added. The final code is able to solve solid-liquid phase change problems with natural convection.

### Step 1: Navier-Stokes solver for Lid-driven flow in rectangular cavity

The Python code is found in [1-NS-LDC.py](https://github.com/RJVogel/solidLiquidPhaseChangeCFD/blob/master/1-NavierStokes-LidDrivenCavity/1-NS-LDC.py) and the documentation is given in the Jupyter Notebook [1-NS-LDC.ipynb](https://nbviewer.jupyter.org/github/RJVogel/solidLiquidPhaseChangeCFD/blob/master/1-NavierStokes-LidDrivenCavity/1-NS-LDC.ipynb).

<table>
    <caption> Pressure contours and velocity vectors of the lid-driven cavity with Re = 20 at steady state after time t = 5 s.
    </caption>
    <tr>
        <td><img src="/1-NavierStokes-LidDrivenCavity/out/liddrivencavity_Re20_5.000.png" width="500"/></td>
    </tr>
</table>


### Step 2: Natural convection in side heated and cooled cavity

work in progress ...

### Step 3: Solid-liquid phase change with natural convection in side heated cavity 

work in progress ...
