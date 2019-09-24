# solidLiquidPhaseChangeCFD
Basic CFD simulation of thermally driven fluid flow and solid-liquid phase change problems

## Resources

Lorena Barba published the wonderful course on how to write a basic solver for the Navier-Stokes equations written in Python "12 steps to Navier Stokes". The course ends with solving a popular 2d test case, the lid-driven cavity.

https://github.com/barbagroup/CFDPython

Another brilliant course explaining linear algebra and CFD methods with examples written in Matlab is "Computational Science and Engineering I" by Gilbert Strang:

https://ocw.mit.edu/courses/mathematics/18-085-computational-science-and-engineering-i-fall-2008/index.htm

Many resources and examples can also be found on the web page of the book Computational Science and Engineering:

http://math.mit.edu/~gs/cse/

There is also a nice example code of a Navier-Stokes solver for the lid-driven cavity:

http://math.mit.edu/~gs/cse/codes/mit18086_navierstokes.pdf

## Three steps to solid-liquid phase change

The CFD method presented here starts with the lid-driven cavity flow problem in a first step. I started with a code similar to the one by Lorena Barba. However, I added some advanced features to enhance the accuracy and computation speed. In the second step, the code is extended to thermally driven fluid flow (natural convection). Finally, in the third step a model for solid-liquid phase change is added. The final code is able to solve a solid-liquid phase change problem with natural convection.

### Step 1: Navier-Stokes solver for Lid-driven flow in rectangular cavity

#### Test case

The test case is a rectangular cavity with a moving lid as depicted in the figure below.

![Alt text](/images/liddrivencavity.png "Lid-driven cavity test case.")

#### Code
The code is documented [here](https://nbviewer.jupyter.org/github/RJVogel/solidLiquidPhaseChangeCFD/blob/master/1-NavierStokes-LidDrivenCavity/1-NS-LDC.ipynb).

### Step 2: Natural convection in side heated and cooled cavity

work in progress ...

### Step 3: Solid-liquid phase change with natural convection in side heated cavity 

work in progress ...
