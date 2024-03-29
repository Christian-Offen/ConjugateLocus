# ConjugateLocus

This GitHub Repository provides source code used fo numerical experiments in relation to the author's participation at the **7th IFAC Workshop on Lagrangian and Hamiltonian Methods for Nonlinear Control 2021** (<https://lhmnlc21.org/>)

## Computation of cusps and hyperbolic umbilics
Run *./variational/Locus3D_Cusps.ipynb* to compute the two closed lines of cusps in the bifurcation diagram as well as the location of the four umbilic singularities.

## Computation of full bifurcation diagram
Run *./variational/ComputeDataIsoSurf_FullLocus.ipynb* and then *./variational/Isosurface.m* to compute preimages of the sheets of folds of the bifurcation diagram. Run *./variational/MapIsoDataLocus.py* to compute data of the conjugate locus and *./variational/LocusIsosurface.m* to compute plots of these.

## Computation of bifurcation diagram near a hyperbolic umbilic singularity
Run *./variational/UmbilicIsosurface_ComputeData.ipynb* and then *./variational/UmbilicIsosurface_CriticalSet.m*  to compute preimages of the sheet of folds. Then continue with the last lines in *./variational/UmbilicIsosurface_ComputeData.ipynb*. Finally, plot the locus with *./variational/UmbilicIsosurface_PlotLocus.m*

All plots and figures are written to the folder *./variational/Plots.* Computed data is written to *./variational/Data*.

## Alternative methods
For experiments with a non-variational method, use the corresponding scripts in subfolders of *./RK/* rather than *./variational/*
