from EndPtJAX import Locus
import jax.numpy as jnp

#import numpy as np
import scipy.io         # Matlab export
#import pickle


# Setup of problem on 3d ellipsoid in R4

n = 4                              # dimension outer space
b = jnp.array([0.9,1.2,1.6,1])     # ellipsoid coefficients
T = 1                              # time
N = 10                             # steps
dt = T/N                           # discretisation parameter
XStart = jnp.array([0.1,0.05,0.2]) # start point of geodesic map

l3 = Locus(n,b,T,N,XStart)         # Create 3d geodesic problem (instance of class Locus)


matdata=scipy.io.loadmat('Data/isodata.mat')
verts=matdata['verts']
vertsLocus=list(map(l3.endptChart,verts))
print('Computation completed. Now writing to file.')
scipy.io.savemat('Data/LocusVerts.mat', dict(LocusVerts=vertsLocus)) # Matlab export
