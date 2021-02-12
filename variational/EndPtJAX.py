# Endpoint map geodesic on (n-1)-dimensional ellipsoid in Rn
# With Jacobian

from jax import ops, lax, jacfwd, jit
import jax.numpy as jnp
from scipy import linalg, optimize
from functools import partial

class Locus:


	def __init__(self, n,b,T,N,XStart):
		self.n = n		# dimension of ambient space
		self.b = b		# ellipsoid coefficients
		self.T = T		# time
		self.N = N		# steps
		self.dt = T/N		# discretisation parameter
		self.XStart = XStart	# start point of geodesic map

		# sample values for 2d ellipsoid in R3
		#n = 3				
		#b = jnp.array([0.9,1.2,1.6])	# ellipsoid coefficients
		#T = 1				# time
		#N=10				# steps
		#dt=T/N			# discretisation parameter
		#XStart = jnp.array([0.1,0.05]) # start point of geodesic map

	#level function
	def g(self,x):
		return sum(self.b*(x**2))-1

	def dg(self,x):
		return 2*x*self.b

	# 1 step map from z=(q_{k-1},q_k) to q_{k+1}
	def Scheme(self,z):
		qn = z[:self.n]
		q  = z[self.n:]

		# compute Lagrange multipliers
		den = self.dt**2*jnp.dot(self.b**3,q**2)
		dff = 2*q-qn
		m1 = jnp.dot(self.b**2*q,dff)/den
		m2 = 1/self.dt**2*(jnp.dot(self.b*dff,dff)-1)/den
		lam = -m1 + jnp.sqrt(m1**2-m2)

		return jnp.block([q,2*q-qn+self.dt**2*self.b*q*lam])

	# Chart for ellipsoid - projection to tangent space of XStart and its antipodal

	def xC2(self,X):
		return (1-sum(self.b[:-1]*(X**2)))/self.b[-1]

	def chartS(self,X):
		return jnp.block([X,jnp.sqrt(self.xC2(X))])

	def chartF(self,X):
		return jnp.block([X,-jnp.sqrt(self.xC2(X))])


	def DchartS(self,X):
		return jnp.block([[jnp.identity(self.n-1)], [-self.b[:-1]*X/(self.b[-1]*jnp.sqrt(self.xC2(X)))]])

	def DchartF(self,X):
		return jnp.block([[jnp.identity(self.n-1)], [self.b[:-1]*X/(self.b[-1]*jnp.sqrt(self.xC2(X)))]])

	def chartSInv(self,X):
		return X[:-1]

	def chartFInv(self,X):
		return X[:-1]

	def DchartSInv(self,X):
		return jnp.identity(self.n+1)[:-1]

	def DchartFInv(self,X):
		return jnp.identity(self.n+1)[:-1]

	# Endpoint map: computes endpoint of geodesic that connects z[:n] with z[n:] in time dt
	def endpt(self,z):
		return lax.fori_loop(2,self.N+1,lambda k,x: self.Scheme(x),z)

	# Endpoint map at chartS(XStart) expressed in charts 
	@partial(jit, static_argnums=(0,))
	def endptChart(self,x):
		return self.chartFInv(self.endpt(jnp.block([self.chartS(self.XStart),self.chartS(x)]))[self.n:])

	# for finding critical points of endpoint map
	@partial(jit, static_argnums=(0,))
	def LocusChart(self,x):
		return jnp.linalg.det(jacfwd(self.endptChart)(x))


# Pseudo-arclength continuation of codim 1 valued map g
def ContFun(xoldold,xold,g,ds):
	gold = g(xold)
	#dgfun = jacfwd(g)
	#dg = dgfun(xold)
	dg = jacfwd(g)(xold)
	n = xold.shape[0]
	if len(dg.shape)==1:
		dg=dg.reshape(1,n)
	v = jnp.transpose(linalg.null_space(dg))
	v0=jnp.sign(jnp.dot(v,xold-xoldold))*v/jnp.linalg.norm(v)
	v0=v0.flatten()
	xpred = xold+ds*v0
	
	obj = lambda y: jnp.block([g(y),jnp.dot(y-xpred,v0)])
		
	#Dobj = lambda y: jnp.block([[dgfun(y)],[v0]])

	return optimize.fsolve(obj,xpred,fprime=jacfwd(obj),xtol=1e-4)	


