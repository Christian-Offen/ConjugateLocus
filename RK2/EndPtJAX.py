# Endpoint map geodesic on (n-1)-dimensional ellipsoid in Rn
# With Jacobian

from jax import ops, lax, jacfwd, jit, jvp
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

	# rhs of equation in 1st order formulation
	def F(self,z,lam):
		return jnp.block([z[self.n:],-1/2*self.dg(z[:self.n])*lam])

	# 1 step with constrained RK2
	def RK2Constr(self,z,lam):
		return z+self.dt*self.F(z+1/2*self.dt*self.F(z,lam),lam)

	# 1 step map
	def RK2(self,z):
		q = z[:self.n]
		p = z[self.n:]

		# compute Lagrange multipliers
		den = self.dt**2*jnp.dot(self.b**3,q**2)
		m1 = 2*jnp.dot(self.b**2*q,q+self.dt*p)/den
		m2 = 4*jnp.dot(self.b,p**2)/den
		lam = m1 - jnp.sqrt(m1**2-m2)

		return self.RK2Constr(z,lam)

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

	# Endpoint map	
	def endpt(self,z):
		return lax.fori_loop(1,self.N,lambda k,x: self.RK2(x),z)

	# Endpoint map at chartS(XStart) maps tangent space to ellipsoid composed. Composed with chart
	@partial(jit, static_argnums=(0,))
	def endptChart(self,p):
		return self.chartFInv(self.endpt(jnp.block([self.chartS(self.XStart),jnp.matmul(self.DchartS(self.XStart),p)]))[:self.n])

	# for finding critical points of endpoint map
	@partial(jit, static_argnums=(0,))
	def LocusChart(self,p):
		return jnp.linalg.det(jacfwd(self.endptChart)(p))


# Pseudo-arclength continuation of codim 1 valued map g
def ContFun(xoldold,xold,g,ds):
	gold = g(xold)
	dg = jacfwd(g)(xold)
	n = xold.shape[0]
	if len(dg.shape)==1:
		dg=dg.reshape(1,n)
	v = jnp.transpose(linalg.null_space(dg))
	v0=jnp.sign(jnp.dot(v,xold-xoldold))*v/jnp.linalg.norm(v)
	v0=v0.flatten()
	xpred = xold+ds*v0
	def obj(y):
		return jnp.block([g(y),jnp.dot(y-xpred,v0)])

	return optimize.fsolve(obj,xpred,xtol=1e-6)	


@partial(jit, static_argnums=(0,))
def cuspCond(f1,Xa,ds):
    
    # shorthands
    x = Xa[:3]
    a = Xa[3:]
    
    f2 = lambda x: jvp(f1,(x,),(a,))[1] # 1st derivative in direction a
    c1 = f2(x)
    c2 = (sum(a**2)-1)/ds
    f3 = lambda x: jvp(f2,(x,),(a,))[1] # 2nd derivative in direction a
    c3 = jnp.matmul(f3(x),a)
    
    return jnp.block([c1, c2, c3])

@partial(jit, static_argnums=(0,))
def SWCond(f1,Xa):
    
    # shorthands
    x = Xa[:3]
    a = Xa[3:]
    
    Jac = jacfwd(f1)(x)
    
    f2 = lambda x: jvp(f1,(x,),(a,))[1] # 1st derivative in direction a
    f3 = lambda x: jvp(f2,(x,),(a,))[1] # 2nd derivative in direction a
    f4 = lambda x: jvp(f3,(x,),(a,))[1] # 3rd derivative in direction a
    
    # consistent solution to v=jnp.linalg.solve(Jac,-f3(x))
    b = -f3(x)
    vbar = jnp.linalg.solve(jnp.matmul(Jac,jnp.transpose(Jac))+jnp.matmul(a,jnp.transpose(a)),b)
    v = jnp.matmul(jnp.transpose(Jac),vbar)
    
    sw = jnp.matmul(f4(x),a) - 3*jnp.matmul(v,b)
    
    return sw

@partial(jit, static_argnums=(0,))
def DCond(f1,p):
    #f1=self.endptChart
    Jac=jacfwd(f1)(p)
    return -Jac[0, 1]*Jac[1, 0]+Jac[0, 0]*Jac[1, 1]-Jac[0, 2]*Jac[2, 0]-Jac[1, 2]*Jac[2, 1]+Jac[0, 0]*Jac[2, 2]+Jac[1, 1]*Jac[2, 2] # trace of 2nd exterior power
    
    
def CuspAndDCond(f1,Xa,ds):
    
    c = cuspCond(f1,Xa,ds)
    det2 = DCond(f1,Xa[:3])
    
    return jnp.block([c,det2])
