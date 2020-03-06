# p34.jl - Allen-Cahn eq. u_t = eps*u_xx+u-u^3, u(-1)=-1, u(1)=1
#         (compare p6.jl and p32.jl)

# Differentiation matrix and initial data:
N = 20
D,x = cheb(N)
D2 = D^2      # use full-size matrix
D2[[1,N+1],:] .= 0                     # for convenience
ϵ = 0.01
t,dt = 0,min(.01,50/(N^4*ϵ))
v = @. 0.53*x + 0.47*sin(-1.5π*x)

# Solve PDE by Euler formula and plot results:
tmax = 100
tplot = 2
nplots = round(Int,tmax/tplot)
plotgap = round(Int,tplot/dt)
dt = tplot/plotgap
ntime = round(Int,tmax/dt)
xx = -1:.025:1
vv = chebinterp(v).(xx);
data = [vv zeros(length(vv),ntime)]
t = [ n*dt for n in 0:ntime ]
for i = 1:ntime
    global v
    v += dt*(ϵ*D2*(v-x) + v - v.^3)    # Euler
    data[:,i+1] = chebinterp(v).(xx)
end
plt = surface(xx,t[1:plotgap:end],data[:,1:plotgap:end]',cam=(30,50),color=:balance,
  xaxis="x",yaxis="t",zaxis=("u(x,t)",(-1.05,1.05)) )
