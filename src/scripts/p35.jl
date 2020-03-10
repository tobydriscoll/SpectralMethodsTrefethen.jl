# p35.jl - Allen-Cahn eq. as in p34, but with boundary condition
#          imposed explicitly ("method (II)")

# Differentiation matrix and initial data:
N = 20
D,x = cheb(N)
D2 = D^2     # use full-size matrix
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
    v += dt*(ϵ*D2*(v-x) + v - v.^3)         # Euler
    v[1],v[end] = 1 + sin(t[i]/5)^2, -1        # BC
    data[:,i+1] = chebinterp(v).(xx)
end
plt = surface(xx,t[1:plotgap:end],data[:,1:plotgap:end]',cam=(30,50),color=:balance,
    clims=(-2,2),xaxis="x",yaxis="t",zaxis=("u(x,t)",(-1.05,2.05)) )
