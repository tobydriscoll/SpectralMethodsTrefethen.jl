# p35.m - Allen-Cahn eq. as in p34.m, but with boundary condition
#         imposed explicitly ("method (II)")

# Differentiation matrix and initial data:
N = 20
D,x = cheb(N)
D2 = D^2     # use full-size matrix
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
anim = @animate for i = 0:ntime
    global v
    str = @sprintf("t = %0.3f",i*dt)
    plot(xx,chebinterp(v).(xx),xlim=(-1,1),ylim=(-1,2),title=str)
    v += dt*(ϵ*D2*(v-x) + v - v.^3)           # Euler
end every plotgap
plt = gif(anim,"p34anim.gif",fps=15)
