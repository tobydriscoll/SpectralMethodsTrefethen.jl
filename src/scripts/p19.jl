# p19.jl - 2nd-order wave eq. on Chebyshev grid (compare p6.jl)

# Time-stepping by leap frog formula:
N = 80
_,x = cheb(N)
dt = 8/N^2
v = @. exp(-200*x^2)
vold = @. exp(-200*(x-dt)^2)

tmax = 4
tplot = .075
plotgap = round(Int,tplot/dt)
dt = tplot/plotgap
ntime = round(Int,tmax/dt)
data = hcat(v,zeros(N+1,ntime))
t = [ n*dt for n in 0:ntime ]
for i = 1:ntime
    global v
    global vold
    w = chebfft(chebfft(v))
    w[[1,N+1]] .= 0
    vnew = 2v - vold + dt^2*w
    data[:,i+1] = vnew
    vold,v = v,vnew
end

# Plot results:
plt = surface(x,t[1:plotgap:end],data[:,1:plotgap:end]',camera=(10,70),
  color=:balance,clims=(-1,1),
  xaxis=("x",(-1,1)),yaxis=("t",(0,tmax)),zaxis=("u(x,t)",(-1,1)) )
