# p19.jl - 2nd-order wave eq. on Chebyshev grid (compare p6.jl)

# Time-stepping by leap frog formula:
N = 80
_,x = cheb(N)
dt = 8/N^2
v = @. exp(-200*x^2)
vold = @. exp(-200*(x-dt)^2)

tmax = 4
tplot = .0375
plotgap = round(Int,tplot/dt)
dt = tplot/plotgap
ntime = round(Int,tmax/dt)
anim = @animate for i = 0:ntime
    global v
    global vold
    str = @sprintf("t = %0.4f",i*dt)
    plot(x,v,xlim=(-1,1),ylim=(-1.1,1.1),title=str)
    
    w = chebfft(chebfft(v))
    w[[1,N+1]] .= 0
    vnew = 2v - vold + dt^2*w
    vold,v = v,vnew
end every plotgap
plt = gif(anim,"p19anim.gif",fps=15)
