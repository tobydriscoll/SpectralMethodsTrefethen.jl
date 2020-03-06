# p6.jl - variable coefficient wave equation

# Grid, variable coefficient, and initial data:
N = 128
h = 2π/N
x = h*(1:N)
t,dt = 0.0,h/4
c = @. 0.2 + sin(x-1)^2
v = @. exp(-100*(x-1).^2)
vold = @. exp(-100*(x-0.2dt-1)^2)

# Time-stepping by leap frog formula:
tmax = 8
tplot = 0.15
plotgap = round(Int,tplot/dt)
dt = tplot/plotgap
ntime = round(Int,tmax/dt)
data = [v zeros(N,ntime)]
t = [ n*dt for n in 0:ntime ]
for i = 1:ntime
    global v
    global vold
    v̂ = fft(v)
    ŵ = 1im*[0:N/2-1;0;-N/2+1:-1] .* v̂
    w = real(ifft(ŵ))
    vnew = vold - 2dt*c.*w
    data[:,i+1] = vnew
    vold,v = v,vnew
end
plt = surface(x,t[1:plotgap:end],data[:,1:plotgap:end]',camera=(10,70),
    xaxis=("x",(0,2π)),yaxis=("t",(0,tmax)),zaxis=("u(x,t)",(0,1.5)))
