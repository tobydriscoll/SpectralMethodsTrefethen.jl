# p6.jl - variable coefficient wave equation

# Grid, variable coefficient, and initial data:
N = 128
h = 2π/N
x = h*(1:N)
t,dt = 0.0,1.9/N
c = @. 0.2 + sin(x-1)^2
v = @. exp(-100*(x-1).^2)
vold = @. exp(-100*(x-0.2dt-1)^2)

# Time-stepping by leap frog formula:
tmax = 8
tplot = 0.15
plotgap = round(Int,tplot/dt)
dt = tplot/plotgap
ntime = round(Int,tmax/dt)
data = [ v zeros(N,ntime) ]
for i = 1:ntime
    global v
    global data
    v̂ = fft(v)
    ŵ = 1im*[0:N/2-1;0;-N/2+1:-1] .* v̂
    w = real(ifft(ŵ))
    global vold
    vnew = vold - 2dt*c.*w
    data[:,i+1] = vnew
    vold,v = v,vnew
    if norm(v,Inf) > 2.5
        data = data[:,1:i+1]
        break
    end
end
t = [ n*dt for n in 1:size(data,2) ]
plt = surface(x,t,data',camera=(10,70),clims=(-3,3),
    xaxis=("x",(0,2π)),yaxis=("t",(0,4)),zaxis=("u(x,t)",(-3,3)))
