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
tmax = 15
tplot = 0.15
plotgap = round(Int,tplot/dt)
dt = tplot/plotgap
ntime = round(Int,tmax/dt)
anim = @animate for i = 0:ntime
    global v
    global vold
    str = @sprintf("t = %0.2f",i*dt)
    plot(x,v,xlim=(0,2π),ylim=(-0.1,1.1),title=str)
    v̂ = fft(v)
    ŵ = 1im*[0:N/2-1;0;-N/2+1:-1] .* v̂
    w = real(ifft(ŵ))
    vnew = vold - 2dt*c.*w
    vold,v = v,vnew
end every plotgap
plt = gif(anim,"p6anim.gif",fps=15)
