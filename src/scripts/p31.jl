# p31.jl - gamma function via complex integral, trapezoid rule

N = 70
θ = -π .+ (2*π/N)*(.5:N-.5)
c,r = -11,16                 # center,radius of circle of integration
x,y = -3.5:.1:4,-2.5:.1:2.5
gaminv = zeros(length(y),length(x))
for i = 1:N
    t = c + r*exp(1im*θ[i])
    global gaminv += [ exp(t)*t^(-x-1im*y)*(t-c) for y in y, x in x ]
end
Γ = N./gaminv
plt = wireframe(x,y,abs.(Γ),color=:darkblue,camera=(-30,30),
  xaxis="Re(z)",yaxis="Im(z)",zaxis=(L"|\,\Gamma(z)\,|",(0,6)))
